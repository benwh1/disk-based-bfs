use std::{
    collections::HashMap,
    sync::{Mutex, RwLock},
};

use crate::{
    io::LockedIO,
    settings::BfsSettings,
    update::{
        block_list::UpdateBlockList,
        blocks::{AvailableUpdateBlock, FilledUpdateBlock},
    },
};

pub struct UpdateManager<'a> {
    settings: &'a BfsSettings,
    locked_io: &'a LockedIO<'a>,
    sizes: RwLock<HashMap<usize, Vec<u64>>>,
    size_file_lock: Mutex<()>,
    update_blocks: Mutex<UpdateBlockList<'a>>,
}

impl<'a> UpdateManager<'a> {
    pub fn new(settings: &'a BfsSettings, locked_io: &'a LockedIO) -> Self {
        Self {
            settings,
            locked_io,
            sizes: RwLock::new(HashMap::new()),
            size_file_lock: Mutex::new(()),
            update_blocks: Mutex::new(UpdateBlockList::new(settings, locked_io)),
        }
    }

    /// Note: the sizes are not guaranteed to be *exactly* correct, because it's possible that we
    /// could write some update files to disk and then the program is terminated before we can write
    /// the sizes. This isn't important though, because the sizes are only used to determine if we
    /// should compress the update files, and it really doesn't matter if we sometimes compress them
    /// slightly before or after the actual size reaches the threshold.
    fn write_sizes_to_disk(&self) {
        let _lock = self.size_file_lock.lock().unwrap();

        let read_lock = self.sizes.read().unwrap();
        let str = serde_json::to_string(&*read_lock).unwrap();
        drop(read_lock);

        let path = self.settings.update_files_size_file_path();
        self.locked_io.write_file(&path, str.as_ref());
    }

    pub fn try_read_sizes_from_disk(&self) {
        let path = self.settings.update_files_size_file_path();

        let str = self.locked_io.try_read_to_string(&path).unwrap();
        let hashmap = serde_json::from_str(&str).unwrap();

        let mut lock = self.sizes.write().unwrap();
        *lock = hashmap;
    }

    pub fn take_n(&self, n: usize) -> Vec<AvailableUpdateBlock> {
        let mut update_blocks_lock = self.update_blocks.lock().unwrap();

        tracing::debug!(
            "taking {n} update blocks, {} blocks remaining",
            update_blocks_lock.num_available_blocks(),
        );

        let mut total_bytes_written = HashMap::new();

        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            let (block, bytes_written) = update_blocks_lock.take_impl(false);
            blocks.push(block);

            // Sum up the total number of bytes written across all calls to `take_impl`
            for (depth, vec) in bytes_written {
                total_bytes_written
                    .entry(depth)
                    .or_insert_with(|| vec![0; self.settings.num_array_chunks()])
                    .iter_mut()
                    .zip(vec.iter())
                    .for_each(|(total, new)| *total += new);
            }
        }

        // Update `self.sizes` with the new bytes written
        let mut sizes_lock = self.sizes.write().unwrap();
        for (depth, vec) in total_bytes_written {
            sizes_lock
                .entry(depth)
                .or_insert_with(|| vec![0; self.settings.num_array_chunks()])
                .iter_mut()
                .zip(vec.iter())
                .for_each(|(total, new)| *total += new);
        }

        drop(sizes_lock);

        self.write_sizes_to_disk();

        blocks
    }

    pub fn put(&self, block: FilledUpdateBlock) {
        self.update_blocks.lock().unwrap().put(block);
    }

    pub fn flush(&self) {
        let bytes_written = self.update_blocks.lock().unwrap().write_all();

        let mut sizes_lock = self.sizes.write().unwrap();
        for (depth, vec) in bytes_written {
            sizes_lock
                .entry(depth)
                .or_insert_with(|| vec![0; self.settings.num_array_chunks()])
                .iter_mut()
                .zip(vec.iter())
                .for_each(|(total, new)| *total += new);
        }

        drop(sizes_lock);

        self.write_sizes_to_disk();
    }

    fn delete_update_files_impl(&self, depth: usize, chunk_idx: usize, delete_used_only: bool) {
        if delete_used_only {
            tracing::debug!("deleting used update files for depth {depth} chunk {chunk_idx}");
        } else {
            tracing::debug!("deleting update files for depth {depth} chunk {chunk_idx}");
        }

        let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
        let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
            return;
        };

        let mut bytes_deleted = 0;

        for path in read_dir.flatten().map(|entry| entry.path()) {
            if !delete_used_only || path.extension().and_then(|ext| ext.to_str()) == Some("used") {
                bytes_deleted += path.metadata().unwrap().len();
                self.locked_io.queue_deletion(path);
            }
        }

        self.sizes
            .write()
            .unwrap()
            .entry(depth)
            .and_modify(|entry| entry[chunk_idx] = entry[chunk_idx].saturating_sub(bytes_deleted));

        self.write_sizes_to_disk();
    }

    pub fn delete_update_files(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_files_impl(depth, chunk_idx, false);
    }

    pub fn delete_used_update_files(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_files_impl(depth, chunk_idx, true);
    }

    pub fn files_size(&self, depth: usize, chunk_idx: usize) -> u64 {
        let read_lock = self.sizes.read().unwrap();
        read_lock.get(&depth).map_or(0, |sizes| sizes[chunk_idx])
    }

    pub fn mark_filled_and_replace(
        &self,
        upd: &mut AvailableUpdateBlock,
        depth: usize,
        chunk_idx: usize,
    ) {
        tracing::debug!(
            "marking update block as filled, contains {}/{} values",
            upd.len(),
            upd.capacity(),
        );

        let (new, bytes_written) = self.update_blocks.lock().unwrap().take();
        let old = std::mem::replace(upd, new).into_filled(depth, chunk_idx);
        self.put(old);

        // Update `self.sizes` with the new bytes written
        let mut sizes_lock = self.sizes.write().unwrap();
        for (depth, vec) in bytes_written {
            sizes_lock
                .entry(depth)
                .or_insert_with(|| vec![0; self.settings.num_array_chunks()])
                .iter_mut()
                .zip(vec.iter())
                .for_each(|(total, new)| *total += new);
        }

        drop(sizes_lock);

        self.write_sizes_to_disk();
    }
}
