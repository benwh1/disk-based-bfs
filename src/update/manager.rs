use std::{
    collections::HashMap,
    sync::{Mutex, RwLock},
    thread::Builder as ThreadBuilder,
};

use itertools::Itertools;
use rand::distributions::{Alphanumeric, DistString as _};

use crate::{
    chunk_allocator::ChunkAllocator,
    io::LockedIO,
    settings::BfsSettings,
    update::blocks::{AvailableUpdateBlock, FilledUpdateBlock},
};

pub struct UpdateManager<'a, C: ChunkAllocator + Sync> {
    settings: &'a BfsSettings<C>,
    locked_io: &'a LockedIO<'a, C>,
    sizes: RwLock<HashMap<usize, Vec<u64>>>,
    size_file_lock: Mutex<()>,
    available_blocks: Mutex<Vec<AvailableUpdateBlock>>,
    filled_blocks: Mutex<Vec<FilledUpdateBlock>>,
}

impl<'a, C: ChunkAllocator + Sync> UpdateManager<'a, C> {
    pub fn new(settings: &'a BfsSettings<C>, locked_io: &'a LockedIO<C>) -> Self {
        let num_blocks = 2 * settings.threads * settings.num_array_chunks();

        tracing::debug!("creating {num_blocks} update blocks");

        let available_blocks = Mutex::new(
            (0..num_blocks)
                .map(|_| AvailableUpdateBlock::new(settings.update_capacity_per_vec()))
                .collect::<Vec<_>>(),
        );
        let filled_blocks = Mutex::new(Vec::new());

        Self {
            settings,
            locked_io,
            sizes: RwLock::new(HashMap::new()),
            size_file_lock: Mutex::new(()),
            available_blocks,
            filled_blocks,
        }
    }

    pub fn write_all(&self) {
        let mut filled_blocks_lock = self.filled_blocks.lock().unwrap();
        let mut filled_blocks = std::mem::take(&mut *filled_blocks_lock);
        drop(filled_blocks_lock);

        if filled_blocks.is_empty() {
            return;
        }

        tracing::info!("writing {} update blocks", filled_blocks.len());

        let num_disks = self.locked_io.num_disks();

        // Sort all the blocks into chunks that will be written to the same disk so that each chunk
        // can be processed by a separate thread
        let mut chunked_filled_blocks = (0..num_disks).map(|_| Vec::new()).collect::<Vec<_>>();
        for block in filled_blocks.drain(..) {
            let chunk_root_idx = self
                .settings
                .chunk_allocator
                .chunk_root_idx(block.chunk_idx());
            chunked_filled_blocks[chunk_root_idx].push(block);
        }

        // Sort each chunk by depth and chunk_idx so that blocks that can be written to the same
        // file are consecutive in the list
        for chunk in chunked_filled_blocks.iter_mut() {
            chunk.sort_unstable_by_key(|block| (block.depth(), block.chunk_idx()));
        }

        std::thread::scope(|s| {
            (0..num_disks)
                .map(|t| {
                    let filled_blocks = std::mem::take(&mut chunked_filled_blocks[t]);
                    let available_blocks = &self.available_blocks;

                    ThreadBuilder::new()
                        .name(format!("write-update-blocks-{t}"))
                        .spawn_scoped(s, move || {
                            for (_, chunk) in &filled_blocks
                                .into_iter()
                                .chunk_by(|b| (b.depth(), b.chunk_idx()))
                            {
                                let filled_blocks = chunk.collect::<Vec<_>>();
                                self.write_updates(&filled_blocks);

                                // Finished processing the blocks, so clear them and put them back
                                // in `self.available_blocks`
                                let mut available_blocks_lock = available_blocks.lock().unwrap();
                                for block in filled_blocks {
                                    available_blocks_lock.push(block.clear());
                                }
                                drop(available_blocks_lock);
                            }
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|t| t.join().unwrap());
        });

        self.write_sizes_to_disk();

        tracing::info!("finished writing update blocks");
    }

    /// Assumes that all the blocks have the same depth and chunk_idx
    fn write_updates(&self, filled_blocks: &[FilledUpdateBlock]) {
        if filled_blocks.is_empty() {
            return;
        }

        let depth = filled_blocks[0].depth();
        let chunk_idx = filled_blocks[0].chunk_idx();

        let bytes_to_write = filled_blocks.iter().map(|block| block.len()).sum::<usize>()
            * std::mem::size_of::<u32>();

        // If the number of bytes to write is greater than the size of a chunk, then compress it
        // into an update array to save space
        if bytes_to_write > self.settings.chunk_size_bytes {
            let mut update_buffer = vec![0u8; self.settings.chunk_size_bytes];

            for block in filled_blocks {
                for &chunk_offset in block.updates() {
                    let (byte_idx, bit_idx) = (chunk_offset / 8, chunk_offset % 8);
                    update_buffer[byte_idx as usize] |= 1 << bit_idx;
                }
            }

            self.write_update_array(&update_buffer, depth, chunk_idx);
        } else {
            let buffers = filled_blocks
                .iter()
                .map(|block| bytemuck::cast_slice(block.updates()))
                .collect::<Vec<_>>();

            // Only write the chunk if it's not empty. We still need to return
            // the blocks to `self.available_blocks` even if we don't write
            // them, though
            if filled_blocks.iter().any(|block| !block.is_empty()) {
                let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
                std::fs::create_dir_all(&dir_path).unwrap();

                let mut rng = rand::thread_rng();
                let file_name = Alphanumeric.sample_string(&mut rng, 16);
                let file_path = dir_path.join(file_name);

                let bytes_written = self
                    .locked_io
                    .write_file_multiple_buffers(&file_path, &buffers);

                let mut sizes_lock = self.sizes.write().unwrap();
                let sizes_for_depth = sizes_lock
                    .entry(depth)
                    .or_insert_with(|| vec![0; self.settings.num_array_chunks()]);
                sizes_for_depth[chunk_idx] += bytes_written;
                drop(sizes_lock);
            }
        }
    }

    fn take_impl(&self, log: bool) -> AvailableUpdateBlock {
        if log {
            let blocks_remaining = self.available_blocks.lock().unwrap().len();
            tracing::trace!("taking update block, {blocks_remaining} blocks remaining");
        }

        loop {
            if let Some(block) = self.available_blocks.lock().unwrap().pop() {
                return block;
            }

            self.write_all();
        }
    }

    fn take(&self) -> AvailableUpdateBlock {
        self.take_impl(true)
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

        if !path.exists() {
            return;
        }

        let str = self.locked_io.try_read_to_string(&path).unwrap();
        let hashmap = serde_json::from_str(&str).unwrap();

        let mut lock = self.sizes.write().unwrap();
        *lock = hashmap;
    }

    pub fn take_n(&self, n: usize) -> Vec<AvailableUpdateBlock> {
        let blocks_remaining = self.available_blocks.lock().unwrap().len();
        tracing::trace!("taking {n} update blocks, {blocks_remaining} blocks remaining");

        let mut blocks = Vec::with_capacity(n);

        for _ in 0..n {
            let block = self.take_impl(false);
            blocks.push(block);
        }

        blocks
    }

    pub fn put(&self, block: FilledUpdateBlock) {
        self.filled_blocks.lock().unwrap().push(block);
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

        // Find all paths to delete and then batch-delete them
        let mut path_bufs_to_delete = Vec::new();
        for path in read_dir.flatten().map(|entry| entry.path()) {
            if !delete_used_only || path.extension().and_then(|ext| ext.to_str()) == Some("used") {
                path_bufs_to_delete.push(path);
            }
        }
        let paths_to_delete = path_bufs_to_delete
            .iter()
            .map(|p| p.as_path())
            .collect::<Vec<_>>();
        self.locked_io.try_delete_files(&paths_to_delete).unwrap();

        // Read the actual size of the remaining files on disk here. This isn't actually necessary,
        // it's just in case `self.sizes` is out of sync somehow
        let real_size = self.real_files_size(depth, chunk_idx);

        self.sizes
            .write()
            .unwrap()
            .entry(depth)
            .and_modify(|entry| entry[chunk_idx] = real_size);

        self.write_sizes_to_disk();
    }

    pub fn delete_update_files(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_files_impl(depth, chunk_idx, false);
    }

    pub fn delete_used_update_files(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_files_impl(depth, chunk_idx, true);
    }

    fn delete_update_arrays_impl(&self, depth: usize, chunk_idx: usize, delete_used_only: bool) {
        if delete_used_only {
            tracing::debug!("deleting used update arrays for depth {depth} chunk {chunk_idx}");
        } else {
            tracing::debug!("deleting update arrays for depth {depth} chunk {chunk_idx}");
        }

        let dir_path = self.settings.update_array_chunk_dir_path(depth, chunk_idx);
        let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
            return;
        };

        // Find all paths to delete and then batch-delete them
        let mut path_bufs_to_delete = Vec::new();
        for path in read_dir.flatten().map(|entry| entry.path()) {
            if !delete_used_only || path.extension().and_then(|ext| ext.to_str()) == Some("used") {
                path_bufs_to_delete.push(path);
            }
        }
        let paths_to_delete = path_bufs_to_delete
            .iter()
            .map(|p| p.as_path())
            .collect::<Vec<_>>();
        self.locked_io.try_delete_files(&paths_to_delete).unwrap();

        // Read the actual size of the remaining files on disk here. This isn't actually necessary,
        // it's just in case `self.sizes` is out of sync somehow
        let real_size = self.real_files_size(depth, chunk_idx);

        self.sizes
            .write()
            .unwrap()
            .entry(depth)
            .and_modify(|entry| entry[chunk_idx] = real_size);

        self.write_sizes_to_disk();
    }

    pub fn delete_update_arrays(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_arrays_impl(depth, chunk_idx, false);
    }

    pub fn delete_used_update_arrays(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_arrays_impl(depth, chunk_idx, true);
    }

    pub fn write_update_array(&self, update_buffer: &[u8], depth: usize, chunk_idx: usize) {
        let dir_path = self.settings.update_array_chunk_dir_path(depth, chunk_idx);
        std::fs::create_dir_all(&dir_path).unwrap();

        let mut rng = rand::thread_rng();
        let file_name = Alphanumeric.sample_string(&mut rng, 16);
        let file_path = dir_path.join(file_name);

        let bytes_written = self.locked_io.write_file(&file_path, &update_buffer);

        let mut sizes_lock = self.sizes.write().unwrap();
        let sizes_for_depth = sizes_lock
            .entry(depth)
            .or_insert_with(|| vec![0; self.settings.num_array_chunks()]);
        sizes_for_depth[chunk_idx] += bytes_written;
        drop(sizes_lock);
    }

    pub fn files_size(&self, depth: usize, chunk_idx: usize) -> u64 {
        let read_lock = self.sizes.read().unwrap();
        read_lock.get(&depth).map_or(0, |sizes| sizes[chunk_idx])
    }

    fn real_files_size(&self, depth: usize, chunk_idx: usize) -> u64 {
        let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
        let update_files_size = std::fs::read_dir(&dir_path)
            .map(|read_dir| {
                read_dir
                    .flatten()
                    // It's possible that one of these files may be a `.tmp` file which was
                    // renamed between the call to `read_dir` and now. In that case, unwrapping
                    // `entry.metadata()` would panic since the file no longer exists, so we use
                    // 0 as the length instead. This does mean we skip over the renamed file,
                    // but it's not important because these numbers are only used as an
                    // approximation to know when to compress update files.
                    .map(|entry| entry.metadata().map_or(0, |m| m.len()))
                    .sum::<u64>()
            })
            .unwrap_or_default();

        let dir_path = self.settings.update_array_chunk_dir_path(depth, chunk_idx);
        let update_arrays_size = std::fs::read_dir(&dir_path)
            .map(|read_dir| {
                read_dir
                    .flatten()
                    .map(|entry| entry.metadata().map_or(0, |m| m.len()))
                    .sum::<u64>()
            })
            .unwrap_or_default();

        update_files_size + update_arrays_size
    }

    pub fn mark_filled_and_replace(
        &self,
        upd: &mut AvailableUpdateBlock,
        depth: usize,
        chunk_idx: usize,
    ) {
        tracing::trace!(
            "marking update block as filled, contains {}/{} values",
            upd.len(),
            upd.capacity(),
        );

        let new = self.take();
        let old = std::mem::replace(upd, new).into_filled(depth, chunk_idx);
        self.put(old);
    }
}
