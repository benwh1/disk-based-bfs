use std::{collections::HashMap, sync::Arc, thread::Builder as ThreadBuilder};

use itertools::Itertools;
use parking_lot::{Condvar, Mutex, RwLock};
use rand::distributions::{Alphanumeric, DistString as _};

use crate::{
    io::LockedIO,
    settings::{BfsSettings, BfsSettingsProvider},
    update::blocks::{AvailableUpdateBlock, FillableUpdateBlock, FilledUpdateBlock},
};

struct BlockCondition {
    is_writing: Mutex<bool>,
    is_writing_cvar: Condvar,
    block_available_cvar: Condvar,
}

pub(crate) struct UpdateManager<'a, P: BfsSettingsProvider + Sync> {
    settings: &'a BfsSettings<P>,
    locked_io: &'a LockedIO<'a, P>,
    sizes: RwLock<HashMap<usize, Vec<u64>>>,
    size_file_lock: Mutex<()>,
    available_blocks: Mutex<Vec<AvailableUpdateBlock>>,
    filled_blocks: Mutex<Vec<FilledUpdateBlock>>,
    block_condition: Arc<BlockCondition>,
}

impl<'a, P: BfsSettingsProvider + Sync> UpdateManager<'a, P> {
    pub(crate) fn new(settings: &'a BfsSettings<P>, locked_io: &'a LockedIO<P>) -> Self {
        let num_blocks = settings.num_update_blocks;

        tracing::debug!("creating {num_blocks} update blocks");

        let capacity = settings.update_block_capacity();

        let available_blocks = Mutex::new(
            (0..num_blocks)
                .map(|_| AvailableUpdateBlock::new(capacity))
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
            block_condition: Arc::new(BlockCondition {
                is_writing: Mutex::new(false),
                is_writing_cvar: Condvar::new(),
                block_available_cvar: Condvar::new(),
            }),
        }
    }

    /// Write filled blocks to disk and put them back in `self.available_blocks`
    fn write_and_put(&self, filled_blocks: Vec<FilledUpdateBlock>) {
        if filled_blocks.is_empty() {
            return;
        }

        tracing::info!("writing {} update blocks", filled_blocks.len());

        let num_disks = self.locked_io.num_disks();

        // Sort all the blocks into chunks that will be written to the same disk so that each chunk
        // can be processed by a separate thread
        let mut chunked_filled_blocks = (0..num_disks).map(|_| Vec::new()).collect::<Vec<_>>();
        for block in filled_blocks.into_iter() {
            let chunk_root_idx = self
                .settings
                .settings_provider
                .chunk_root_idx(block.chunk_idx());
            chunked_filled_blocks[chunk_root_idx].push(block);
        }

        // Sort each chunk by depth and chunk_idx so that blocks that can be written to the same
        // file are consecutive in the list
        for chunk in &mut chunked_filled_blocks {
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
                                let mut available_blocks_lock = available_blocks.lock();
                                for block in filled_blocks {
                                    available_blocks_lock.push(block.clear());
                                }
                                drop(available_blocks_lock);

                                // Notify any waiting threads that blocks are now available
                                self.block_condition.block_available_cvar.notify_all();
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

        // Notify any waiting threads that we're done writing
        let block_condition = &*self.block_condition;
        *block_condition.is_writing.lock() = false;
        block_condition.is_writing_cvar.notify_all();
    }

    /// Assumes that all the blocks have the same `depth` and `chunk_idx`
    fn write_updates(&self, filled_blocks: &[FilledUpdateBlock]) {
        if filled_blocks.is_empty() {
            return;
        }

        let depth = filled_blocks[0].depth();
        let chunk_idx = filled_blocks[0].chunk_idx();

        let bytes_to_write = filled_blocks
            .iter()
            .map(|block| block.len() as u64)
            .sum::<u64>()
            * std::mem::size_of::<u32>() as u64;

        // If the number of bytes to write is greater than the threshold, compress it into an
        // update array
        if bytes_to_write > self.settings.update_array_threshold {
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
                let mut rng = rand::thread_rng();
                let file_name = Alphanumeric.sample_string(&mut rng, 16);
                let file_path = dir_path.join(file_name);

                let bytes_written = self
                    .locked_io
                    .write_file_multiple_buffers(&file_path, &buffers, false);

                let mut sizes_lock = self.sizes.write();
                let sizes_for_depth = sizes_lock
                    .entry(depth)
                    .or_insert_with(|| vec![0; self.settings.num_array_chunks()]);
                sizes_for_depth[chunk_idx] += bytes_written;
                drop(sizes_lock);
            }
        }
    }

    pub(crate) fn write_all(&self) {
        let block_condition = &*self.block_condition;
        let mut is_writing_lock = block_condition.is_writing.lock();
        while *is_writing_lock {
            block_condition.is_writing_cvar.wait(&mut is_writing_lock);
        }
        *is_writing_lock = true;
        drop(is_writing_lock);

        let mut filled_blocks_lock = self.filled_blocks.lock();
        let filled_blocks = std::mem::take(&mut *filled_blocks_lock);
        drop(filled_blocks_lock);

        self.write_and_put(filled_blocks);
    }

    /// Write all blocks that have the given `source_depth` and `source_chunk_idx`
    pub(crate) fn write_from_source(&self, source_depth: usize, source_chunk_idx: usize) {
        let block_condition = &*self.block_condition;
        let mut is_writing_lock = block_condition.is_writing.lock();
        while *is_writing_lock {
            block_condition.is_writing_cvar.wait(&mut is_writing_lock);
        }
        *is_writing_lock = true;
        drop(is_writing_lock);

        let mut filled_blocks_lock = self.filled_blocks.lock();
        let filled_blocks = std::mem::take(&mut *filled_blocks_lock);

        // Separate out the blocks that we will write to disk
        let (to_write, mut to_keep): (Vec<_>, Vec<_>) =
            filled_blocks.into_iter().partition(|block| {
                block.source_depth() == source_depth && block.source_chunk_idx() == source_chunk_idx
            });

        // Return the blocks that we're not writing to disk back to `self.filled_blocks`
        std::mem::swap(&mut *filled_blocks_lock, &mut to_keep);

        drop(filled_blocks_lock);

        self.write_and_put(to_write);
    }

    pub(crate) fn take(&self) -> AvailableUpdateBlock {
        loop {
            if let Some(block) = self.available_blocks.lock().pop() {
                return block;
            }

            // If another thread is writing updates, just wait for a block to become available
            // instead of trying to write updates ourselves
            let block_condition = &*self.block_condition;
            let mut is_writing_lock = block_condition.is_writing.lock();
            if !*is_writing_lock {
                self.write_all();
            } else {
                block_condition
                    .block_available_cvar
                    .wait(&mut is_writing_lock);
            }
        }
    }

    /// Note: the sizes are not guaranteed to be *exactly* correct, because it's possible that we
    /// could write some update files to disk and then the program is terminated before we can write
    /// the sizes. This isn't important though, because the sizes are only used to determine if we
    /// should compress the update files, and it really doesn't matter if we sometimes compress them
    /// slightly before or after the actual size reaches the threshold.
    fn write_sizes_to_disk(&self) {
        let _lock = self.size_file_lock.lock();

        let read_lock = self.sizes.read();
        let str = serde_json::to_string(&*read_lock).unwrap();
        drop(read_lock);

        let path = self.settings.update_files_size_file_path();
        self.locked_io.write_file(&path, str.as_ref(), false);
    }

    pub(crate) fn try_read_sizes_from_disk(&self) {
        let path = self.settings.update_files_size_file_path();

        if !path.exists() {
            return;
        }

        let str = self.locked_io.try_read_to_string(&path, false).unwrap();
        let hashmap = serde_json::from_str(&str).unwrap();

        let mut lock = self.sizes.write();
        *lock = hashmap;
    }

    pub(crate) fn put(&self, block: FilledUpdateBlock) {
        self.filled_blocks.lock().push(block);
    }

    fn delete_update_files_impl(&self, depth: usize, chunk_idx: usize, delete_used_only: bool) {
        if delete_used_only {
            tracing::debug!("deleting used update files for depth {depth} chunk {chunk_idx}");
        } else {
            tracing::debug!("deleting update files for depth {depth} chunk {chunk_idx}");
        }

        let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        // Find all paths to delete and then batch-delete them
        let mut path_bufs_to_delete = Vec::new();
        for path in read_dir.flatten().map(|entry| entry.path()) {
            let ext = path.extension().and_then(|ext| ext.to_str());
            if (!delete_used_only && ext.is_none()) || ext == Some("used") {
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
            .entry(depth)
            .and_modify(|entry| entry[chunk_idx] = real_size);

        self.write_sizes_to_disk();
    }

    pub(crate) fn delete_update_files(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_files_impl(depth, chunk_idx, false);
    }

    pub(crate) fn delete_used_update_files(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_files_impl(depth, chunk_idx, true);
    }

    fn delete_update_arrays_impl(&self, depth: usize, chunk_idx: usize, delete_used_only: bool) {
        if delete_used_only {
            tracing::debug!("deleting used update arrays for depth {depth} chunk {chunk_idx}");
        } else {
            tracing::debug!("deleting update arrays for depth {depth} chunk {chunk_idx}");
        }

        let dir_path = self.settings.update_array_chunk_dir_path(depth, chunk_idx);
        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        // Find all paths to delete and then batch-delete them
        let mut path_bufs_to_delete = Vec::new();
        for path in read_dir.flatten().map(|entry| entry.path()) {
            let ext = path.extension().and_then(|ext| ext.to_str());
            if (!delete_used_only && ext.is_none()) || ext == Some("used") {
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
            .entry(depth)
            .and_modify(|entry| entry[chunk_idx] = real_size);

        self.write_sizes_to_disk();
    }

    pub(crate) fn delete_update_arrays(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_arrays_impl(depth, chunk_idx, false);
    }

    pub(crate) fn delete_used_update_arrays(&self, depth: usize, chunk_idx: usize) {
        self.delete_update_arrays_impl(depth, chunk_idx, true);
    }

    pub(crate) fn back_up_update_arrays(&self, depth: usize, chunk_idx: usize) {
        let dir_path = self.settings.update_array_chunk_dir_path(depth, chunk_idx);
        let backup_dir_path = self
            .settings
            .backup_update_array_chunk_dir_path(depth, chunk_idx);

        std::fs::create_dir_all(&backup_dir_path).unwrap();
        std::fs::rename(&dir_path, &backup_dir_path).unwrap();
    }

    pub(crate) fn write_update_array(&self, update_buffer: &[u8], depth: usize, chunk_idx: usize) {
        let dir_path = self.settings.update_array_chunk_dir_path(depth, chunk_idx);
        let mut rng = rand::thread_rng();
        let file_name = Alphanumeric.sample_string(&mut rng, 16);
        let file_path = dir_path.join(file_name);

        let bytes_written =
            self.locked_io
                .write_file(&file_path, update_buffer, self.settings.compress_bit_arrays);

        let mut sizes_lock = self.sizes.write();
        let sizes_for_depth = sizes_lock
            .entry(depth)
            .or_insert_with(|| vec![0; self.settings.num_array_chunks()]);
        sizes_for_depth[chunk_idx] += bytes_written;
        drop(sizes_lock);
    }

    pub(crate) fn files_size(&self, depth: usize, chunk_idx: usize) -> u64 {
        let read_lock = self.sizes.read();
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

    pub(crate) fn mark_filled_and_replace(
        &self,
        upd: &mut FillableUpdateBlock,
        depth: usize,
        chunk_idx: usize,
    ) {
        let new = self
            .take()
            .into_fillable(upd.source_depth(), upd.source_chunk_idx());
        let old = std::mem::replace(upd, new).into_filled(depth, chunk_idx);
        self.put(old);
    }
}
