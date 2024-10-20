use std::{cell::OnceCell, sync::Arc, thread::Builder as ThreadBuilder};

use cityhasher::{CityHasher, HashSet};
use parking_lot::{Condvar, Mutex, RwLock};
use rand::distributions::{Alphanumeric, DistString as _};
use serde_derive::{Deserialize, Serialize};

use crate::{
    callback::BfsCallback,
    chunk_buffer_list::ChunkBufferList,
    expander::{BfsExpander, NONE},
    io::{self, LockedIO},
    provider::{BfsSettingsProvider, ChunkFilesBehavior, UpdateFilesBehavior},
    settings::BfsSettings,
    update::{blocks::FillableUpdateBlock, manager::UpdateManager},
};

enum InMemoryBfsResult {
    Complete,
    OutOfMemory {
        old: HashSet<u64, CityHasher>,
        current: HashSet<u64, CityHasher>,
        next: HashSet<u64>,
        depth: usize,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum State {
    Iteration { depth: usize },
    MergeUpdateFiles { depth: usize },
    Cleanup { depth: usize },
    Done,
}

pub(crate) struct Bfs<'a, Expander, Callback, Provider, const EXPANSION_NODES: usize> {
    settings: &'a BfsSettings<Expander, Callback, Provider, EXPANSION_NODES>,
    locked_io: &'a LockedIO<'a, Expander, Callback, Provider, EXPANSION_NODES>,
    chunk_buffers: ChunkBufferList,
    update_file_manager: UpdateManager<'a, Expander, Callback, Provider, EXPANSION_NODES>,
}

impl<'a, Expander, Callback, Provider, const EXPANSION_NODES: usize>
    Bfs<'a, Expander, Callback, Provider, EXPANSION_NODES>
where
    Expander: BfsExpander<EXPANSION_NODES> + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    Provider: BfsSettingsProvider + Sync,
{
    pub(crate) fn new(
        settings: &'a BfsSettings<Expander, Callback, Provider, EXPANSION_NODES>,
        locked_io: &'a LockedIO<Expander, Callback, Provider, EXPANSION_NODES>,
    ) -> Self {
        let chunk_buffers = ChunkBufferList::new_empty(settings.threads);
        let update_file_manager = UpdateManager::new(settings, locked_io);

        Self {
            settings,
            locked_io,
            chunk_buffers,
            update_file_manager,
        }
    }

    fn read_new_positions_data_file(&self, depth: usize, chunk_idx: usize) -> u64 {
        let file_path = self.settings.new_positions_data_file_path(depth, chunk_idx);
        let mut buf = [0u8; 8];

        self.locked_io.read_file(&file_path, &mut buf, false);

        u64::from_le_bytes(buf)
    }

    fn write_new_positions_data_file(&self, new: u64, depth: usize, chunk_idx: usize) {
        let file_path = self.settings.new_positions_data_file_path(depth, chunk_idx);
        self.locked_io
            .write_file(&file_path, &new.to_le_bytes(), false);
    }

    fn delete_new_positions_data_dir(&self, depth: usize) {
        let file_path = self.settings.new_positions_data_dir_path(depth);
        if file_path.exists() {
            std::fs::remove_dir_all(file_path).unwrap();
        }
    }

    fn read_state(&self) -> Option<State> {
        let file_path = self.settings.state_file_path();
        let str = self.locked_io.try_read_to_string(&file_path, false).ok()?;
        serde_json::from_str(&str).ok()
    }

    fn write_state(&self, state: State) {
        let str = serde_json::to_string(&state).unwrap();
        let file_path = self.settings.state_file_path();
        self.locked_io.write_file(&file_path, str.as_ref(), false);
    }

    fn try_read_chunk(&self, chunk_buffer: &mut [u8], depth: usize, chunk_idx: usize) -> bool {
        let file_path = self.settings.chunk_file_path(depth, chunk_idx);

        let result =
            self.locked_io
                .try_read_file(&file_path, chunk_buffer, self.settings.use_compression);

        match result {
            Ok(()) => true,
            Err(err) if err.is_read_nonexistent_file_error() => false,
            Err(err) => panic!("failed to read file {file_path:?}: {err}"),
        }
    }

    fn create_chunk(&self, chunk_buffer: &mut [u8], hashsets: &[&HashSet<u64>], chunk_idx: usize) {
        chunk_buffer.fill(0);

        for hashset in hashsets {
            for (_, byte_idx, bit_idx) in hashset
                .iter()
                .map(|&val| self.node_to_bit_coords(val))
                .filter(|&(i, _, _)| chunk_idx == i)
            {
                chunk_buffer[byte_idx] |= 1 << bit_idx;
            }
        }
    }

    fn write_chunk(&self, chunk_buffer: &[u8], depth: usize, chunk_idx: usize) {
        let file_path = self.settings.chunk_file_path(depth, chunk_idx);
        self.locked_io
            .write_file(&file_path, chunk_buffer, self.settings.use_compression);
    }

    fn delete_chunk_file(&self, depth: usize, chunk_idx: usize) {
        let file_path = self.settings.chunk_file_path(depth, chunk_idx);
        if file_path.exists() {
            self.locked_io.try_delete_file(&file_path).unwrap();
        }
    }

    fn back_up_chunk_file(&self, depth: usize, chunk_idx: usize) {
        let backup_dir_path = self.settings.backup_chunk_dir_path(depth, chunk_idx);
        std::fs::create_dir_all(&backup_dir_path).unwrap();

        let file_path = self.settings.chunk_file_path(depth, chunk_idx);
        let backup_file_path = self.settings.backup_chunk_file_path(depth, chunk_idx);

        std::fs::rename(file_path, backup_file_path).unwrap();
    }

    fn mark_chunk_exhausted(&self, depth: usize, chunk_idx: usize) {
        let dir_path = self.settings.exhausted_chunk_dir_path();
        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path = self.settings.exhausted_chunk_file_path(chunk_idx);
        self.locked_io
            .write_file(&file_path, &depth.to_le_bytes(), false);
    }

    fn chunk_exhausted_depth(&self, chunk_idx: usize) -> Option<usize> {
        let file_path = self.settings.exhausted_chunk_file_path(chunk_idx);
        let mut buf = [0u8; std::mem::size_of::<usize>()];

        if self
            .locked_io
            .try_read_file(&file_path, &mut buf, false)
            .is_err()
        {
            return None;
        }

        Some(usize::from_le_bytes(buf))
    }

    fn update_update_buffer_from_update_arrays(
        &self,
        update_buffer: &mut [u8],
        depth: usize,
        chunk_idx: usize,
    ) {
        let dir_path = self.settings.update_array_chunk_dir_path(depth, chunk_idx);
        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        // Mark "used" files as unused, in case we restart the program while this loop is running
        // and need to re-read all the update arrays
        for file_path in read_dir
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("used"))
        {
            let file_path_unused = file_path.with_extension("");
            std::fs::rename(file_path, file_path_unused).unwrap();
        }

        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        // Update from update arrays
        for file_path in read_dir
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_none())
        {
            let bytes = self
                .locked_io
                .read_to_vec(&file_path, self.settings.use_compression);
            assert_eq!(bytes.len(), update_buffer.len());

            for (buf_byte, &new) in update_buffer.iter_mut().zip(bytes.iter()) {
                *buf_byte |= new;
            }

            // Rename the file to mark it as used
            let file_path_used = file_path.with_extension("used");
            std::fs::rename(file_path, file_path_used).unwrap();
        }
    }

    fn update_update_buffer_from_update_files(
        &self,
        update_buffer: &mut [u8],
        depth: usize,
        chunk_idx: usize,
    ) {
        let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        // Mark "used" files as unused, in case we restart the program while this loop is running
        // and need to re-read all the update files
        for file_path in read_dir
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("used"))
        {
            let file_path_unused = file_path.with_extension("");
            std::fs::rename(file_path, file_path_unused).unwrap();
        }

        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        // Update from update files
        for file_path in read_dir
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_none())
        {
            let bytes = self
                .locked_io
                .read_to_vec(&file_path, self.settings.use_compression);
            assert_eq!(bytes.len() % 4, 0);

            // Group the bytes into groups of 4
            let chunk_offsets: &[u32] = bytemuck::cast_slice(&bytes);
            for &chunk_offset in chunk_offsets {
                let (byte_idx, bit_idx) = self.chunk_offset_to_bit_coords(chunk_offset);
                update_buffer[byte_idx] |= 1 << bit_idx;
            }

            // Rename the file to mark it as used
            let file_path_used = file_path.with_extension("used");
            std::fs::rename(file_path, file_path_used).unwrap();
        }
    }

    fn has_update_files_to_merge(&self, depth: usize, chunk_idx: usize) -> bool {
        let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        if read_dir.flatten().map(|entry| entry.path()).any(|path| {
            let ext = path.extension().and_then(|ext| ext.to_str());
            ext.is_none() || ext == Some("used")
        }) {
            return true;
        }

        let dir_path = self.settings.update_array_chunk_dir_path(depth, chunk_idx);
        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        read_dir
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                let ext = path.extension().and_then(|ext| ext.to_str());
                ext.is_none() || ext == Some("used")
            })
            .count()
            > 1
    }

    fn merge_update_files(&self, update_buffer: &mut [u8], depth: usize, chunk_idx: usize) {
        if !self.has_update_files_to_merge(depth, chunk_idx) {
            tracing::debug!(
                "no update files to merge for depth {} -> {} chunk {chunk_idx}, skipping",
                depth - 1,
                depth,
            );
            return;
        }

        let used_space = self.update_file_manager.files_size(depth, chunk_idx);
        let gb = used_space as f64 / (1 << 30) as f64;

        tracing::info!(
            "merging {gb:.3} GiB of update files for depth {} -> {} chunk {chunk_idx}",
            depth - 1,
            depth,
        );

        update_buffer.fill(0);

        self.update_update_buffer_from_update_arrays(update_buffer, depth, chunk_idx);
        self.update_update_buffer_from_update_files(update_buffer, depth, chunk_idx);

        self.update_file_manager
            .write_update_array(update_buffer, depth, chunk_idx);

        if self.settings.sync_filesystem {
            io::sync();
        }

        self.update_file_manager
            .delete_used_update_arrays(depth, chunk_idx);
        self.update_file_manager
            .delete_used_update_files(depth, chunk_idx);

        tracing::info!(
            "finished merging update files for depth {} -> {} chunk {chunk_idx}",
            depth - 1,
            depth,
        );
    }

    fn merge_all_update_files(&self, depth: usize) {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum UpdateFileState {
            NotMerging,
            CurrentlyMerging,
            Merged,
        }

        let update_file_states = Arc::new(RwLock::new(vec![
            UpdateFileState::NotMerging;
            self.settings.num_array_chunks()
        ]));
        let pair = Arc::new((Mutex::new(false), Condvar::new()));

        tracing::info!("merging all depth {depth} update files");

        std::thread::scope(|s| {
            let threads = (0..self.settings.threads)
                .map(|t| {
                    let update_file_states = update_file_states.clone();
                    let pair = pair.clone();

                    ThreadBuilder::new()
                        .name({
                            let digits = self.settings.threads.ilog10() as usize + 1;
                            format!("merge-update-files-{t:0>digits$}")
                        })
                        .spawn_scoped(s, move || loop {
                            let (lock, cvar) = &*pair;
                            let mut has_work = lock.lock();

                            // Wait for work
                            while !*has_work {
                                cvar.wait(&mut has_work);
                            }

                            *has_work = false;
                            drop(has_work);

                            // If everything is done, notify all and break
                            let update_file_states_read = update_file_states.read();
                            if update_file_states_read
                                .iter()
                                .all(|&x| x == UpdateFileState::Merged)
                            {
                                *lock.lock() = true;
                                cvar.notify_all();
                                break;
                            }

                            // Check for update files to merge
                            let chunk_idx = update_file_states_read
                                .iter()
                                .position(|&x| x == UpdateFileState::NotMerging);
                            drop(update_file_states_read);

                            if let Some(chunk_idx) = chunk_idx {
                                // Set the state to currently merging
                                let mut update_file_states_write = update_file_states.write();
                                if update_file_states_write[chunk_idx]
                                    == UpdateFileState::NotMerging
                                {
                                    update_file_states_write[chunk_idx] =
                                        UpdateFileState::CurrentlyMerging;
                                } else {
                                    // Another thread got here first
                                    continue;
                                }
                                drop(update_file_states_write);

                                *lock.lock() = true;
                                cvar.notify_one();

                                // Get a chunk buffer
                                let mut chunk_buffer = self.chunk_buffers.take().unwrap();

                                self.merge_update_files(&mut chunk_buffer, depth, chunk_idx);

                                // Put the chunk buffer back
                                self.chunk_buffers.put(chunk_buffer);

                                // Mark the update files as merged
                                let mut update_file_states_write = update_file_states.write();
                                update_file_states_write[chunk_idx] = UpdateFileState::Merged;
                                drop(update_file_states_write);

                                // Delete (now empty) update files directory
                                self.update_file_manager
                                    .delete_update_files(depth, chunk_idx);

                                *lock.lock() = true;
                                cvar.notify_one();
                            }
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let (lock, cvar) = &*pair;
            let mut has_work = lock.lock();
            *has_work = true;
            cvar.notify_one();
            drop(has_work);

            threads.into_iter().for_each(|t| t.join().unwrap());
        });

        tracing::info!("finished merging all depth {depth} update files");
    }

    fn update_and_expand_chunk(
        &self,
        chunk_buffer: &mut [u8],
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0;
        let mut updates = (0..self.settings.num_array_chunks())
            .map(|_| OnceCell::new())
            .collect::<Vec<_>>();

        let mut callback = self.settings.callback.clone();

        new_positions += self.update_and_expand_from_update_files(
            chunk_buffer,
            &mut updates,
            &mut callback,
            depth,
            chunk_idx,
        );

        new_positions += self.update_and_expand_from_update_arrays(
            chunk_buffer,
            &mut updates,
            &mut callback,
            depth,
            chunk_idx,
        );

        callback.end_of_chunk(depth + 1, chunk_idx);

        // Send the remaining updates back to the update manager
        for (idx, upd) in updates
            .into_iter()
            .enumerate()
            .filter_map(|(i, u)| u.into_inner().map(|u| (i, u)))
        {
            self.update_file_manager
                .put(upd.into_filled(depth + 2, idx));
        }

        // Force the updates to be written to disk - this is necessary otherwise it's possible that
        // this chunk will be deleted and the program terminated before the final updates are
        // written to disk
        tracing::debug!(
            "writing remaining update files for depth {depth} -> {} chunk {chunk_idx}",
            depth + 1,
        );
        self.update_file_manager
            .write_from_source(depth + 1, chunk_idx);

        // At this point, it's possible that another thread could be in the middle of a call to
        // `write_all`, and be holding some update blocks sourced from this chunk. We need to wait
        // for those updates to be written to disk before it's safe to return from this function
        // and delete the chunk and update files.
        //
        // Only one thread can call `write_all` at a time, so we just need to wait for a
        // notification from the condition variable.
        self.update_file_manager.wait_for_write_all();

        new_positions
    }

    fn check_update_vec_capacity(
        &self,
        updates: &mut [OnceCell<FillableUpdateBlock>],
        depth: usize,
    ) {
        // Check if any of the update vecs may go over capacity
        let max_new_nodes = self.settings.capacity_check_frequency * EXPANSION_NODES;

        for (idx, upd) in updates
            .iter_mut()
            .enumerate()
            .filter_map(|(i, u)| u.get_mut().map(|u| (i, u)))
        {
            if upd.len() + max_new_nodes > upd.capacity() {
                // Possible to reach capacity on the next block of expansions
                self.update_file_manager
                    .mark_filled_and_replace(upd, depth, idx);
            }
        }
    }

    fn update_and_expand_from_update_files(
        &self,
        chunk_buffer: &mut [u8],
        updates: &mut [OnceCell<FillableUpdateBlock>],
        callback: &mut Callback,
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0u64;

        let mut expander = self.settings.expander.clone();
        let mut expanded_nodes = [0u64; EXPANSION_NODES];

        let dir_path = self.settings.update_chunk_dir_path(depth + 1, chunk_idx);
        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        for file_path in read_dir.flatten().map(|entry| entry.path()).filter(|path| {
            let ext = path.extension().and_then(|ext| ext.to_str());
            ext.is_none() || ext == Some("used")
        }) {
            let bytes = self
                .locked_io
                .read_to_vec(&file_path, self.settings.use_compression);
            assert_eq!(bytes.len() % 4, 0);

            let chunk_offsets: &[u32] = bytemuck::cast_slice(&bytes);
            for &chunk_offset in chunk_offsets {
                let (byte_idx, bit_idx) = self.chunk_offset_to_bit_coords(chunk_offset);
                let byte = chunk_buffer[byte_idx];

                if (byte >> bit_idx) & 1 == 0 {
                    chunk_buffer[byte_idx] |= 1 << bit_idx;
                    new_positions += 1;

                    let encoded = self.bit_coords_to_node(chunk_idx, byte_idx, bit_idx);
                    callback.new_state(depth + 1, encoded);

                    if new_positions % self.settings.capacity_check_frequency as u64 == 0 {
                        self.check_update_vec_capacity(updates, depth + 2);
                    }

                    // Expand the node
                    expander.expand(encoded, &mut expanded_nodes);

                    for node in expanded_nodes {
                        if node == NONE {
                            continue;
                        }

                        let (idx, offset) = self.node_to_chunk_coords(node);
                        updates[idx]
                            .get_mut_or_init(|| {
                                self.update_file_manager
                                    .take()
                                    .into_fillable(depth + 1, chunk_idx)
                            })
                            .push(offset);
                    }
                }
            }
        }

        new_positions
    }

    fn update_and_expand_from_update_arrays(
        &self,
        chunk_buffer: &mut [u8],
        updates: &mut [OnceCell<FillableUpdateBlock>],
        callback: &mut Callback,
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0u64;

        let mut expander = self.settings.expander.clone();
        let mut expanded_nodes = [0u64; EXPANSION_NODES];

        let dir_path = self
            .settings
            .update_array_chunk_dir_path(depth + 1, chunk_idx);
        let read_dir = std::fs::read_dir(&dir_path)
            .inspect_err(|_| panic!("failed to read directory {dir_path:?}"))
            .unwrap();

        for file_path in read_dir.flatten().map(|entry| entry.path()).filter(|path| {
            let ext = path.extension().and_then(|ext| ext.to_str());
            ext.is_none() || ext == Some("used")
        }) {
            let update_array_bytes = self
                .locked_io
                .read_to_vec(&file_path, self.settings.use_compression);
            assert_eq!(update_array_bytes.len(), self.settings.chunk_size_bytes);

            for (byte_idx, &update_byte) in update_array_bytes.iter().enumerate() {
                let mut diff = update_byte & !chunk_buffer[byte_idx];

                while diff != 0 {
                    new_positions += 1;

                    // Index of the least significant set bit
                    let bit_idx = diff.trailing_zeros() as usize;

                    let encoded = self.bit_coords_to_node(chunk_idx, byte_idx, bit_idx);
                    callback.new_state(depth + 1, encoded);

                    if new_positions % self.settings.capacity_check_frequency as u64 == 0 {
                        self.check_update_vec_capacity(updates, depth + 2);
                    }

                    expander.expand(encoded, &mut expanded_nodes);

                    for node in expanded_nodes {
                        if node == NONE {
                            continue;
                        }

                        let (idx, offset) = self.node_to_chunk_coords(node);
                        updates[idx]
                            .get_mut_or_init(|| {
                                self.update_file_manager
                                    .take()
                                    .into_fillable(depth + 1, chunk_idx)
                            })
                            .push(offset);
                    }

                    // Clear the least significant set bit
                    diff &= diff - 1;
                }

                chunk_buffer[byte_idx] |= update_byte;
            }
        }

        new_positions
    }

    fn in_memory_bfs(&self) -> InMemoryBfsResult {
        let max_capacity = self.settings.initial_memory_limit / 8;

        let mut old = HashSet::with_capacity_and_hasher(max_capacity / 2, CityHasher::default());
        let mut current = HashSet::with_hasher(CityHasher::default());
        let mut next = HashSet::with_hasher(CityHasher::default());

        let mut expanded_nodes = [0u64; EXPANSION_NODES];
        let mut depth = 0;

        let mut callback = self.settings.callback.clone();

        for &state in &self.settings.initial_states {
            if current.insert(state) {
                callback.new_state(0, state);
            }
        }

        callback.end_of_chunk(0, 0);

        let mut new;
        let mut total = 1;

        tracing::info!("starting in-memory BFS");

        tracing::info!("depth 0 new 1");

        loop {
            new = 0;

            std::thread::scope(|s| {
                let threads = (0..self.settings.threads)
                    .map(|t| {
                        let current = &current;
                        let old = &old;

                        ThreadBuilder::new()
                            .name({
                                let digits = self.settings.threads.ilog10() as usize + 1;
                                format!("in-memory-bfs-{t:0>digits$}")
                            })
                            .spawn_scoped(s, move || {
                                let mut expander = self.settings.expander.clone();

                                let mut next = HashSet::with_hasher(CityHasher::default());

                                let thread_start = self.settings.state_size
                                    / self.settings.threads as u64
                                    * t as u64;
                                let thread_end = if t == self.settings.threads - 1 {
                                    self.settings.state_size
                                } else {
                                    self.settings.state_size / self.settings.threads as u64
                                        * (t as u64 + 1)
                                };

                                for &encoded in current
                                    .iter()
                                    .filter(|&&val| (thread_start..thread_end).contains(&val))
                                {
                                    expander.expand(encoded, &mut expanded_nodes);
                                    for node in expanded_nodes {
                                        if node == NONE {
                                            continue;
                                        }

                                        if !old.contains(&node) && !current.contains(&node) {
                                            next.insert(node);
                                        }
                                    }
                                }

                                next
                            })
                            .unwrap()
                    })
                    .collect::<Vec<_>>();

                for thread in threads {
                    let mut thread_next = thread.join().unwrap();
                    for node in thread_next.drain() {
                        if next.insert(node) {
                            new += 1;
                            callback.new_state(depth + 1, node);
                        }
                    }
                }
            });
            callback.end_of_chunk(depth + 1, 0);

            depth += 1;
            total += new;

            if total > max_capacity {
                tracing::info!("exceeded memory limit");
                break;
            }

            tracing::info!("depth {depth} new {new}");

            // No new nodes, we are done already.
            if new == 0 {
                return InMemoryBfsResult::Complete;
            }

            for val in current.drain() {
                old.insert(val);
            }
            std::mem::swap(&mut current, &mut next);
        }

        depth -= 1;

        InMemoryBfsResult::OutOfMemory {
            old,
            current,
            next,
            depth,
        }
    }

    /// Converts an encoded node value to (`chunk_idx`, `byte_idx`, `bit_idx`)
    fn node_to_bit_coords(&self, node: u64) -> (usize, usize, usize) {
        let node = node as usize;
        let chunk_idx = (node / 8) / self.settings.chunk_size_bytes;
        let byte_idx = (node / 8) % self.settings.chunk_size_bytes;
        let bit_idx = node % 8;
        (chunk_idx, byte_idx, bit_idx)
    }

    fn bit_coords_to_node(&self, chunk_idx: usize, byte_idx: usize, bit_idx: usize) -> u64 {
        (chunk_idx * self.settings.chunk_size_bytes * 8 + byte_idx * 8 + bit_idx) as u64
    }

    /// Converts an encoded node value to (`chunk_idx`, `chunk_offset`)
    fn node_to_chunk_coords(&self, node: u64) -> (usize, u32) {
        let node = node as usize;
        let n = self.settings.states_per_chunk();
        (node / n, (node % n) as u32)
    }

    fn chunk_offset_to_bit_coords(&self, chunk_offset: u32) -> (usize, usize) {
        let byte_idx = (chunk_offset / 8) as usize;
        let bit_idx = (chunk_offset % 8) as usize;
        (byte_idx, bit_idx)
    }

    fn create_initial_update_files(&self, next: HashSet<u64>, depth: usize) {
        let mut rng = rand::thread_rng();
        let num_chunks = self.settings.num_array_chunks();

        self.start_of_depth_init(depth - 1);

        let mut vecs = vec![Vec::new(); num_chunks];

        for val in next {
            let (chunk_idx, offset) = self.node_to_chunk_coords(val);
            vecs[chunk_idx].push(offset);
        }

        for (chunk_idx, chunk_updates) in
            vecs.into_iter().enumerate().filter(|(_, v)| !v.is_empty())
        {
            let update_bytes = bytemuck::cast_slice(&chunk_updates);
            let dir_path = self.settings.update_chunk_dir_path(depth + 1, chunk_idx);
            let file_name = Alphanumeric.sample_string(&mut rng, 16);
            let file_path = dir_path.join(file_name);
            self.locked_io
                .write_file(&file_path, update_bytes, self.settings.use_compression);
        }
    }

    fn process_chunk(
        &self,
        chunk_buffer: &mut [u8],
        create_chunk_hashsets: Option<&[&HashSet<u64>]>,
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        if let Some(d) = self.chunk_exhausted_depth(chunk_idx) {
            // If d > depth, then we must have written the exhausted file, and then the program was
            // terminated before we could delete the chunk file, and we are now re-processing the
            // same chunk. In that case, there are still states to count.
            if d <= depth {
                tracing::debug!("chunk {chunk_idx} is exhausted");
                tracing::info!("depth {} chunk {chunk_idx} new 0", depth + 1);
                return 0;
            }
        }

        if let Some(hashsets) = create_chunk_hashsets {
            tracing::debug!("creating depth {depth} chunk {chunk_idx}");
            self.create_chunk(chunk_buffer, hashsets, chunk_idx);
        } else {
            tracing::debug!("reading depth {depth} chunk {chunk_idx}");
            if !self.try_read_chunk(chunk_buffer, depth, chunk_idx) {
                // No chunk file, so check that it has already been expanded
                let next_chunk_file = self.settings.chunk_file_path(depth + 1, chunk_idx);
                assert!(
                    next_chunk_file.exists(),
                    "no chunk {chunk_idx} found at depth {depth} or {}",
                    depth + 1,
                );

                // Read the number of new positions from that chunk and return it
                let new = self.read_new_positions_data_file(depth + 1, chunk_idx);
                return new;
            }
        }

        tracing::info!(
            "updating and expanding depth {depth} -> {} chunk {chunk_idx}",
            depth + 1,
        );
        let new = self.update_and_expand_chunk(chunk_buffer, depth, chunk_idx);
        self.write_new_positions_data_file(new, depth + 1, chunk_idx);

        tracing::info!("depth {} chunk {chunk_idx} new {new}", depth + 1);

        tracing::debug!("writing depth {} chunk {chunk_idx}", depth + 1);
        self.write_chunk(chunk_buffer, depth + 1, chunk_idx);

        // Check if the chunk is exhausted. If so, there are no new positions at depth `depth + 2`
        // or beyond. The depth written to the exhausted file is `depth + 1`, which is the maximum
        // depth of a state in this chunk.
        if chunk_buffer.iter().all(|&byte| byte == 0xFF) {
            tracing::debug!("marking chunk {chunk_idx} as exhausted");
            self.mark_chunk_exhausted(depth + 1, chunk_idx);
        }

        if self.settings.sync_filesystem {
            io::sync();
        }

        match self.settings.settings_provider.chunk_files_behavior(depth) {
            ChunkFilesBehavior::Delete => {
                tracing::debug!("deleting depth {depth} chunk {chunk_idx}");
                self.delete_chunk_file(depth, chunk_idx);
            }
            ChunkFilesBehavior::Keep => {
                tracing::debug!("backing up depth {depth} chunk {chunk_idx}");
                self.back_up_chunk_file(depth, chunk_idx);
            }
        }

        tracing::debug!(
            "deleting update files for depth {depth} -> {} chunk {chunk_idx}",
            depth + 1,
        );
        self.update_file_manager
            .delete_update_files(depth + 1, chunk_idx);

        match self
            .settings
            .settings_provider
            .update_files_behavior(depth + 1)
        {
            UpdateFilesBehavior::DontMerge | UpdateFilesBehavior::MergeAndDelete => {
                tracing::debug!(
                    "deleting update arrays for depth {depth} -> {} chunk {chunk_idx}",
                    depth + 1,
                );
                self.update_file_manager
                    .delete_update_arrays(depth + 1, chunk_idx);
            }
            UpdateFilesBehavior::MergeAndKeep => {
                tracing::debug!(
                    "backing up update arrays for depth {depth} -> {} chunk {chunk_idx}",
                    depth + 1,
                );
                self.update_file_manager
                    .back_up_update_arrays(depth + 1, chunk_idx);
            }
        }

        new
    }

    fn start_of_depth_init(&self, depth: usize) {
        for root_idx in 0..self.settings.root_directories.len() {
            let dir_path = self.settings.chunk_dir_path(depth + 1, root_idx);
            tracing::trace!("creating directory {dir_path:?}");
            std::fs::create_dir_all(&dir_path).unwrap();
        }

        for chunk_idx in 0..self.settings.num_array_chunks() {
            let dir_path = self.settings.update_chunk_dir_path(depth + 2, chunk_idx);
            tracing::trace!("creating directory {dir_path:?}");
            std::fs::create_dir_all(&dir_path).unwrap();

            let dir_path = self
                .settings
                .update_array_chunk_dir_path(depth + 2, chunk_idx);
            tracing::trace!("creating directory {dir_path:?}");
            std::fs::create_dir_all(&dir_path).unwrap();
        }

        let dir_path = self.settings.new_positions_data_dir_path(depth + 2);
        tracing::trace!("creating directory {dir_path:?}");
        std::fs::create_dir_all(&dir_path).unwrap();
    }

    fn end_of_depth_cleanup(&self, depth: usize) {
        // We now have the array at depth `depth + 1`, and update files/arrays for depth
        // `depth + 2`, so we can delete the directories (which should be empty) for the
        // previous depth.
        for root_idx in 0..self.settings.root_directories.len() {
            tracing::debug!("deleting root directory {root_idx} depth {depth} chunk files");
            let dir_path = self.settings.chunk_dir_path(depth, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }

            tracing::debug!(
                "deleting root directory {root_idx} depth {} update files",
                depth + 1,
            );
            let dir_path = self.settings.update_depth_dir_path(depth + 1, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }

            tracing::debug!(
                "deleting root directory {root_idx} depth {} update arrays",
                depth + 1,
            );
            let dir_path = self.settings.update_array_dir_path(depth + 1, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }
        }

        tracing::debug!("deleting depth {} new positions files", depth + 1);
        self.delete_new_positions_data_dir(depth + 1);
    }

    fn do_iteration(&self, create_chunk_hashsets: Option<&[&HashSet<u64>]>, depth: usize) -> u64 {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum ChunkState {
            NotExpanded,
            CurrentlyExpanding,
            Expanded,
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum UpdateFileState {
            NotMerging,
            CurrentlyMerging,
        }

        let chunk_states = Arc::new(RwLock::new(vec![
            ChunkState::NotExpanded;
            self.settings.num_array_chunks()
        ]));
        let update_file_states = Arc::new(RwLock::new(vec![
            UpdateFileState::NotMerging;
            self.settings.num_array_chunks()
        ]));

        self.write_state(State::Iteration { depth });

        self.start_of_depth_init(depth);

        // If this is the first time that `do_iteration` was called, then we will need to fill the
        // chunk buffers. After the first, they should already be full, so this should do nothing.
        self.chunk_buffers.fill(self.settings.chunk_size_bytes);

        let new_states = Arc::new(Mutex::new(0u64));

        let pair = Arc::new((Mutex::new(false), Condvar::new()));

        std::thread::scope(|s| {
            let threads = (0..self.settings.threads)
                .map(|t| {
                    let new_states = new_states.clone();
                    let chunk_states = chunk_states.clone();
                    let update_file_states = update_file_states.clone();

                    let pair = pair.clone();

                    ThreadBuilder::new()
                        .name({
                            let digits = self.settings.threads.ilog10() as usize + 1;
                            format!("bfs-{t:0>digits$}")
                        })
                        .spawn_scoped(s, move || loop {
                            let (lock, cvar) = &*pair;
                            let mut has_work = lock.lock();

                            tracing::debug!("waiting for work");

                            // Wait for work
                            while !*has_work {
                                cvar.wait(&mut has_work);
                            }

                            tracing::debug!("checking for work");

                            *has_work = false;
                            drop(has_work);

                            // If everything is done, notify all and break
                            let chunk_states_read = chunk_states.read();
                            if chunk_states_read
                                .iter()
                                .all(|&state| state == ChunkState::Expanded)
                            {
                                *lock.lock() = true;
                                cvar.notify_all();
                                break;
                            }
                            drop(chunk_states_read);

                            // Check for update files to merge

                            // Amount of space remaining on each disk
                            let mut available_space = self
                                .settings
                                .root_directories
                                .iter()
                                .map(|path| fs4::available_space(path).unwrap_or(u64::MAX))
                                .collect::<Vec<_>>();

                            // Which chunk has the greatest update file size, and is not currently
                            // being merged
                            let mut largest_not_merged_per_drive =
                                vec![None; self.settings.root_directories.len()];

                            let updates_size = self.update_file_manager.all_files_size(depth + 2);

                            // Calculate how much space would be available after all the currently
                            // running update merges are complete, and find the chunks with the
                            // largest update file size that is not currently being merged on each
                            // drive
                            for (chunk_idx, (&state, &size)) in update_file_states
                                .read()
                                .iter()
                                .zip(updates_size.iter())
                                .enumerate()
                            {
                                let chunk_root_idx =
                                    self.settings.settings_provider.chunk_root_idx(chunk_idx);
                                if state == UpdateFileState::CurrentlyMerging {
                                    let a = available_space[chunk_root_idx];
                                    available_space[chunk_root_idx] = a.saturating_add(size);
                                } else if size
                                    > largest_not_merged_per_drive[chunk_root_idx]
                                        .map_or(0, |(_, size)| size)
                                {
                                    largest_not_merged_per_drive[chunk_root_idx] =
                                        Some((chunk_idx, size));
                                }
                            }

                            // Choose the disk with the least available space, after the current
                            // update merges
                            let disk_to_use = available_space
                                .iter()
                                .enumerate()
                                .min_by_key(|&(_, &space)| space)
                                .filter(|&(_, &space)| {
                                    space < self.settings.available_disk_space_limit
                                })
                                .map(|(i, _)| i);

                            // Choose the chunk with the largest update file size on the disk that
                            // is not currently being merged
                            let chunk_idx = disk_to_use.and_then(|root_idx| {
                                largest_not_merged_per_drive[root_idx]
                                    .map(|(chunk_idx, _)| chunk_idx)
                            });

                            if let Some(chunk_idx) = chunk_idx {
                                // Set the state to currently merging
                                let mut update_file_states_write = update_file_states.write();
                                if update_file_states_write[chunk_idx]
                                    == UpdateFileState::NotMerging
                                {
                                    update_file_states_write[chunk_idx] =
                                        UpdateFileState::CurrentlyMerging;
                                } else {
                                    // Another thread got here first
                                    continue;
                                }
                                drop(update_file_states_write);

                                *lock.lock() = true;
                                cvar.notify_one();

                                // Get a chunk buffer
                                let mut chunk_buffer = self.chunk_buffers.take().unwrap();

                                self.merge_update_files(&mut chunk_buffer, depth + 2, chunk_idx);

                                // Set the state back to not merging
                                let mut update_file_states_write = update_file_states.write();
                                update_file_states_write[chunk_idx] = UpdateFileState::NotMerging;
                                drop(update_file_states_write);

                                // Put the chunk buffer back
                                self.chunk_buffers.put(chunk_buffer);

                                *lock.lock() = true;
                                cvar.notify_one();
                                continue;
                            }

                            // Check for chunks to expand
                            let chunk_states_read = chunk_states.read();
                            let chunk_idx = chunk_states_read
                                .iter()
                                .position(|&state| state == ChunkState::NotExpanded);
                            drop(chunk_states_read);

                            if let Some(chunk_idx) = chunk_idx {
                                // Set the state to currently expanding
                                let mut chunk_states_write = chunk_states.write();
                                if chunk_states_write[chunk_idx] == ChunkState::NotExpanded {
                                    chunk_states_write[chunk_idx] = ChunkState::CurrentlyExpanding;
                                } else {
                                    // Another thread got here first
                                    continue;
                                }
                                drop(chunk_states_write);

                                *lock.lock() = true;
                                cvar.notify_one();

                                // Get a chunk buffer
                                let mut chunk_buffer = self.chunk_buffers.take().unwrap();

                                // Process the chunk
                                let chunk_new = self.process_chunk(
                                    &mut chunk_buffer,
                                    create_chunk_hashsets,
                                    depth,
                                    chunk_idx,
                                );

                                *new_states.lock() += chunk_new;

                                // Set the state to expanded
                                let mut chunk_states_write = chunk_states.write();
                                chunk_states_write[chunk_idx] = ChunkState::Expanded;
                                drop(chunk_states_write);

                                // Put the chunk buffer back
                                self.chunk_buffers.put(chunk_buffer);

                                *lock.lock() = true;
                                cvar.notify_one();
                                continue;
                            }
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let (lock, cvar) = &*pair;
            let mut has_work = lock.lock();
            *has_work = true;
            cvar.notify_one();
            drop(has_work);

            threads.into_iter().for_each(|t| t.join().unwrap());
        });

        self.update_file_manager.write_all();

        let new = *new_states.lock();
        tracing::info!("depth {} new {new}", depth + 1);

        if self
            .settings
            .settings_provider
            .update_files_behavior(depth + 2)
            .should_merge()
        {
            self.write_state(State::MergeUpdateFiles { depth });
            self.merge_all_update_files(depth + 2);
        }

        self.write_state(State::Cleanup { depth });

        if self.settings.sync_filesystem {
            io::sync();
        }

        self.end_of_depth_cleanup(depth);

        new
    }

    fn run_from_start(&self) {
        let (old, current, next, mut depth) = match self.in_memory_bfs() {
            InMemoryBfsResult::Complete => return,
            InMemoryBfsResult::OutOfMemory {
                old,
                current,
                next,
                depth,
            } => (old, current, next, depth),
        };

        tracing::info!("starting disk BFS");

        self.create_initial_update_files(next, depth);

        let new_positions = self.do_iteration(Some(&[&old, &current]), depth);

        if new_positions == 0 {
            self.write_state(State::Done);
            return;
        }

        drop(old);
        drop(current);

        depth += 1;

        while self.do_iteration(None, depth) != 0 {
            depth += 1;
        }

        self.write_state(State::Done);
    }

    pub(crate) fn run(&self) {
        match self.read_state() {
            Some(s) => match s {
                State::Iteration { mut depth } => {
                    // Initialize update file manager with the current update file sizes
                    self.update_file_manager.try_read_sizes_from_disk();

                    while self.do_iteration(None, depth) != 0 {
                        depth += 1;
                    }

                    self.write_state(State::Done);
                }
                State::MergeUpdateFiles { mut depth } => {
                    // Initialize update file manager with the current update file sizes
                    self.update_file_manager.try_read_sizes_from_disk();

                    self.chunk_buffers.fill(self.settings.chunk_size_bytes);

                    if self
                        .settings
                        .settings_provider
                        .update_files_behavior(depth + 2)
                        .should_merge()
                    {
                        self.merge_all_update_files(depth + 2);
                    }

                    self.write_state(State::Cleanup { depth });

                    if self.settings.sync_filesystem {
                        io::sync();
                    }

                    self.end_of_depth_cleanup(depth);
                    depth += 1;

                    while self.do_iteration(None, depth) != 0 {
                        depth += 1;
                    }

                    self.write_state(State::Done);
                }
                State::Cleanup { mut depth } => {
                    // Initialize update file manager with the current update file sizes
                    self.update_file_manager.try_read_sizes_from_disk();

                    if self.settings.sync_filesystem {
                        io::sync();
                    }

                    self.end_of_depth_cleanup(depth);
                    depth += 1;

                    while self.do_iteration(None, depth) != 0 {
                        depth += 1;
                    }

                    self.write_state(State::Done);
                }
                State::Done => return,
            },
            None => self.run_from_start(),
        }

        self.write_state(State::Done);
    }
}
