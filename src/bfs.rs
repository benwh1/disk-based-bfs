use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Read, Write},
    path::PathBuf,
    sync::{Arc, Condvar, Mutex, RwLock},
};

use cityhasher::{CityHasher, HashSet};
use rand::distributions::{Alphanumeric, DistString};
use serde_derive::{Deserialize, Serialize};

use crate::{callback::BfsCallback, settings::BfsSettings};

pub enum InMemoryBfsResult {
    Complete,
    OutOfMemory {
        old: HashSet<u64, CityHasher>,
        current: HashSet<u64, CityHasher>,
        next: HashSet<u64>,
        depth: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum State {
    Iteration { depth: usize },
    Cleanup { depth: usize },
    Done,
}

#[derive(Clone)]
struct ChunkBufferList {
    buffers: Arc<Mutex<Vec<Option<Vec<u8>>>>>,
}

impl ChunkBufferList {
    fn new_empty(num_buffers: usize) -> Self {
        let buffers = vec![None; num_buffers];
        let buffers = Arc::new(Mutex::new(buffers));
        Self { buffers }
    }

    fn fill(&self, buffer_size: usize) {
        let mut lock = self.buffers.lock().unwrap();
        for buf in lock.iter_mut() {
            if buf.is_none() {
                buf.replace(vec![0; buffer_size]);
            }
        }
    }

    fn take(&self) -> Option<Vec<u8>> {
        let mut lock = self.buffers.lock().unwrap();
        lock.iter_mut().find_map(|buf| buf.take())
    }

    fn put(&self, buffer: Vec<u8>) {
        let mut lock = self.buffers.lock().unwrap();
        for buf in lock.iter_mut() {
            if buf.is_none() {
                buf.replace(buffer);
                return;
            }
        }
    }
}

struct AvailableUpdateBlock {
    updates: Vec<u32>,
}

struct FilledUpdateBlock {
    updates: Vec<u32>,
    depth: usize,
    chunk_idx: usize,
}

impl AvailableUpdateBlock {
    fn new(capacity: usize) -> Self {
        Self {
            updates: Vec::with_capacity(capacity),
        }
    }

    fn into_filled(self, depth: usize, chunk_idx: usize) -> FilledUpdateBlock {
        FilledUpdateBlock {
            updates: self.updates,
            depth,
            chunk_idx,
        }
    }

    fn push(&mut self, update: u32) {
        self.updates.push(update);
    }
}

impl FilledUpdateBlock {
    fn clear(mut self) -> AvailableUpdateBlock {
        self.updates.clear();

        AvailableUpdateBlock {
            updates: self.updates,
        }
    }
}

struct UpdateBlockList<'a> {
    settings: &'a BfsSettings,
    available_blocks: Vec<AvailableUpdateBlock>,
    filled_blocks: Vec<FilledUpdateBlock>,
}

impl<'a> UpdateBlockList<'a> {
    fn new(settings: &'a BfsSettings) -> Self {
        let num_blocks = settings.threads * settings.num_array_chunks();
        let available_blocks = (0..num_blocks)
            .map(|_| AvailableUpdateBlock::new(settings.update_capacity_per_vec()))
            .collect::<Vec<_>>();
        let filled_blocks = Vec::new();

        Self {
            settings,
            available_blocks,
            filled_blocks,
        }
    }

    fn write_all(&mut self) {
        tracing::info!("writing {} update files", self.filled_blocks.len());

        // Sort the updates so that all the blocks that belong in the same file are consecutive,
        // so that we can use `chunk_by_mut` to group them together
        self.filled_blocks
            .sort_unstable_by_key(|block| (block.depth, block.chunk_idx));

        for chunk in self
            .filled_blocks
            .chunk_by_mut(|b1, b2| (b1.depth, b1.chunk_idx) == (b2.depth, b2.chunk_idx))
        {
            let depth = chunk[0].depth;
            let chunk_idx = chunk[0].chunk_idx;

            let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
            std::fs::create_dir_all(&dir_path).unwrap();

            let mut rng = rand::thread_rng();
            let file_name = Alphanumeric.sample_string(&mut rng, 16);
            let mut file_path = dir_path.join(file_name);
            file_path.set_extension("dat");

            let file_path_tmp = file_path.with_extension("tmp");
            let file = File::create(&file_path_tmp).unwrap();
            let mut writer = BufWriter::with_capacity(self.settings.buf_io_capacity, file);

            // Write all the blocks in the chunk to a single big update file
            for block in chunk {
                for &val in &block.updates {
                    if writer.write(&val.to_le_bytes()).unwrap() != 4 {
                        panic!("failed to write to update file");
                    }
                }
            }

            drop(writer);

            std::fs::rename(file_path_tmp, &file_path).unwrap();

            tracing::debug!("finished writing file {file_path:?}");
        }

        for block in self.filled_blocks.drain(..) {
            self.available_blocks.push(block.clear());
        }
    }

    fn take(&mut self) -> AvailableUpdateBlock {
        if let Some(block) = self.available_blocks.pop() {
            return block;
        }

        self.write_all();

        // All blocks will be available now
        self.available_blocks.pop().unwrap()
    }

    fn put(&mut self, block: FilledUpdateBlock) {
        self.filled_blocks.push(block);
    }
}

struct UpdateManager<'a> {
    settings: &'a BfsSettings,
    sizes: RwLock<HashMap<usize, Vec<u64>>>,
    size_file_lock: Mutex<()>,
    update_blocks: Mutex<UpdateBlockList<'a>>,
}

impl<'a> UpdateManager<'a> {
    fn new(settings: &'a BfsSettings) -> Self {
        Self {
            settings,
            sizes: RwLock::new(HashMap::new()),
            size_file_lock: Mutex::new(()),
            update_blocks: Mutex::new(UpdateBlockList::new(settings)),
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

        let file_path = self.settings.update_files_size_file_path();
        let file_path_tmp = file_path.with_extension("tmp");

        let mut file = File::create(&file_path_tmp).unwrap();
        file.write_all(str.as_ref()).unwrap();

        drop(file);

        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn try_read_sizes_from_disk(&self) {
        let file_path = self.settings.update_files_size_file_path();
        let Ok(str) = std::fs::read_to_string(file_path) else {
            return;
        };
        let hashmap = serde_json::from_str(&str).unwrap();

        let mut lock = self.sizes.write().unwrap();
        *lock = hashmap;
    }

    fn take_n(&self, n: usize) -> Vec<AvailableUpdateBlock> {
        let mut update_blocks_lock = self.update_blocks.lock().unwrap();

        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            blocks.push(update_blocks_lock.take());
        }

        blocks
    }

    fn put(&self, block: FilledUpdateBlock) {
        self.update_blocks.lock().unwrap().put(block);
    }

    fn flush(&self) {
        let mut update_blocks_lock = self.update_blocks.lock().unwrap();
        let mut sizes_lock = self.sizes.write().unwrap();

        for block in &update_blocks_lock.filled_blocks {
            let bytes = block.updates.len() as u64 * 4;
            sizes_lock
                .entry(block.depth)
                .and_modify(|entry| entry[block.chunk_idx] += bytes)
                .or_insert_with(|| vec![0; self.settings.num_array_chunks()]);
        }

        drop(sizes_lock);

        update_blocks_lock.write_all();

        drop(update_blocks_lock);

        self.write_sizes_to_disk();
    }

    fn delete_update_files(&self, depth: usize, chunk_idx: usize) {
        let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);

        if !dir_path.exists() {
            return;
        }

        std::fs::remove_dir_all(dir_path).unwrap();

        let mut lock = self.sizes.write().unwrap();
        lock.entry(depth).and_modify(|entry| entry[chunk_idx] = 0);
    }

    fn delete_used_update_files(&self, depth: usize, chunk_idx: usize) {
        let mut bytes_deleted = 0;

        let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
        let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
            return;
        };

        for file_path in read_dir
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("used"))
        {
            bytes_deleted += file_path.metadata().unwrap().len();
            std::fs::remove_file(file_path).unwrap();
        }

        let mut lock = self.sizes.write().unwrap();
        lock.entry(depth)
            .and_modify(|entry| entry[chunk_idx] = entry[chunk_idx].saturating_sub(bytes_deleted));
    }

    fn files_size(&self, depth: usize, chunk_idx: usize) -> u64 {
        let read_lock = self.sizes.read().unwrap();
        read_lock.get(&depth).map_or(0, |sizes| sizes[chunk_idx])
    }

    fn mark_filled_and_replace(
        &self,
        upd: &mut AvailableUpdateBlock,
        depth: usize,
        chunk_idx: usize,
    ) {
        let new = self.update_blocks.lock().unwrap().take();
        let old = std::mem::replace(upd, new).into_filled(depth, chunk_idx);
        self.put(old);
    }
}

pub struct Bfs<
    'a,
    Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    const EXPANSION_NODES: usize,
> {
    settings: &'a BfsSettings,
    expander: Expander,
    callback: Callback,
    chunk_buffers: ChunkBufferList,
    update_file_manager: UpdateManager<'a>,
}

impl<
        'a,
        Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
        Callback: BfsCallback + Clone + Sync,
        const EXPANSION_NODES: usize,
    > Bfs<'a, Expander, Callback, EXPANSION_NODES>
{
    pub fn new(settings: &'a BfsSettings, expander: Expander, callback: Callback) -> Self {
        let chunk_buffers = ChunkBufferList::new_empty(settings.threads);
        let update_file_manager = UpdateManager::new(settings);

        Self {
            settings,
            expander,
            callback,
            chunk_buffers,
            update_file_manager,
        }
    }

    fn read_new_positions_data_file(&self, depth: usize, chunk_idx: usize) -> u64 {
        let file_path = self.settings.new_positions_data_file_path(depth, chunk_idx);
        let mut file = File::open(file_path).unwrap();
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf).unwrap();
        u64::from_le_bytes(buf)
    }

    fn write_new_positions_data_file(&self, new: u64, depth: usize, chunk_idx: usize) {
        let dir_path = self.settings.new_positions_data_dir_path(depth);
        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = self
            .settings
            .new_positions_data_file_path(depth, chunk_idx)
            .with_extension("tmp");
        let mut file = File::create(&file_path_tmp).unwrap();
        file.write_all(&new.to_le_bytes()).unwrap();
        drop(file);

        let file_path = self.settings.new_positions_data_file_path(depth, chunk_idx);
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn delete_new_positions_data_dir(&self, depth: usize) {
        let file_path = self.settings.new_positions_data_dir_path(depth);
        if file_path.exists() {
            std::fs::remove_dir_all(file_path).unwrap();
        }
    }

    fn read_state(&self) -> Option<State> {
        let file_path = self.settings.state_file_path();
        let str = std::fs::read_to_string(file_path).ok()?;
        serde_json::from_str(&str).ok()
    }

    fn write_state(&self, state: State) {
        let str = serde_json::to_string(&state).unwrap();

        let file_path_tmp = self.settings.state_file_path().with_extension("tmp");
        let mut file = File::create(&file_path_tmp).unwrap();

        file.write_all(str.as_ref()).unwrap();

        drop(file);

        let file_path = self.settings.state_file_path();
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn try_read_chunk(&self, chunk_buffer: &mut [u8], depth: usize, chunk_idx: usize) -> bool {
        let file_path = self.settings.chunk_file_path(depth, chunk_idx);
        let Ok(mut file) = File::open(&file_path) else {
            return false;
        };

        // Check that the file size is correct
        let expected_size = self.settings.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        tracing::debug!("reading file {file_path:?}");
        file.read_exact(chunk_buffer).unwrap();
        tracing::debug!("finished reading file {file_path:?}");

        true
    }

    fn try_read_update_array(
        &self,
        update_buffer: &mut [u8],
        depth: usize,
        chunk_idx: usize,
    ) -> bool {
        let file_path = self.settings.update_array_file_path(depth, chunk_idx);
        if !file_path.exists() {
            return false;
        }

        let mut file = File::open(&file_path).unwrap();

        // Check that the file size is correct
        let expected_size = self.settings.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        tracing::debug!("reading file {file_path:?}");
        file.read_exact(update_buffer).unwrap();
        tracing::debug!("finished reading file {file_path:?}");

        true
    }

    fn write_update_array(&self, update_buffer: &[u8], depth: usize, chunk_idx: usize) {
        let dir_path = self.settings.update_array_dir_path(depth, chunk_idx);

        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = dir_path.join(format!("update-chunk-{chunk_idx}.dat.tmp"));
        let mut file = File::create(&file_path_tmp).unwrap();

        tracing::debug!("writing file {file_path_tmp:?}");
        file.write_all(update_buffer).unwrap();
        tracing::debug!("finished writing file {file_path_tmp:?}");

        drop(file);

        let file_path = self.settings.update_array_file_path(depth, chunk_idx);
        std::fs::rename(file_path_tmp, file_path).unwrap();
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
        let dir_path = self.settings.chunk_dir_path(depth, chunk_idx);

        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = dir_path.join(format!("chunk-{chunk_idx}.dat.tmp"));
        let mut file = File::create(&file_path_tmp).unwrap();

        tracing::debug!("writing file {file_path_tmp:?}");
        file.write_all(chunk_buffer).unwrap();
        tracing::debug!("finished writing file {file_path_tmp:?}");

        drop(file);

        let file_path = self.settings.chunk_file_path(depth, chunk_idx);
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn delete_chunk_file(&self, depth: usize, chunk_idx: usize) {
        let file_path = self.settings.chunk_file_path(depth, chunk_idx);
        if file_path.exists() {
            std::fs::remove_file(file_path).unwrap();
        }
    }

    fn delete_update_array(&self, depth: usize, chunk_idx: usize) {
        let file_path = self.settings.update_array_file_path(depth, chunk_idx);
        if file_path.exists() {
            std::fs::remove_file(file_path).unwrap();
        }
    }

    fn exhausted_chunk_dir_path(&self) -> PathBuf {
        self.settings.root_dir(0).join("exhausted-chunks")
    }

    fn exhausted_chunk_file_path(&self, chunk_idx: usize) -> PathBuf {
        self.exhausted_chunk_dir_path()
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    fn mark_chunk_exhausted(&self, depth: usize, chunk_idx: usize) {
        let dir_path = self.exhausted_chunk_dir_path();
        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = self
            .exhausted_chunk_file_path(chunk_idx)
            .with_extension("tmp");
        let mut file = File::create(&file_path_tmp).unwrap();
        file.write_all(&depth.to_le_bytes()).unwrap();
        drop(file);

        let file_path = self.exhausted_chunk_file_path(chunk_idx);
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn chunk_exhausted_depth(&self, chunk_idx: usize) -> Option<usize> {
        let file_path = self.exhausted_chunk_file_path(chunk_idx);

        if !file_path.exists() {
            return None;
        }

        let mut file = File::open(file_path).unwrap();
        let mut buf = [0u8; std::mem::size_of::<usize>()];
        file.read_exact(&mut buf).unwrap();

        Some(usize::from_le_bytes(buf))
    }

    fn compress_update_files(&self, update_buffer: &mut [u8], depth: usize, chunk_idx: usize) {
        // If there is already an update array file, read it into the buffer first so we don't
        // overwrite the old array. Otherwise, just fill with zeros.
        if !self.try_read_update_array(update_buffer, depth, chunk_idx) {
            update_buffer.fill(0);
        }

        let dir_path = self.settings.update_chunk_dir_path(depth, chunk_idx);
        let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
            return;
        };

        for file_path in read_dir.flatten().map(|entry| entry.path()).filter(|path| {
            let ext = path.extension().and_then(|ext| ext.to_str());
            // Look for "used" as well, in case we restart the program while this loop is
            // running and need to re-read all the update files
            ext == Some("dat") || ext == Some("used")
        }) {
            let bytes = std::fs::read(&file_path).unwrap();
            assert_eq!(bytes.len() % 4, 0);

            // Group the bytes into groups of 4
            for chunk_offset in bytes
                .array_chunks()
                .map(|&chunk: &[u8; 4]| u32::from_le_bytes(chunk))
            {
                let (byte_idx, bit_idx) = self.chunk_offset_to_bit_coords(chunk_offset);
                update_buffer[byte_idx] |= 1 << bit_idx;
            }

            // Rename the file to mark it as used
            let file_path_used = file_path.with_extension("used");
            std::fs::rename(file_path, file_path_used).unwrap();
        }

        self.write_update_array(update_buffer, depth, chunk_idx);
        self.update_file_manager
            .delete_used_update_files(depth, chunk_idx);
    }

    fn update_and_expand_chunk(
        &self,
        chunk_buffer: &mut [u8],
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0;
        let mut updates = self
            .update_file_manager
            .take_n(self.settings.num_array_chunks());

        let mut callback = self.callback.clone();

        new_positions += self.update_and_expand_from_update_files(
            chunk_buffer,
            &mut updates,
            &mut callback,
            depth,
            chunk_idx,
        );

        new_positions += self.update_and_expand_from_update_array(
            chunk_buffer,
            &mut updates,
            &mut callback,
            depth,
            chunk_idx,
        );

        callback.end_of_chunk(depth + 1, chunk_idx);

        // Write remaining update files
        for (idx, upd) in updates.into_iter().enumerate() {
            self.update_file_manager
                .put(upd.into_filled(depth + 2, idx));
        }

        new_positions
    }

    fn check_update_vec_capacity(&self, updates: &mut [AvailableUpdateBlock], depth: usize) {
        // Check if any of the update vecs may go over capacity
        let max_new_nodes = self.settings.capacity_check_frequency * EXPANSION_NODES;

        for (idx, upd) in updates.iter_mut().enumerate() {
            if upd.updates.len() + max_new_nodes > upd.updates.capacity() {
                // Possible to reach capacity on the next block of expansions
                self.update_file_manager
                    .mark_filled_and_replace(upd, depth, idx);
            }
        }
    }

    fn update_and_expand_from_update_files(
        &self,
        chunk_buffer: &mut [u8],
        updates: &mut [AvailableUpdateBlock],
        callback: &mut Callback,
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0u64;

        let mut expander = self.expander.clone();
        let mut expanded = [0u64; EXPANSION_NODES];

        let dir_path = self.settings.update_chunk_dir_path(depth + 1, chunk_idx);
        let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
            // No update files, so nothing to do
            return 0;
        };

        for file_path in read_dir.flatten().map(|entry| entry.path()) {
            let bytes = std::fs::read(&file_path).unwrap();
            assert_eq!(bytes.len() % 4, 0);

            // Group the bytes into groups of 4
            for chunk_offset in bytes
                .array_chunks()
                .map(|&chunk: &[u8; 4]| u32::from_le_bytes(chunk))
            {
                let (byte_idx, bit_idx) = self.chunk_offset_to_bit_coords(chunk_offset);
                let byte = chunk_buffer[byte_idx];

                if (byte >> bit_idx) & 1 == 0 {
                    chunk_buffer[byte_idx] |= 1 << bit_idx;
                    new_positions += 1;

                    let encoded = self.bit_coords_to_node(chunk_idx, byte_idx, bit_idx);
                    callback.new_state(depth + 1, encoded);

                    if new_positions as usize % self.settings.capacity_check_frequency == 0 {
                        self.check_update_vec_capacity(updates, depth + 2);
                    }

                    // Expand the node
                    expander(encoded, &mut expanded);

                    for node in expanded {
                        let (idx, offset) = self.node_to_chunk_coords(node);
                        updates[idx].push(offset);
                    }
                }
            }
        }

        new_positions
    }

    fn update_and_expand_from_update_array(
        &self,
        chunk_buffer: &mut [u8],
        updates: &mut [AvailableUpdateBlock],
        callback: &mut Callback,
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0u64;

        let mut expander = self.expander.clone();
        let mut expanded = [0u64; EXPANSION_NODES];

        let file_path = self.settings.update_array_file_path(depth + 1, chunk_idx);
        if !file_path.exists() {
            return 0;
        }

        let file_len = file_path.metadata().unwrap().len() as usize;
        assert_eq!(file_len, self.settings.chunk_size_bytes);

        tracing::debug!("reading file {file_path:?}");
        let update_array_bytes = std::fs::read(&file_path).unwrap();
        tracing::debug!("finished reading file {file_path:?}");

        for (byte_idx, &update_byte) in update_array_bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let chunk_byte = chunk_buffer[byte_idx];
                if (update_byte >> bit_idx) & 1 == 1 && (chunk_byte >> bit_idx) & 1 == 0 {
                    chunk_buffer[byte_idx] |= 1 << bit_idx;
                    new_positions += 1;

                    let encoded = self.bit_coords_to_node(chunk_idx, byte_idx, bit_idx);
                    callback.new_state(depth + 1, encoded);

                    if new_positions as usize % self.settings.capacity_check_frequency == 0 {
                        self.check_update_vec_capacity(updates, depth + 2);
                    }

                    // Expand the node
                    let encoded = self.bit_coords_to_node(chunk_idx, byte_idx, bit_idx);
                    expander(encoded, &mut expanded);

                    for node in expanded {
                        let (idx, offset) = self.node_to_chunk_coords(node);
                        updates[idx].push(offset);
                    }
                }
            }
        }

        new_positions
    }

    fn in_memory_bfs(&self) -> InMemoryBfsResult {
        let max_capacity = self.settings.initial_memory_limit / 8;

        let mut old = HashSet::with_capacity_and_hasher(max_capacity / 2, CityHasher::default());
        let mut current = HashSet::with_hasher(CityHasher::default());
        let mut next = HashSet::with_hasher(CityHasher::default());

        let mut expanded = [0u64; EXPANSION_NODES];
        let mut depth = 0;

        let mut callback = self.callback.clone();

        for &state in &self.settings.initial_states {
            if current.insert(state) {
                callback.new_state(depth + 1, state);
            }
        }

        callback.end_of_chunk(0, 0);

        let mut new;
        let mut total = 1;

        tracing::info!("starting in-memory BFS");

        loop {
            new = 0;

            std::thread::scope(|s| {
                let threads = (0..self.settings.threads)
                    .map(|t| {
                        let current = &current;
                        let old = &old;

                        s.spawn(move || {
                            let mut expander = self.expander.clone();

                            let mut next = HashSet::with_hasher(CityHasher::default());

                            let thread_start =
                                self.settings.state_size / self.settings.threads as u64 * t as u64;
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
                                expander(encoded, &mut expanded);
                                for node in expanded {
                                    if !old.contains(&node) && !current.contains(&node) {
                                        next.insert(node);
                                    }
                                }
                            }

                            next
                        })
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

            tracing::info!("depth {depth} new {new}");

            // No new nodes, we are done already.
            if new == 0 {
                tracing::info!("no new nodes, done");
                return InMemoryBfsResult::Complete;
            }

            if total > max_capacity {
                tracing::info!("exceeded memory limit");
                break;
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

    /// Converts an encoded node value to (chunk_idx, byte_idx, bit_idx)
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

    /// Converts an encoded node value to (chunk_idx, chunk_offset)
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

    fn create_initial_update_files(&self, next: &HashSet<u64>, depth: usize) {
        // Write values from `next` to initial update files
        let mut update_files = (0..self.settings.num_array_chunks())
            .map(|chunk_idx| {
                let dir_path = self.settings.update_chunk_dir_path(depth + 1, chunk_idx);
                std::fs::create_dir_all(&dir_path).unwrap();

                let mut rng = rand::thread_rng();
                let file_name = Alphanumeric.sample_string(&mut rng, 16);
                let mut file_path = dir_path.join(file_name);
                file_path.set_extension("dat");

                let file = File::create(&file_path).unwrap();
                BufWriter::with_capacity(self.settings.buf_io_capacity, file)
            })
            .collect::<Vec<_>>();

        for &val in next {
            let (chunk_idx, chunk_offset) = self.node_to_chunk_coords(val);
            update_files[chunk_idx]
                .write(&chunk_offset.to_le_bytes())
                .unwrap();
        }
    }

    fn process_chunk(
        &self,
        chunk_buffer: &mut [u8],
        create_chunk_hashsets: Option<&[&HashSet<u64>]>,
        thread: usize,
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let t = thread;

        if let Some(d) = self.chunk_exhausted_depth(chunk_idx) {
            // If d > depth, then we must have written the exhausted file, and then the program was
            // terminated before we could delete the chunk file, and we are now re-processing the
            // same chunk. In that case, there are still states to count.
            if d <= depth {
                tracing::info!("[Thread {t}] chunk {chunk_idx} is exhausted");
                tracing::info!("[Thread {t}] depth {} chunk {chunk_idx} new 0", depth + 1);
                return 0;
            }
        }

        if let Some(hashsets) = create_chunk_hashsets {
            tracing::info!("[Thread {t}] creating depth {depth} chunk {chunk_idx}");
            self.create_chunk(chunk_buffer, hashsets, chunk_idx);
        } else {
            tracing::info!("[Thread {t}] reading depth {depth} chunk {chunk_idx}");
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
            "[Thread {t}] updating and expanding depth {depth} -> {} chunk {chunk_idx}",
            depth + 1
        );
        let new = self.update_and_expand_chunk(chunk_buffer, depth, chunk_idx);
        self.write_new_positions_data_file(new, depth + 1, chunk_idx);

        tracing::info!(
            "[Thread {t}] depth {} chunk {chunk_idx} new {new}",
            depth + 1
        );

        tracing::info!("[Thread {t}] writing depth {} chunk {chunk_idx}", depth + 1);
        self.write_chunk(chunk_buffer, depth + 1, chunk_idx);

        // Check if the chunk is exhausted. If so, there are no new positions at depth `depth + 2`
        // or beyond. The depth written to the exhausted file is `depth + 1`, which is the maximum
        // depth of a state in this chunk.
        if chunk_buffer.iter().all(|&byte| byte == 0xFF) {
            tracing::info!("[Thread {t}] marking chunk {chunk_idx} as exhausted");
            self.mark_chunk_exhausted(depth + 1, chunk_idx);
        }

        tracing::info!("[Thread {t}] deleting depth {depth} chunk {chunk_idx}");
        self.delete_chunk_file(depth, chunk_idx);

        tracing::info!(
            "[Thread {t}] deleting update files for depth {depth} -> {} chunk {chunk_idx}",
            depth + 1
        );
        self.update_file_manager
            .delete_update_files(depth + 1, chunk_idx);

        tracing::info!(
            "[Thread {t}] deleting update array for depth {depth} -> {} chunk {chunk_idx}",
            depth + 1
        );
        self.delete_update_array(depth + 1, chunk_idx);

        new
    }

    fn end_of_depth_cleanup(&self, depth: usize) {
        // We now have the array at depth `depth + 1`, and update files/arrays for depth
        // `depth + 2`, so we can delete the directories (which should be empty) for the
        // previous depth.
        for root_idx in 0..self.settings.root_directories.len() {
            tracing::info!("Deleting root directory {root_idx} depth {depth} chunk files");
            let dir_path = self.settings.chunk_dir_path(depth, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }

            tracing::info!(
                "Deleting root directory {root_idx} depth {} update files",
                depth + 1
            );
            let dir_path = self.settings.update_depth_dir_path(depth + 1, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }

            tracing::info!(
                "Deleting root directory {root_idx} depth {} update arrays",
                depth + 1
            );
            let dir_path = self.settings.update_array_dir_path(depth + 1, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }
        }

        tracing::info!("Deleting depth {} new positions files", depth + 1);
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
            NotCompressing,
            CurrentlyCompressing,
        }

        let chunk_states = Arc::new(RwLock::new(vec![
            ChunkState::NotExpanded;
            self.settings.num_array_chunks()
        ]));
        let update_file_states = Arc::new(RwLock::new(vec![
            UpdateFileState::NotCompressing;
            self.settings.num_array_chunks()
        ]));

        // If this is the first time that `do_iteration` was called, then we will need to fill the
        // chunk buffers. After the first, they should already be full, so this should do nothing.
        self.chunk_buffers.fill(self.settings.chunk_size_bytes);

        let new_states = Arc::new(Mutex::new(0u64));

        self.write_state(State::Iteration { depth });

        let pair = Arc::new((Mutex::new(false), Condvar::new()));

        std::thread::scope(|s| {
            let threads = (0..self.settings.threads)
                .map(|t| {
                    let new_states = new_states.clone();
                    let chunk_states = chunk_states.clone();
                    let update_file_states = update_file_states.clone();

                    let pair = pair.clone();

                    s.spawn(move || loop {
                        let (lock, cvar) = &*pair;
                        let mut has_work = lock.lock().unwrap();

                        tracing::debug!("[Thread {t}] waiting for work");

                        // Wait for work
                        while !*has_work {
                            has_work = cvar.wait(has_work).unwrap();
                        }

                        tracing::debug!("[Thread {t}] checking for work");

                        *has_work = false;
                        drop(has_work);

                        // If everything is done, notify all and break
                        let chunk_states_read = chunk_states.read().unwrap();
                        if chunk_states_read.iter().all(|&state| state == ChunkState::Expanded) {
                            *lock.lock().unwrap() = true;
                            cvar.notify_all();
                            break;
                        }
                        drop(chunk_states_read);

                        // Check for update files to compress
                        let update_file_states_read = update_file_states.read().unwrap();
                        let chunk_idx = update_file_states_read.iter().enumerate().find_map(|(i, state)| {
                            if *state == UpdateFileState::CurrentlyCompressing {
                                return None;
                            }

                            let path = self.settings.update_chunk_dir_path(depth + 2, i);

                            if !path.exists() {
                                return None;
                            }

                            let used_space = self.update_file_manager.files_size(depth+2, i);

                            if used_space <= self.settings.update_files_compression_threshold {
                                return None;
                            }

                            Some(i)
                        });
                        drop(update_file_states_read);

                        if let Some(chunk_idx) = chunk_idx {
                            // Set the state to currently compressing
                            let mut update_file_states_write = update_file_states.write().unwrap();
                            if update_file_states_write[chunk_idx] == UpdateFileState::NotCompressing {
                                update_file_states_write[chunk_idx] = UpdateFileState::CurrentlyCompressing;
                            } else {
                                // Another thread got here first
                                continue;
                            }
                            drop(update_file_states_write);

                            *lock.lock().unwrap() = true;
                            cvar.notify_one();

                            // Get a chunk buffer
                            let mut chunk_buffer = self.chunk_buffers.take().unwrap();

                            // Compress the update files
                            tracing::info!(
                                "[Thread {t}] compressing update files for depth {} -> {} chunk {chunk_idx}",
                                depth + 1,
                                depth + 2,
                            );
                            self.compress_update_files(&mut chunk_buffer, depth + 2, chunk_idx);
                            tracing::info!(
                                "[Thread {t}] finished compressing update files for depth {} -> {} chunk {chunk_idx}",
                                depth + 1,
                                depth + 2,
                            );

                            // Set the state back to not compressing
                            let mut update_file_states_write = update_file_states.write().unwrap();
                            update_file_states_write[chunk_idx] = UpdateFileState::NotCompressing;
                            drop(update_file_states_write);

                            // Put the chunk buffer back
                            self.chunk_buffers.put(chunk_buffer);

                            // Write new update file sizes to disk
                            self.update_file_manager.write_sizes_to_disk();

                            *lock.lock().unwrap() = true;
                            cvar.notify_one();
                            continue;
                        }

                        // Check for chunks to expand
                        let chunk_states_read = chunk_states.read().unwrap();
                        let chunk_idx = chunk_states_read.iter().position(|&state| {
                            state == ChunkState::NotExpanded
                        });
                        drop(chunk_states_read);

                        if let Some(chunk_idx) = chunk_idx {
                            // Set the state to currently expanding
                            let mut chunk_states_write = chunk_states.write().unwrap();
                            if chunk_states_write[chunk_idx] == ChunkState::NotExpanded {
                                chunk_states_write[chunk_idx] = ChunkState::CurrentlyExpanding;
                            } else {
                                // Another thread got here first
                                continue;
                            }
                            drop(chunk_states_write);

                            *lock.lock().unwrap() = true;
                            cvar.notify_one();

                            // Get a chunk buffer
                            let mut chunk_buffer = self.chunk_buffers.take().unwrap();

                            // Process the chunk
                            tracing::info!("[Thread {t}] processing depth {depth} chunk {chunk_idx}");
                            let chunk_new = self.process_chunk(
                                &mut chunk_buffer,
                                create_chunk_hashsets,
                                t,
                                depth,
                                chunk_idx,
                            );
                            tracing::info!("[Thread {t}] finished processing depth {depth} chunk {chunk_idx}");

                            *new_states.lock().unwrap() += chunk_new;

                            // Set the state to expanded
                            let mut chunk_states_write = chunk_states.write().unwrap();
                            chunk_states_write[chunk_idx] = ChunkState::Expanded;
                            drop(chunk_states_write);

                            // Put the chunk buffer back
                            self.chunk_buffers.put(chunk_buffer);

                            // Write new update file sizes to disk
                            self.update_file_manager.write_sizes_to_disk();

                            *lock.lock().unwrap() = true;
                            cvar.notify_one();
                            continue;
                        }
                    })
                })
                .collect::<Vec<_>>();

            let (lock, cvar) = &*pair;
            let mut has_work = lock.lock().unwrap();
            *has_work = true;
            cvar.notify_one();
            drop(has_work);

            threads.into_iter().for_each(|t| t.join().unwrap());
        });

        tracing::info!("writing remaining update files");
        self.update_file_manager.flush();

        let new = *new_states.lock().unwrap();
        tracing::info!("depth {} new {new}", depth + 1);

        self.write_state(State::Cleanup { depth });
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

        self.create_initial_update_files(&next, depth);
        drop(next);

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

    pub fn run(&self) {
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
                State::Cleanup { mut depth } => {
                    self.end_of_depth_cleanup(depth);
                    depth += 1;

                    // Initialize update file manager with the current update file sizes
                    self.update_file_manager.try_read_sizes_from_disk();

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
