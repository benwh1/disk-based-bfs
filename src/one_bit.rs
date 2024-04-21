use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use cityhasher::{CityHasher, HashSet};
use serde_derive::{Deserialize, Serialize};

use crate::callback::BfsCallback;

pub struct BfsBuilder<
    Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    const EXPANSION_NODES: usize,
> {
    expander: Option<Expander>,
    callback: Option<Callback>,
    threads: usize,
    chunk_size_bytes: Option<usize>,
    update_set_capacity: Option<usize>,
    capacity_check_frequency: Option<usize>,
    initial_states: Option<Vec<u64>>,
    state_size: Option<u64>,
    root_directories: Option<Vec<PathBuf>>,
    initial_memory_limit: Option<usize>,
    update_files_compression_threshold: Option<u64>,
}

impl<
        Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
        Callback: BfsCallback + Clone + Sync,
        const EXPANSION_NODES: usize,
    > Default for BfsBuilder<Expander, Callback, EXPANSION_NODES>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
        Callback: BfsCallback + Clone + Sync,
        const EXPANSION_NODES: usize,
    > BfsBuilder<Expander, Callback, EXPANSION_NODES>
{
    pub fn new() -> Self {
        Self {
            expander: None,
            callback: None,
            threads: 1,
            chunk_size_bytes: None,
            update_set_capacity: None,
            capacity_check_frequency: None,
            initial_states: None,
            state_size: None,
            root_directories: None,
            initial_memory_limit: None,
            update_files_compression_threshold: None,
        }
    }

    pub fn expander(mut self, expander: Expander) -> Self {
        self.expander = Some(expander);
        self
    }

    pub fn callback(mut self, callback: Callback) -> Self {
        self.callback = Some(callback);
        self
    }

    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn chunk_size_bytes(mut self, chunk_size_bytes: usize) -> Self {
        // Limit to 2^29 bytes so that we can store 32 bit values in the update files
        if chunk_size_bytes < 1 << 29 {
            self.chunk_size_bytes = Some(chunk_size_bytes);
        }
        self
    }

    pub fn update_set_capacity(mut self, update_set_capacity: usize) -> Self {
        self.update_set_capacity = Some(update_set_capacity);
        self
    }

    pub fn capacity_check_frequency(mut self, capacity_check_frequency: usize) -> Self {
        self.capacity_check_frequency = Some(capacity_check_frequency);
        self
    }

    pub fn initial_states(mut self, initial_states: &[u64]) -> Self {
        self.initial_states = Some(initial_states.to_vec());
        self
    }

    pub fn state_size(mut self, state_size: u64) -> Self {
        self.state_size = Some(state_size);
        self
    }

    pub fn root_directories(mut self, root_directories: &[PathBuf]) -> Self {
        self.root_directories = Some(root_directories.to_vec());
        self
    }

    pub fn initial_memory_limit(mut self, initial_memory_limit: usize) -> Self {
        self.initial_memory_limit = Some(initial_memory_limit);
        self
    }

    pub fn update_files_compression_threshold(
        mut self,
        update_files_compression_threshold: u64,
    ) -> Self {
        self.update_files_compression_threshold = Some(update_files_compression_threshold);
        self
    }

    pub fn build(self) -> Option<Bfs<Expander, Callback, EXPANSION_NODES>> {
        // Require that all chunks are the same size
        let chunk_size_bytes = self.chunk_size_bytes?;
        let state_size = self.state_size? as usize;
        if state_size % (8 * chunk_size_bytes) != 0 {
            return None;
        }

        Some(Bfs {
            expander: self.expander?,
            callback: self.callback?,
            threads: self.threads,
            chunk_size_bytes: self.chunk_size_bytes?,
            update_set_capacity: self.update_set_capacity?,
            capacity_check_frequency: self.capacity_check_frequency?,
            initial_states: self.initial_states?,
            state_size: self.state_size?,
            root_directories: self.root_directories?,
            initial_memory_limit: self.initial_memory_limit?,
            update_files_compression_threshold: self.update_files_compression_threshold?,
        })
    }
}

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

pub struct Bfs<
    Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    const EXPANSION_NODES: usize,
> {
    expander: Expander,
    callback: Callback,
    threads: usize,
    chunk_size_bytes: usize,
    update_set_capacity: usize,
    capacity_check_frequency: usize,
    initial_states: Vec<u64>,
    state_size: u64,
    root_directories: Vec<PathBuf>,
    initial_memory_limit: usize,
    update_files_compression_threshold: u64,
}

impl<
        Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
        Callback: BfsCallback + Clone + Sync,
        const EXPANSION_NODES: usize,
    > Bfs<Expander, Callback, EXPANSION_NODES>
{
    fn array_bytes(&self) -> usize {
        self.state_size.div_ceil(8) as usize
    }

    fn num_array_chunks(&self) -> usize {
        self.array_bytes() / self.chunk_size_bytes
    }

    fn states_per_chunk(&self) -> usize {
        self.chunk_size_bytes * 8
    }

    fn root_dir(&self, chunk_idx: usize) -> &Path {
        &self.root_directories[chunk_idx % self.root_directories.len()]
    }

    fn state_file_path(&self) -> PathBuf {
        self.root_dir(0).join("state.dat")
    }

    fn update_depth_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("update")
            .join(format!("depth-{depth}"))
    }

    fn update_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.update_depth_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}"))
    }

    fn update_chunk_from_chunk_dir_path(
        &self,
        depth: usize,
        updated_chunk_idx: usize,
        from_chunk_idx: usize,
    ) -> PathBuf {
        self.update_chunk_dir_path(depth, updated_chunk_idx)
            .join(format!("from-chunk-{from_chunk_idx}"))
    }

    fn chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("array")
            .join(format!("depth-{depth}"))
    }

    fn chunk_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.chunk_dir_path(depth, chunk_idx)
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    fn update_array_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("update-array")
            .join(format!("depth-{depth}"))
    }

    fn update_array_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.update_array_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}.dat"))
    }

    fn new_positions_data_dir_path(&self, depth: usize) -> PathBuf {
        self.root_dir(0)
            .join("new-positions")
            .join(format!("depth-{depth}"))
    }

    fn new_positions_data_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.new_positions_data_dir_path(depth)
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    fn read_new_positions_data_file(&self, depth: usize, chunk_idx: usize) -> u64 {
        let file_path = self.new_positions_data_file_path(depth, chunk_idx);
        let mut file = File::open(file_path).unwrap();
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf).unwrap();
        u64::from_le_bytes(buf)
    }

    fn write_new_positions_data_file(&self, new: u64, depth: usize, chunk_idx: usize) {
        let dir_path = self.new_positions_data_dir_path(depth);
        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = self
            .new_positions_data_file_path(depth, chunk_idx)
            .with_extension("tmp");
        let mut file = File::create(&file_path_tmp).unwrap();
        file.write_all(&new.to_le_bytes()).unwrap();
        drop(file);

        let file_path = self.new_positions_data_file_path(depth, chunk_idx);
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn delete_new_positions_data_dir(&self, depth: usize) {
        let file_path = self.new_positions_data_dir_path(depth);
        if file_path.exists() {
            std::fs::remove_dir_all(file_path).unwrap();
        }
    }

    fn read_state(&self) -> Option<State> {
        let file_path = self.state_file_path();
        let str = std::fs::read_to_string(file_path).ok()?;
        serde_json::from_str(&str).ok()
    }

    fn write_state(&self, state: State) {
        let str = serde_json::to_string(&state).unwrap();
        let file_path_tmp = self.state_file_path().with_extension("tmp");
        std::fs::write(&file_path_tmp, &str).unwrap();
        let file_path = self.state_file_path();
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn try_read_chunk(&self, chunk_buffer: &mut [u8], depth: usize, chunk_idx: usize) -> bool {
        let file_path = self.chunk_file_path(depth, chunk_idx);
        let Ok(mut file) = File::open(file_path) else {
            return false;
        };

        // Check that the file size is correct
        let expected_size = self.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        file.read_exact(chunk_buffer).unwrap();

        true
    }

    fn try_read_update_array(
        &self,
        update_buffer: &mut [u8],
        depth: usize,
        chunk_idx: usize,
    ) -> bool {
        let file_path = self.update_array_file_path(depth, chunk_idx);
        if !file_path.exists() {
            return false;
        }

        let mut file = File::open(file_path).unwrap();

        // Check that the file size is correct
        let expected_size = self.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        file.read_exact(update_buffer).unwrap();

        true
    }

    fn write_update_file(
        &self,
        update_set: &mut HashSet<u32>,
        depth: usize,
        updated_chunk_idx: usize,
        from_chunk_idx: usize,
    ) {
        let dir_path =
            self.update_chunk_from_chunk_dir_path(depth, updated_chunk_idx, from_chunk_idx);

        std::fs::create_dir_all(&dir_path).unwrap();

        let part = std::fs::read_dir(&dir_path).unwrap().flatten().count();
        let file_path_tmp = dir_path.join(format!("part-{part}.dat.tmp"));
        let file = File::create(&file_path_tmp).unwrap();
        let mut writer = BufWriter::new(file);

        for val in update_set.drain() {
            if writer.write(&val.to_le_bytes()).unwrap() != 4 {
                panic!("failed to write to update file");
            }
        }

        drop(writer);

        let file_path = dir_path.join(format!("part-{part}.dat"));
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn write_update_array(&self, update_buffer: &[u8], depth: usize, chunk_idx: usize) {
        let dir_path = self.update_array_dir_path(depth, chunk_idx);

        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = dir_path.join(format!("update-chunk-{chunk_idx}.dat.tmp"));
        let mut file = File::create(&file_path_tmp).unwrap();

        file.write_all(update_buffer).unwrap();
        drop(file);

        let file_path = self.update_array_file_path(depth, chunk_idx);
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
        let dir_path = self.chunk_dir_path(depth, chunk_idx);

        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = dir_path.join(format!("chunk-{chunk_idx}.dat.tmp"));
        let mut file = File::create(&file_path_tmp).unwrap();

        file.write_all(chunk_buffer).unwrap();
        drop(file);

        let file_path = self.chunk_file_path(depth, chunk_idx);
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn delete_chunk_file(&self, depth: usize, chunk_idx: usize) {
        let file_path = self.chunk_file_path(depth, chunk_idx);
        if file_path.exists() {
            std::fs::remove_file(file_path).unwrap();
        }
    }

    fn delete_update_files(&self, depth: usize, chunk_idx: usize) {
        let dir_path = self.update_chunk_dir_path(depth, chunk_idx);
        if dir_path.exists() {
            std::fs::remove_dir_all(dir_path).unwrap();
        }
    }

    fn delete_update_array(&self, depth: usize, chunk_idx: usize) {
        let file_path = self.update_array_file_path(depth, chunk_idx);
        if file_path.exists() {
            std::fs::remove_file(file_path).unwrap();
        }
    }

    fn exhausted_chunk_dir_path(&self) -> PathBuf {
        self.root_dir(0).join("exhausted-chunks")
    }

    fn exhausted_chunk_file_path(&self, chunk_idx: usize) -> PathBuf {
        self.exhausted_chunk_dir_path()
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    fn mark_chunk_exhausted(&self, chunk_idx: usize) {
        let dir_path = self.exhausted_chunk_dir_path();
        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path = self.exhausted_chunk_file_path(chunk_idx);
        File::create(file_path).unwrap();
    }

    fn is_chunk_exhausted(&self, chunk_idx: usize) -> bool {
        self.exhausted_chunk_file_path(chunk_idx).exists()
    }

    fn compress_update_files(&self, update_buffer: &mut [u8], depth: usize, chunk_idx: usize) {
        // If there is already an update array file, read it into the buffer first so we don't
        // overwrite the old array. Otherwise, just fill with zeros.
        if !self.try_read_update_array(update_buffer, depth, chunk_idx) {
            update_buffer.fill(0);
        }

        for from_chunk_idx in 0..self.num_array_chunks() {
            let dir_path = self.update_chunk_from_chunk_dir_path(depth, chunk_idx, from_chunk_idx);

            let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
                continue;
            };

            for file_path in read_dir.flatten().map(|entry| entry.path()) {
                let file = File::open(file_path).unwrap();
                let mut reader = BufReader::new(file);

                let mut buf = [0u8; 4];

                while let Ok(bytes_read) = reader.read(&mut buf) {
                    if bytes_read == 0 {
                        break;
                    }

                    let chunk_offset = u32::from_le_bytes(buf);
                    let (byte_idx, bit_idx) = self.chunk_offset_to_bit_coords(chunk_offset);
                    update_buffer[byte_idx] |= 1 << bit_idx;
                }
            }
        }

        self.write_update_array(update_buffer, depth, chunk_idx);
        self.delete_update_files(depth, chunk_idx);
    }

    fn update_and_expand_chunk(
        &self,
        chunk_buffer: &mut [u8],
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0;

        let mut update_sets = vec![
            HashSet::<u32, CityHasher>::with_capacity_and_hasher(
                self.update_set_capacity,
                CityHasher::default()
            );
            self.num_array_chunks()
        ];

        let mut callback = self.callback.clone();

        new_positions += self.update_and_expand_from_update_files(
            chunk_buffer,
            &mut update_sets,
            &mut callback,
            depth,
            chunk_idx,
        );

        new_positions += self.update_and_expand_from_update_array(
            chunk_buffer,
            &mut update_sets,
            &mut callback,
            depth,
            chunk_idx,
        );

        callback.end_of_chunk(depth + 1, chunk_idx);

        // Write remaining update files
        for (idx, set) in update_sets.iter_mut().enumerate() {
            self.write_update_file(set, depth + 2, idx, chunk_idx);
        }

        new_positions
    }

    fn check_update_set_capacity(
        &self,
        update_sets: &mut [HashSet<u32>],
        depth: usize,
        chunk_idx: usize,
    ) {
        // Check if any of the update sets may go over capacity
        let max_new_nodes = self.capacity_check_frequency * EXPANSION_NODES;

        for (idx, set) in update_sets.iter_mut().enumerate() {
            if set.len() + max_new_nodes > set.capacity() {
                // Possible to reach capacity on the next block of expansions, so
                // write update file to disk
                self.write_update_file(set, depth, idx, chunk_idx);
                set.clear();
            }
        }
    }

    fn update_and_expand_from_update_files(
        &self,
        chunk_buffer: &mut [u8],
        update_sets: &mut [HashSet<u32>],
        callback: &mut Callback,
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0u64;

        let mut expander = self.expander.clone();
        let mut expanded = [0u64; EXPANSION_NODES];

        for i in 0..self.num_array_chunks() {
            let dir_path = self.update_chunk_from_chunk_dir_path(depth + 1, chunk_idx, i);

            let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
                continue;
            };

            for file_path in read_dir.flatten().map(|entry| entry.path()) {
                let file = File::open(file_path).unwrap();
                let expected_entries = file.metadata().unwrap().len() / 4;
                let mut reader = BufReader::new(file);

                // Read 4 bytes at a time, and update the current chunk
                let mut buf = [0u8; 4];
                let mut entries = 0;

                while let Ok(bytes_read) = reader.read(&mut buf) {
                    if bytes_read == 0 {
                        break;
                    }

                    let chunk_offset = u32::from_le_bytes(buf);
                    let (byte_idx, bit_idx) = self.chunk_offset_to_bit_coords(chunk_offset);
                    let byte = chunk_buffer[byte_idx];

                    if (byte >> bit_idx) & 1 == 0 {
                        chunk_buffer[byte_idx] |= 1 << bit_idx;
                        new_positions += 1;

                        let encoded = self.bit_coords_to_node(chunk_idx, byte_idx, bit_idx);
                        callback.new_state(depth + 1, encoded);

                        if new_positions as usize % self.capacity_check_frequency == 0 {
                            self.check_update_set_capacity(update_sets, depth + 2, chunk_idx);
                        }

                        // Expand the node
                        expander(encoded, &mut expanded);

                        for node in expanded {
                            let (idx, offset) = self.node_to_chunk_coords(node);
                            update_sets[idx].insert(offset);
                        }
                    }

                    entries += 1;
                }

                assert_eq!(entries, expected_entries);
            }
        }

        new_positions
    }

    fn update_and_expand_from_update_array(
        &self,
        chunk_buffer: &mut [u8],
        update_sets: &mut [HashSet<u32>],
        callback: &mut Callback,
        depth: usize,
        chunk_idx: usize,
    ) -> u64 {
        let mut new_positions = 0u64;

        let mut expander = self.expander.clone();
        let mut expanded = [0u64; EXPANSION_NODES];

        let file_path = self.update_array_file_path(depth + 1, chunk_idx);
        if !file_path.exists() {
            return 0;
        }

        let file = File::open(file_path).unwrap();
        let file_len = file.metadata().unwrap().len() as usize;
        assert_eq!(file_len, self.chunk_size_bytes);

        let mut reader = BufReader::new(file);

        let mut buf = [0];
        let mut entries = 0;

        while let Ok(bytes_read) = reader.read(&mut buf) {
            if bytes_read == 0 {
                break;
            }

            let update_byte = buf[0];
            let byte_idx = entries;

            for bit_idx in 0..8 {
                let chunk_byte = chunk_buffer[byte_idx];
                if (update_byte >> bit_idx) & 1 == 1 && (chunk_byte >> bit_idx) & 1 == 0 {
                    chunk_buffer[byte_idx] |= 1 << bit_idx;
                    new_positions += 1;

                    let encoded = self.bit_coords_to_node(chunk_idx, byte_idx, bit_idx);
                    callback.new_state(depth + 1, encoded);

                    if new_positions as usize % self.capacity_check_frequency == 0 {
                        self.check_update_set_capacity(update_sets, depth + 2, chunk_idx);
                    }

                    // Expand the node
                    let encoded = self.bit_coords_to_node(chunk_idx, byte_idx, bit_idx);
                    expander(encoded, &mut expanded);

                    for node in expanded {
                        let (idx, offset) = self.node_to_chunk_coords(node);
                        update_sets[idx].insert(offset);
                    }
                }
            }

            entries += 1;
        }

        assert_eq!(entries, file_len);

        new_positions
    }

    fn in_memory_bfs(&self) -> InMemoryBfsResult {
        let max_capacity = self.initial_memory_limit / 8;

        let mut old = HashSet::with_capacity_and_hasher(max_capacity / 2, CityHasher::default());
        let mut current = HashSet::with_hasher(CityHasher::default());
        let mut next = HashSet::with_hasher(CityHasher::default());

        let mut expander = self.expander.clone();
        let mut expanded = [0u64; EXPANSION_NODES];
        let mut depth = 0;

        let mut callback = self.callback.clone();

        for &state in &self.initial_states {
            if current.insert(state) {
                callback.new_state(depth + 1, state);
            }
        }

        callback.end_of_chunk(0, 0);

        let mut new;
        let mut total = 1;

        tracing::info!("starting in-memory BFS");

        loop {
            let mut callback = self.callback.clone();

            new = 0;

            for &encoded in &current {
                expander(encoded, &mut expanded);
                for node in expanded {
                    if !old.contains(&node) && !current.contains(&node) && next.insert(node) {
                        new += 1;
                        callback.new_state(depth + 1, node);
                    }
                }
            }
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
        let chunk_idx = (node / 8) / self.chunk_size_bytes;
        let byte_idx = (node / 8) % self.chunk_size_bytes;
        let bit_idx = node % 8;
        (chunk_idx, byte_idx, bit_idx)
    }

    fn bit_coords_to_node(&self, chunk_idx: usize, byte_idx: usize, bit_idx: usize) -> u64 {
        (chunk_idx * self.chunk_size_bytes * 8 + byte_idx * 8 + bit_idx) as u64
    }

    /// Converts an encoded node value to (chunk_idx, chunk_offset)
    fn node_to_chunk_coords(&self, node: u64) -> (usize, u32) {
        let node = node as usize;
        let n = self.states_per_chunk();
        (node / n, (node % n) as u32)
    }

    fn chunk_offset_to_bit_coords(&self, chunk_offset: u32) -> (usize, usize) {
        let byte_idx = (chunk_offset / 8) as usize;
        let bit_idx = (chunk_offset % 8) as usize;
        (byte_idx, bit_idx)
    }

    fn create_initial_update_files(&self, next: &HashSet<u64>, depth: usize) {
        // Write values from `next` to initial update files
        let mut update_files = (0..self.num_array_chunks())
            .map(|chunk_idx| {
                let dir_path = self.update_chunk_from_chunk_dir_path(depth + 1, chunk_idx, 0);
                std::fs::create_dir_all(&dir_path).unwrap();
                let file_path = dir_path.join("part-0.dat");
                let file = File::create(&file_path).unwrap();
                BufWriter::new(file)
            })
            .collect::<Vec<_>>();

        for &val in next {
            let (chunk_idx, chunk_offset) = self.node_to_chunk_coords(val);
            update_files[chunk_idx]
                .write(&chunk_offset.to_le_bytes())
                .unwrap();
        }
    }

    fn process_chunk_group(
        &self,
        chunk_buffers: &mut [Vec<u8>],
        create_chunk_hashsets: Option<&[&HashSet<u64>]>,
        depth: usize,
        group_idx: usize,
    ) -> u64 {
        let mut new_positions = 0;

        std::thread::scope(|s| {
            let threads = (0..self.threads)
                .map(|t| {
                    let mut chunk_buffer = std::mem::take(&mut chunk_buffers[t]);

                    s.spawn(move || {
                        let chunk_idx = group_idx + t;

                        // If the number of chunks isn't divisible by the number of
                        // threads, then `chunk_idx` may be out of bounds during the last
                        // group of threads.
                        if chunk_idx >= self.num_array_chunks() {
                            return None;
                        }

                        if self.is_chunk_exhausted(chunk_idx) {
                            tracing::info!("[Thread {t}] chunk {chunk_idx} is exhausted");
                            tracing::info!("[Thread {t}] depth {} chunk {chunk_idx} new 0", depth + 1);
                            return Some((chunk_buffer, 0));
                        }

                        if let Some(hashsets) = create_chunk_hashsets {
                            tracing::info!("[Thread {t}] creating depth {depth} chunk {chunk_idx}");
                            self.create_chunk(&mut chunk_buffer, hashsets, chunk_idx);
                        } else {
                            tracing::info!("[Thread {t}] reading depth {depth} chunk {chunk_idx}");
                            if !self.try_read_chunk(&mut chunk_buffer, depth, chunk_idx) {
                                // No chunk file, so check that it has already been expanded
                                let next_chunk_file = self.chunk_file_path(depth + 1, chunk_idx);
                                assert!(
                                    next_chunk_file.exists(),
                                    "no chunk {chunk_idx} found at depth {depth} or {}",
                                    depth + 1,
                                );

                                // Read the number of new positions from that chunk and return it
                                let new = self.read_new_positions_data_file(depth + 1, chunk_idx);
                                return Some((chunk_buffer, new));
                            }
                        }

                        tracing::info!("[Thread {t}] updating and expanding depth {depth} -> {} chunk {chunk_idx}", depth + 1);
                        let new = self.update_and_expand_chunk(&mut chunk_buffer, depth, chunk_idx);
                        self.write_new_positions_data_file(new, depth + 1, chunk_idx);

                        tracing::info!("[Thread {t}] depth {} chunk {chunk_idx} new {new}", depth + 1);

                        tracing::info!("[Thread {t}] writing depth {} chunk {chunk_idx}", depth + 1);
                        self.write_chunk(&mut chunk_buffer, depth + 1, chunk_idx);

                        // Check if the chunk is exhausted
                        if chunk_buffer.iter().all(|&byte| byte == 0xFF) {
                            tracing::info!("[Thread {t}] marking chunk {chunk_idx} as exhausted");
                            self.mark_chunk_exhausted(chunk_idx);
                        }

                        tracing::info!("[Thread {t}] deleting depth {depth} chunk {chunk_idx}");
                        self.delete_chunk_file(depth, chunk_idx);

                        tracing::info!("[Thread {t}] deleting update files for depth {depth} -> {} chunk {chunk_idx}", depth + 1);
                        self.delete_update_files(depth + 1, chunk_idx);

                        tracing::info!("[Thread {t}] deleting update array for depth {depth} -> {} chunk {chunk_idx}", depth + 1);
                        self.delete_update_array(depth + 1, chunk_idx);

                        Some((chunk_buffer, new))
                    })
                })
                .collect::<Vec<_>>();

            for (t, thread) in threads.into_iter().enumerate() {
                if let Some((mut chunk_buffer, new)) = thread.join().unwrap() {
                    new_positions += new;

                    // Swap the buffer back into place so the next group of threads can
                    // reuse it
                    std::mem::swap(&mut chunk_buffers[t], &mut chunk_buffer);
                }
            }
        });

        new_positions
    }

    fn create_and_process_chunk_group(
        &self,
        chunk_buffers: &mut [Vec<u8>],
        hashsets: &[&HashSet<u64>],
        depth: usize,
        group_idx: usize,
    ) -> u64 {
        self.process_chunk_group(chunk_buffers, Some(hashsets), depth, group_idx)
    }

    fn read_and_process_chunk_group(
        &self,
        chunk_buffers: &mut [Vec<u8>],
        depth: usize,
        group_idx: usize,
    ) -> u64 {
        self.process_chunk_group(chunk_buffers, None, depth, group_idx)
    }

    fn check_for_update_files_compression(&self, update_buffers: &mut [Vec<u8>], depth: usize) {
        std::thread::scope(|s| {
            let threads = (0..self.threads)
                .map(|t| {
                    let mut update_buffer = std::mem::take(&mut update_buffers[t]);

                    s.spawn(move || {
                        for chunk_idx in
                            (0..self.num_array_chunks()).skip(t).step_by(self.threads)
                        {
                            let path = self.update_chunk_dir_path(depth + 2, chunk_idx);
                            if !path.exists() {
                                continue;
                            }

                            let used_space = fs_extra::dir::get_size(&path).unwrap();

                            if used_space > self.update_files_compression_threshold {
                                tracing::info!(
                                    "[Thread {t}] compressing update files for depth {} -> {} chunk {chunk_idx}",
                                    depth + 1,
                                    depth + 2,
                                );
                                self.compress_update_files(
                                    &mut update_buffer,
                                    depth + 2,
                                    chunk_idx,
                                );
                            }
                        }

                        update_buffer
                    })
                })
                .collect::<Vec<_>>();

            for (t, mut update_buffer) in threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .enumerate()
            {
                std::mem::swap(&mut update_buffers[t], &mut update_buffer);
            }
        });
    }

    fn end_of_depth_cleanup(&self, depth: usize) {
        // We now have the array at depth `depth + 1`, and update files/arrays for depth
        // `depth + 2`, so we can delete the directories (which should be empty) for the
        // previous depth.
        for root_idx in 0..self.root_directories.len() {
            let dir_path = self.chunk_dir_path(depth, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }

            let dir_path = self.update_depth_dir_path(depth + 1, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }

            let dir_path = self.update_array_dir_path(depth + 1, root_idx);
            if dir_path.exists() {
                std::fs::remove_dir_all(dir_path).unwrap();
            }
        }

        self.delete_new_positions_data_dir(depth + 1);
    }

    fn first_disk_iteration(
        &self,
        chunk_buffers: &mut [Vec<u8>],
        hashsets: &[&HashSet<u64>],
        depth: usize,
    ) -> u64 {
        let mut new_positions = 0;

        for group_idx in (0..self.num_array_chunks()).step_by(self.threads) {
            new_positions +=
                self.create_and_process_chunk_group(chunk_buffers, hashsets, depth, group_idx);
            self.check_for_update_files_compression(chunk_buffers, depth);
        }

        tracing::info!("depth {} new {new_positions}", depth + 1);

        self.write_state(State::Cleanup { depth });
        self.end_of_depth_cleanup(depth);

        new_positions
    }

    fn do_iteration(&self, chunk_buffers: &mut [Vec<u8>], depth: usize) -> u64 {
        let mut new = 0;

        for group_idx in (0..self.num_array_chunks()).step_by(self.threads) {
            self.write_state(State::Iteration { depth });

            new += self.read_and_process_chunk_group(chunk_buffers, depth, group_idx);
            self.check_for_update_files_compression(chunk_buffers, depth);
        }

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

        let mut chunk_buffers = vec![vec![0u8; self.chunk_size_bytes]; self.threads];

        let new_positions = self.first_disk_iteration(&mut chunk_buffers, &[&old, &current], depth);

        if new_positions == 0 {
            self.write_state(State::Done);
            return;
        }

        drop(old);
        drop(current);

        depth += 1;

        while self.do_iteration(&mut chunk_buffers, depth) != 0 {
            depth += 1;
        }

        self.write_state(State::Done);
    }

    pub fn run(&self) {
        match self.read_state() {
            Some(s) => match s {
                State::Iteration { mut depth } => {
                    let mut chunk_buffers = vec![vec![0u8; self.chunk_size_bytes]; self.threads];

                    if self.do_iteration(&mut chunk_buffers, depth) == 0 {
                        self.write_state(State::Done);
                        return;
                    }

                    depth += 1;

                    while self.do_iteration(&mut chunk_buffers, depth) != 0 {
                        depth += 1;
                    }

                    self.write_state(State::Done);
                }
                State::Cleanup { mut depth } => {
                    self.end_of_depth_cleanup(depth);
                    depth += 1;

                    let mut chunk_buffers = vec![vec![0u8; self.chunk_size_bytes]; self.threads];

                    while self.do_iteration(&mut chunk_buffers, depth) != 0 {
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
