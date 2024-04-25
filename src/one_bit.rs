use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex, RwLock},
};

use cityhasher::{CityHasher, HashSet};
use rand::distributions::{Alphanumeric, DistString};
use serde_derive::{Deserialize, Serialize};

use crate::callback::BfsCallback;

pub struct BfsSettingsBuilder {
    threads: usize,
    chunk_size_bytes: Option<usize>,
    update_set_capacity: Option<usize>,
    capacity_check_frequency: Option<usize>,
    initial_states: Option<Vec<u64>>,
    state_size: Option<u64>,
    root_directories: Option<Vec<PathBuf>>,
    initial_memory_limit: Option<usize>,
    update_files_compression_threshold: Option<u64>,
    buf_io_capacity: Option<usize>,
}

impl Default for BfsSettingsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BfsSettingsBuilder {
    pub fn new() -> Self {
        Self {
            threads: 1,
            chunk_size_bytes: None,
            update_set_capacity: None,
            capacity_check_frequency: None,
            initial_states: None,
            state_size: None,
            root_directories: None,
            initial_memory_limit: None,
            update_files_compression_threshold: None,
            buf_io_capacity: None,
        }
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

    pub fn buf_io_capacity(mut self, buf_io_capacity: usize) -> Self {
        self.buf_io_capacity = Some(buf_io_capacity);
        self
    }

    pub fn build(self) -> Option<BfsSettings> {
        // Require that all chunks are the same size
        let chunk_size_bytes = self.chunk_size_bytes?;
        let state_size = self.state_size? as usize;
        if state_size % (8 * chunk_size_bytes) != 0 {
            return None;
        }

        Some(BfsSettings {
            threads: self.threads,
            chunk_size_bytes: self.chunk_size_bytes?,
            update_set_capacity: self.update_set_capacity?,
            capacity_check_frequency: self.capacity_check_frequency?,
            initial_states: self.initial_states?,
            state_size: self.state_size?,
            root_directories: self.root_directories?,
            initial_memory_limit: self.initial_memory_limit?,
            update_files_compression_threshold: self.update_files_compression_threshold?,
            buf_io_capacity: self.buf_io_capacity?,
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

pub struct BfsSettings {
    threads: usize,
    chunk_size_bytes: usize,
    update_set_capacity: usize,
    capacity_check_frequency: usize,
    initial_states: Vec<u64>,
    state_size: u64,
    root_directories: Vec<PathBuf>,
    initial_memory_limit: usize,
    update_files_compression_threshold: u64,
    buf_io_capacity: usize,
}

impl BfsSettings {
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

    fn update_files_size_file_path(&self) -> PathBuf {
        self.root_dir(0).join("update-files-size.dat")
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
}

struct UpdateFileManager<'a> {
    settings: &'a BfsSettings,
    sizes: RwLock<HashMap<usize, Vec<u64>>>,
    size_file_lock: Mutex<()>,
}

impl<'a> UpdateFileManager<'a> {
    fn new(settings: &'a BfsSettings) -> Self {
        Self {
            settings,
            sizes: RwLock::new(HashMap::new()),
            size_file_lock: Mutex::new(()),
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

        let file_path_tmp = self
            .settings
            .update_files_size_file_path()
            .with_extension("tmp");
        std::fs::write(&file_path_tmp, &str).unwrap();
        let file_path = self.settings.update_files_size_file_path();
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

    fn write_update_file(
        &self,
        update_set: &mut HashSet<u32>,
        depth: usize,
        updated_chunk_idx: usize,
        from_chunk_idx: usize,
    ) {
        let dir_path = self.settings.update_chunk_from_chunk_dir_path(
            depth,
            updated_chunk_idx,
            from_chunk_idx,
        );

        std::fs::create_dir_all(&dir_path).unwrap();

        let mut rng = rand::thread_rng();
        let file_name = Alphanumeric.sample_string(&mut rng, 16);
        let mut file_path = dir_path.join(file_name);
        file_path.set_extension("dat");

        let file_path_tmp = file_path.with_extension("tmp");
        let file = File::create(&file_path_tmp).unwrap();
        let mut writer = BufWriter::with_capacity(self.settings.buf_io_capacity, file);

        let bytes_to_write = update_set.len() as u64 * 4;

        for val in update_set.drain() {
            if writer.write(&val.to_le_bytes()).unwrap() != 4 {
                panic!("failed to write to update file");
            }
        }

        drop(writer);

        let mut lock = self.sizes.write().unwrap();
        let entry = lock
            .entry(depth)
            .or_insert_with(|| vec![0; self.settings.num_array_chunks()]);
        entry[updated_chunk_idx] += bytes_to_write;
        drop(lock);

        std::fs::rename(file_path_tmp, file_path).unwrap();
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

        for from_chunk_idx in 0..self.settings.num_array_chunks() {
            let dir_path =
                self.settings
                    .update_chunk_from_chunk_dir_path(depth, chunk_idx, from_chunk_idx);

            let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
                continue;
            };

            for file_path in read_dir
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("used"))
            {
                bytes_deleted += file_path.metadata().unwrap().len();
                std::fs::remove_file(file_path).unwrap();
            }
        }

        let mut lock = self.sizes.write().unwrap();
        lock.entry(depth)
            .and_modify(|entry| entry[chunk_idx] = entry[chunk_idx].saturating_sub(bytes_deleted));
    }

    fn files_size(&self, depth: usize, chunk_idx: usize) -> u64 {
        let read_lock = self.sizes.read().unwrap();
        read_lock.get(&depth).map_or(0, |sizes| sizes[chunk_idx])
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
    update_file_manager: UpdateFileManager<'a>,
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
        let update_file_manager = UpdateFileManager::new(settings);

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
        std::fs::write(&file_path_tmp, &str).unwrap();
        let file_path = self.settings.state_file_path();
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn try_read_chunk(&self, chunk_buffer: &mut [u8], depth: usize, chunk_idx: usize) -> bool {
        let file_path = self.settings.chunk_file_path(depth, chunk_idx);
        let Ok(mut file) = File::open(file_path) else {
            return false;
        };

        // Check that the file size is correct
        let expected_size = self.settings.chunk_size_bytes;
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
        let file_path = self.settings.update_array_file_path(depth, chunk_idx);
        if !file_path.exists() {
            return false;
        }

        let mut file = File::open(file_path).unwrap();

        // Check that the file size is correct
        let expected_size = self.settings.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        file.read_exact(update_buffer).unwrap();

        true
    }

    fn write_update_array(&self, update_buffer: &[u8], depth: usize, chunk_idx: usize) {
        let dir_path = self.settings.update_array_dir_path(depth, chunk_idx);

        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = dir_path.join(format!("update-chunk-{chunk_idx}.dat.tmp"));
        let mut file = File::create(&file_path_tmp).unwrap();

        file.write_all(update_buffer).unwrap();
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

        file.write_all(chunk_buffer).unwrap();
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

        for from_chunk_idx in 0..self.settings.num_array_chunks() {
            let dir_path =
                self.settings
                    .update_chunk_from_chunk_dir_path(depth, chunk_idx, from_chunk_idx);

            let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
                continue;
            };

            for file_path in read_dir.flatten().map(|entry| entry.path()).filter(|path| {
                let ext = path.extension().and_then(|ext| ext.to_str());
                // Look for "used" as well, in case we restart the program while this loop is
                // running and need to re-read all the update files
                ext == Some("dat") || ext == Some("used")
            }) {
                let file = File::open(&file_path).unwrap();
                let mut reader = BufReader::with_capacity(self.settings.buf_io_capacity, file);

                let mut buf = [0u8; 4];

                while let Ok(bytes_read) = reader.read(&mut buf) {
                    if bytes_read == 0 {
                        break;
                    }

                    let chunk_offset = u32::from_le_bytes(buf);
                    let (byte_idx, bit_idx) = self.chunk_offset_to_bit_coords(chunk_offset);
                    update_buffer[byte_idx] |= 1 << bit_idx;
                }

                drop(reader);

                // Rename the file to mark it as used
                let file_path_used = file_path.with_extension("used");
                std::fs::rename(file_path, file_path_used).unwrap();
            }
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

        let mut update_sets = vec![
            HashSet::<u32, CityHasher>::with_capacity_and_hasher(
                self.settings.update_set_capacity,
                CityHasher::default()
            );
            self.settings.num_array_chunks()
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
        for (idx, set) in update_sets
            .iter_mut()
            .enumerate()
            .filter(|(_, set)| !set.is_empty())
        {
            self.update_file_manager
                .write_update_file(set, depth + 2, idx, chunk_idx);
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
        let max_new_nodes = self.settings.capacity_check_frequency * EXPANSION_NODES;

        for (idx, set) in update_sets.iter_mut().enumerate() {
            if set.len() + max_new_nodes > set.capacity() {
                // Possible to reach capacity on the next block of expansions, so
                // write update file to disk
                self.update_file_manager
                    .write_update_file(set, depth, idx, chunk_idx);
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

        for i in 0..self.settings.num_array_chunks() {
            let dir_path = self
                .settings
                .update_chunk_from_chunk_dir_path(depth + 1, chunk_idx, i);

            let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
                continue;
            };

            for file_path in read_dir.flatten().map(|entry| entry.path()) {
                let file = File::open(file_path).unwrap();
                let expected_entries = file.metadata().unwrap().len() / 4;
                let mut reader = BufReader::with_capacity(self.settings.buf_io_capacity, file);

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

                        if new_positions as usize % self.settings.capacity_check_frequency == 0 {
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

        let file_path = self.settings.update_array_file_path(depth + 1, chunk_idx);
        if !file_path.exists() {
            return 0;
        }

        let file = File::open(file_path).unwrap();
        let file_len = file.metadata().unwrap().len() as usize;
        assert_eq!(file_len, self.settings.chunk_size_bytes);

        let mut reader = BufReader::with_capacity(self.settings.buf_io_capacity, file);

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

                    if new_positions as usize % self.settings.capacity_check_frequency == 0 {
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
                let dir_path =
                    self.settings
                        .update_chunk_from_chunk_dir_path(depth + 1, chunk_idx, 0);
                std::fs::create_dir_all(&dir_path).unwrap();
                let file_path = dir_path.join("update.dat");
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
