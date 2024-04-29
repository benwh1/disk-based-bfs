use std::path::{Path, PathBuf};

pub struct BfsSettingsBuilder {
    threads: usize,
    chunk_size_bytes: Option<usize>,
    update_memory: Option<usize>,
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
            update_memory: None,
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

    pub fn update_memory(mut self, update_memory: usize) -> Self {
        self.update_memory = Some(update_memory);
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
            update_memory: self.update_memory?,
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

pub struct BfsSettings {
    pub(crate) threads: usize,
    pub(crate) chunk_size_bytes: usize,
    pub(crate) update_memory: usize,
    pub(crate) capacity_check_frequency: usize,
    pub(crate) initial_states: Vec<u64>,
    pub(crate) state_size: u64,
    pub(crate) root_directories: Vec<PathBuf>,
    pub(crate) initial_memory_limit: usize,
    pub(crate) update_files_compression_threshold: u64,
    pub(crate) buf_io_capacity: usize,
}

impl BfsSettings {
    pub fn array_bytes(&self) -> usize {
        self.state_size.div_ceil(8) as usize
    }

    pub fn num_array_chunks(&self) -> usize {
        self.array_bytes() / self.chunk_size_bytes
    }

    pub fn states_per_chunk(&self) -> usize {
        self.chunk_size_bytes * 8
    }

    /// `self.update_memory` is the total amount of space we use for storing all updates across all
    /// chunks and all threads, so this is the total size of each small `Vec` of updates.
    pub fn update_capacity_per_vec(&self) -> usize {
        // We have one vec per chunk per thread in `update_and_expand_chunk`, and another vec per
        // chunk per thread in `UpdateManager`, and each entry is a `u32`.
        self.update_memory
            / (self.threads * self.num_array_chunks() * 2 * std::mem::size_of::<u32>())
    }

    pub fn root_dir(&self, chunk_idx: usize) -> &Path {
        &self.root_directories[chunk_idx % self.root_directories.len()]
    }

    pub fn state_file_path(&self) -> PathBuf {
        self.root_dir(0).join("state.dat")
    }

    pub fn update_files_size_file_path(&self) -> PathBuf {
        self.root_dir(0).join("update-files-size.dat")
    }

    pub fn update_depth_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("update")
            .join(format!("depth-{depth}"))
    }

    pub fn update_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.update_depth_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}"))
    }

    pub fn chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("array")
            .join(format!("depth-{depth}"))
    }

    pub fn chunk_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.chunk_dir_path(depth, chunk_idx)
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    pub fn update_array_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("update-array")
            .join(format!("depth-{depth}"))
    }

    pub fn update_array_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.update_array_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}.dat"))
    }

    pub fn new_positions_data_dir_path(&self, depth: usize) -> PathBuf {
        self.root_dir(0)
            .join("new-positions")
            .join(format!("depth-{depth}"))
    }

    pub fn new_positions_data_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.new_positions_data_dir_path(depth)
            .join(format!("chunk-{chunk_idx}.dat"))
    }
}
