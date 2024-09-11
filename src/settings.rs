use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateFilesBehavior {
    DontCompress,
    CompressAndDelete,
    CompressAndKeep,
}

impl UpdateFilesBehavior {
    pub fn should_compress(self) -> bool {
        matches!(self, Self::CompressAndDelete | Self::CompressAndKeep)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkFilesBehavior {
    Delete,
    Keep,
}

pub trait BfsSettingsProvider {
    fn chunk_root_idx(&self, chunk_idx: usize) -> usize;
    fn update_files_behavior(&self, depth: usize) -> UpdateFilesBehavior;
    fn chunk_files_behavior(&self, depth: usize) -> ChunkFilesBehavior;
}

pub struct BfsSettingsBuilder<P: BfsSettingsProvider> {
    threads: usize,
    chunk_size_bytes: Option<usize>,
    update_memory: Option<usize>,
    update_vec_capacity: Option<usize>,
    capacity_check_frequency: Option<usize>,
    initial_states: Option<Vec<u64>>,
    state_size: Option<u64>,
    root_directories: Option<Vec<PathBuf>>,
    initial_memory_limit: Option<usize>,
    update_files_compression_threshold: Option<u64>,
    buf_io_capacity: Option<usize>,
    use_locked_io: Option<bool>,
    sync_filesystem: Option<bool>,
    compute_checksums: Option<bool>,
    settings_provider: Option<P>,
}

impl<P: BfsSettingsProvider> Default for BfsSettingsBuilder<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: BfsSettingsProvider> BfsSettingsBuilder<P> {
    pub fn new() -> Self {
        Self {
            threads: 1,
            chunk_size_bytes: None,
            update_memory: None,
            update_vec_capacity: None,
            capacity_check_frequency: None,
            initial_states: None,
            state_size: None,
            root_directories: None,
            initial_memory_limit: None,
            update_files_compression_threshold: None,
            buf_io_capacity: None,
            use_locked_io: None,
            sync_filesystem: None,
            compute_checksums: None,
            settings_provider: None,
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

    pub fn update_vec_capacity(mut self, update_vec_capacity: usize) -> Self {
        self.update_vec_capacity = Some(update_vec_capacity);
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

    pub fn use_locked_io(mut self, use_locked_io: bool) -> Self {
        self.use_locked_io = Some(use_locked_io);
        self
    }

    pub fn sync_filesystem(mut self, sync_filesystem: bool) -> Self {
        self.sync_filesystem = Some(sync_filesystem);
        self
    }

    pub fn compute_checksums(mut self, compute_checksums: bool) -> Self {
        self.compute_checksums = Some(compute_checksums);
        self
    }

    pub fn settings_provider(mut self, settings_provider: P) -> Self {
        self.settings_provider = Some(settings_provider);
        self
    }

    pub fn build(self) -> Option<BfsSettings<P>> {
        // Require that all chunks are the same size
        let chunk_size_bytes = self.chunk_size_bytes?;
        let state_size = self.state_size? as usize;
        if state_size % (8 * chunk_size_bytes) != 0 {
            return None;
        }

        let update_files_compression_threshold = self.update_files_compression_threshold?;
        if chunk_size_bytes as u64 >= update_files_compression_threshold {
            // If this is the case then we would get stuck in an infinite loop when compressing
            // update files, because the total file size after compressing would still be greater
            // than the threshold.
            return None;
        }

        // Each thread can hold one update vec per chunk, so we need more than (threads * chunks)
        // update vecs in total
        let num_update_vecs =
            self.update_memory? / (self.update_vec_capacity? * std::mem::size_of::<u32>());
        let num_chunks = state_size / (8 * chunk_size_bytes);
        if num_update_vecs <= self.threads * num_chunks {
            return None;
        }

        Some(BfsSettings {
            threads: self.threads,
            chunk_size_bytes: self.chunk_size_bytes?,
            update_memory: self.update_memory?,
            update_vec_capacity: self.update_vec_capacity?,
            capacity_check_frequency: self.capacity_check_frequency?,
            initial_states: self.initial_states?,
            state_size: self.state_size?,
            root_directories: self.root_directories?,
            initial_memory_limit: self.initial_memory_limit?,
            update_files_compression_threshold: self.update_files_compression_threshold?,
            buf_io_capacity: self.buf_io_capacity?,
            use_locked_io: self.use_locked_io?,
            sync_filesystem: self.sync_filesystem?,
            compute_checksums: self.compute_checksums?,
            settings_provider: self.settings_provider?,
        })
    }
}

pub struct BfsSettings<P: BfsSettingsProvider> {
    pub(crate) threads: usize,
    pub(crate) chunk_size_bytes: usize,
    pub(crate) update_memory: usize,
    pub(crate) update_vec_capacity: usize,
    pub(crate) capacity_check_frequency: usize,
    pub(crate) initial_states: Vec<u64>,
    pub(crate) state_size: u64,
    pub(crate) root_directories: Vec<PathBuf>,
    pub(crate) initial_memory_limit: usize,
    pub(crate) update_files_compression_threshold: u64,
    pub(crate) buf_io_capacity: usize,
    pub(crate) use_locked_io: bool,
    pub(crate) sync_filesystem: bool,
    pub(crate) compute_checksums: bool,
    pub(crate) settings_provider: P,
}

impl<P: BfsSettingsProvider> BfsSettings<P> {
    pub fn array_bytes(&self) -> usize {
        self.state_size.div_ceil(8) as usize
    }

    pub fn num_array_chunks(&self) -> usize {
        self.array_bytes() / self.chunk_size_bytes
    }

    pub fn states_per_chunk(&self) -> usize {
        self.chunk_size_bytes * 8
    }

    pub fn num_update_blocks(&self) -> usize {
        self.update_memory / (self.update_vec_capacity * std::mem::size_of::<u32>())
    }

    pub fn root_dir(&self, chunk_idx: usize) -> &Path {
        let chunk_root_idx = self.settings_provider.chunk_root_idx(chunk_idx);
        &self.root_directories[chunk_root_idx]
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

    pub fn backup_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("backup-array")
            .join(format!("depth-{depth}"))
    }

    pub fn backup_chunk_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.backup_chunk_dir_path(depth, chunk_idx)
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    pub fn update_array_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("update-array")
            .join(format!("depth-{depth}"))
    }

    pub fn update_array_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.update_array_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}"))
    }

    pub fn backup_update_array_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("backup-update-array")
            .join(format!("depth-{depth}"))
    }

    pub fn backup_update_array_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.backup_update_array_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}"))
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

    pub fn exhausted_chunk_dir_path(&self) -> PathBuf {
        self.root_dir(0).join("exhausted-chunks")
    }

    pub fn exhausted_chunk_file_path(&self, chunk_idx: usize) -> PathBuf {
        self.exhausted_chunk_dir_path()
            .join(format!("chunk-{chunk_idx}.dat"))
    }
}
