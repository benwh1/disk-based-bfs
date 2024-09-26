use std::path::PathBuf;

use thiserror::Error;

use crate::{
    bfs::Bfs, callback::BfsCallback, io::LockedIO, provider::BfsSettingsProvider,
    settings::BfsSettings,
};

#[derive(Debug, Error)]
pub enum BfsBuilderError {
    #[error("`threads` not set")]
    ThreadsNotSet,

    #[error("`chunk_size_bytes` not set")]
    ChunkSizeBytesNotSet,

    #[error("`update_memory` not set")]
    UpdateMemoryNotSet,

    #[error("`num_update_blocks` not set")]
    NumUpdateBlocksNotSet,

    #[error("`capacity_check_frequency` not set")]
    CapacityCheckFrequencyNotSet,

    #[error("`initial_states` not set")]
    InitialStatesNotSet,

    #[error("`state_size` not set")]
    StateSizeNotSet,

    #[error("`root_directories` not set")]
    RootDirectoriesNotSet,

    #[error("`initial_memory_limit` not set")]
    InitialMemoryLimitNotSet,

    #[error("`available_disk_space_limit` not set")]
    AvailableDiskSpaceLimitNotSet,

    #[error("`update_array_threshold` not set")]
    UpdateArrayThresholdNotSet,

    #[error("`use_locked_io` not set")]
    UseLockedIoNotSet,

    #[error("`sync_filesystem` not set")]
    SyncFilesystemNotSet,

    #[error("`compute_checksums` not set")]
    ComputeChecksumsNotSet,

    #[error("`use_compression` not set")]
    UseCompressionNotSet,

    #[error("`expander` not set")]
    ExpanderNotSet,

    #[error("`callback` not set")]
    CallbackNotSet,

    #[error("`settings_provider` not set")]
    SettingsProviderNotSet,

    #[error("Chunk size ({chunk_size_bytes}) must be less than 2^29 = 536870912 bytes")]
    ChunkSizeTooLarge { chunk_size_bytes: usize },

    #[error(
        "State size ({state_size}) must be divisible by 8 * chunk size ({})",
        8 * chunk_size_bytes,
    )]
    ChunksNotSameSize {
        state_size: u64,
        chunk_size_bytes: usize,
    },

    #[error(
        "Number of update blocks ({num_update_blocks}) must be greater than threads * num chunks \
        ({})",
        threads * num_chunks,
    )]
    NotEnoughUpdateBlocks {
        num_update_blocks: usize,
        threads: usize,
        num_chunks: usize,
    },
}

#[derive(Debug)]
pub struct BfsBuilder<Expander, Callback, Provider, const EXPANSION_NODES: usize> {
    threads: Option<usize>,
    chunk_size_bytes: Option<usize>,
    update_memory: Option<usize>,
    num_update_blocks: Option<usize>,
    capacity_check_frequency: Option<usize>,
    initial_states: Option<Vec<u64>>,
    state_size: Option<u64>,
    root_directories: Option<Vec<PathBuf>>,
    initial_memory_limit: Option<usize>,
    available_disk_space_limit: Option<u64>,
    update_array_threshold: Option<u64>,
    use_locked_io: Option<bool>,
    sync_filesystem: Option<bool>,
    compute_checksums: Option<bool>,
    use_compression: Option<bool>,
    expander: Option<Expander>,
    callback: Option<Callback>,
    settings_provider: Option<Provider>,
}

impl<Expander, Callback, Provider, const EXPANSION_NODES: usize> Default
    for BfsBuilder<Expander, Callback, Provider, EXPANSION_NODES>
where
    Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    Provider: BfsSettingsProvider + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Expander, Callback, Provider, const EXPANSION_NODES: usize>
    BfsBuilder<Expander, Callback, Provider, EXPANSION_NODES>
where
    Expander: FnMut(u64, &mut [u64; EXPANSION_NODES]) + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    Provider: BfsSettingsProvider + Sync,
{
    #[must_use]
    pub fn new() -> Self {
        Self {
            threads: None,
            chunk_size_bytes: None,
            update_memory: None,
            num_update_blocks: None,
            capacity_check_frequency: None,
            initial_states: None,
            state_size: None,
            root_directories: None,
            initial_memory_limit: None,
            available_disk_space_limit: None,
            update_array_threshold: None,
            use_locked_io: None,
            sync_filesystem: None,
            compute_checksums: None,
            use_compression: None,
            expander: None,
            callback: None,
            settings_provider: None,
        }
    }

    #[must_use]
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    #[must_use]
    pub fn chunk_size_bytes(mut self, chunk_size_bytes: usize) -> Self {
        self.chunk_size_bytes = Some(chunk_size_bytes);
        self
    }

    #[must_use]
    pub fn update_memory(mut self, update_memory: usize) -> Self {
        self.update_memory = Some(update_memory);
        self
    }

    #[must_use]
    pub fn num_update_blocks(mut self, num_update_blocks: usize) -> Self {
        self.num_update_blocks = Some(num_update_blocks);
        self
    }

    #[must_use]
    pub fn capacity_check_frequency(mut self, capacity_check_frequency: usize) -> Self {
        self.capacity_check_frequency = Some(capacity_check_frequency);
        self
    }

    #[must_use]
    pub fn initial_states(mut self, initial_states: &[u64]) -> Self {
        self.initial_states = Some(initial_states.to_vec());
        self
    }

    #[must_use]
    pub fn state_size(mut self, state_size: u64) -> Self {
        self.state_size = Some(state_size);
        self
    }

    #[must_use]
    pub fn root_directories(mut self, root_directories: &[PathBuf]) -> Self {
        self.root_directories = Some(root_directories.to_vec());
        self
    }

    #[must_use]
    pub fn initial_memory_limit(mut self, initial_memory_limit: usize) -> Self {
        self.initial_memory_limit = Some(initial_memory_limit);
        self
    }

    #[must_use]
    pub fn available_disk_space_limit(mut self, available_disk_space_limit: u64) -> Self {
        self.available_disk_space_limit = Some(available_disk_space_limit);
        self
    }

    #[must_use]
    pub fn update_array_threshold(mut self, update_array_threshold: u64) -> Self {
        self.update_array_threshold = Some(update_array_threshold);
        self
    }

    #[must_use]
    pub fn use_locked_io(mut self, use_locked_io: bool) -> Self {
        self.use_locked_io = Some(use_locked_io);
        self
    }

    #[must_use]
    pub fn sync_filesystem(mut self, sync_filesystem: bool) -> Self {
        self.sync_filesystem = Some(sync_filesystem);
        self
    }

    #[must_use]
    pub fn compute_checksums(mut self, compute_checksums: bool) -> Self {
        self.compute_checksums = Some(compute_checksums);
        self
    }

    #[must_use]
    pub fn use_compression(mut self, use_compression: bool) -> Self {
        self.use_compression = Some(use_compression);
        self
    }

    #[must_use]
    pub fn expander(mut self, expander: Expander) -> Self {
        self.expander = Some(expander);
        self
    }

    #[must_use]
    pub fn callback(mut self, callback: Callback) -> Self {
        self.callback = Some(callback);
        self
    }

    #[must_use]
    pub fn settings_provider(mut self, settings_provider: Provider) -> Self {
        self.settings_provider = Some(settings_provider);
        self
    }

    pub fn run_no_defaults(self) -> Result<(), BfsBuilderError> {
        // Limit to 2^29 bytes so that we can store 32 bit values in the update files
        let chunk_size_bytes = self
            .chunk_size_bytes
            .ok_or(BfsBuilderError::ChunkSizeBytesNotSet)?;
        if chunk_size_bytes > 1 << 29 {
            return Err(BfsBuilderError::ChunkSizeTooLarge { chunk_size_bytes });
        }

        // Require that all chunks are the same size
        let state_size = self.state_size.ok_or(BfsBuilderError::StateSizeNotSet)?;
        if state_size as usize % (8 * chunk_size_bytes) != 0 {
            return Err(BfsBuilderError::ChunksNotSameSize {
                state_size,
                chunk_size_bytes,
            });
        }

        // Each thread can hold one update vec per chunk, so we need more than (threads * chunks)
        // update vecs in total
        let num_update_blocks = self
            .num_update_blocks
            .ok_or(BfsBuilderError::NumUpdateBlocksNotSet)?;
        let threads = self.threads.ok_or(BfsBuilderError::ThreadsNotSet)?;
        let num_chunks = state_size as usize / (8 * chunk_size_bytes);
        if num_update_blocks <= threads * num_chunks {
            return Err(BfsBuilderError::NotEnoughUpdateBlocks {
                num_update_blocks,
                threads,
                num_chunks,
            });
        }

        let settings = BfsSettings {
            threads,
            chunk_size_bytes,
            update_memory: self
                .update_memory
                .ok_or(BfsBuilderError::UpdateMemoryNotSet)?,
            num_update_blocks,
            capacity_check_frequency: self
                .capacity_check_frequency
                .ok_or(BfsBuilderError::CapacityCheckFrequencyNotSet)?,
            initial_states: self
                .initial_states
                .ok_or(BfsBuilderError::InitialStatesNotSet)?,
            state_size,
            root_directories: self
                .root_directories
                .ok_or(BfsBuilderError::RootDirectoriesNotSet)?,
            initial_memory_limit: self
                .initial_memory_limit
                .ok_or(BfsBuilderError::InitialMemoryLimitNotSet)?,
            available_disk_space_limit: self
                .available_disk_space_limit
                .ok_or(BfsBuilderError::AvailableDiskSpaceLimitNotSet)?,
            update_array_threshold: self
                .update_array_threshold
                .ok_or(BfsBuilderError::UpdateArrayThresholdNotSet)?,
            use_locked_io: self
                .use_locked_io
                .ok_or(BfsBuilderError::UseLockedIoNotSet)?,
            sync_filesystem: self
                .sync_filesystem
                .ok_or(BfsBuilderError::SyncFilesystemNotSet)?,
            compute_checksums: self
                .compute_checksums
                .ok_or(BfsBuilderError::ComputeChecksumsNotSet)?,
            use_compression: self
                .use_compression
                .ok_or(BfsBuilderError::UseCompressionNotSet)?,
            expander: self.expander.ok_or(BfsBuilderError::ExpanderNotSet)?,
            callback: self.callback.ok_or(BfsBuilderError::CallbackNotSet)?,
            settings_provider: self
                .settings_provider
                .ok_or(BfsBuilderError::SettingsProviderNotSet)?,
        };

        let locked_io = LockedIO::new(&settings);

        let bfs = Bfs::new(&settings, &locked_io);
        bfs.run();

        Ok(())
    }

    pub fn run(mut self) -> Result<(), BfsBuilderError> {
        self.threads.get_or_insert(1);
        self.update_memory.get_or_insert(1 << 30);
        self.capacity_check_frequency.get_or_insert(1 << 8);
        self.initial_memory_limit.get_or_insert(1 << 26);
        self.use_locked_io.get_or_insert(false);
        self.sync_filesystem.get_or_insert(true);
        self.compute_checksums.get_or_insert(true);
        self.use_compression.get_or_insert(true);

        let chunk_size_bytes = self
            .chunk_size_bytes
            .ok_or(BfsBuilderError::ChunkSizeBytesNotSet)?;
        self.update_array_threshold
            .get_or_insert(chunk_size_bytes as u64);

        let state_size = self.state_size.ok_or(BfsBuilderError::StateSizeNotSet)?;
        let num_chunks = state_size as usize / (8 * chunk_size_bytes);
        self.num_update_blocks
            .get_or_insert(2 * self.threads.unwrap() * num_chunks);

        self.run_no_defaults()
    }
}
