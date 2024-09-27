//! Builder used to define settings for a BFS and run it.

use std::path::PathBuf;

use thiserror::Error;

use crate::{
    bfs::Bfs, callback::BfsCallback, expander::BfsExpander, io::LockedIO,
    provider::BfsSettingsProvider, settings::BfsSettings,
};

/// Error type for [`BfsBuilder`].
#[derive(Debug, Error)]
pub enum BfsBuilderError {
    /// The `threads` parameter was not set.
    #[error("`threads` not set")]
    ThreadsNotSet,

    /// The `chunk_size_bytes` parameter was not set.
    #[error("`chunk_size_bytes` not set")]
    ChunkSizeBytesNotSet,

    /// The `update_memory` parameter was not set.
    #[error("`update_memory` not set")]
    UpdateMemoryNotSet,

    /// The `num_update_blocks` parameter was not set.
    #[error("`num_update_blocks` not set")]
    NumUpdateBlocksNotSet,

    /// The `capacity_check_frequency` parameter was not set.
    #[error("`capacity_check_frequency` not set")]
    CapacityCheckFrequencyNotSet,

    /// The `initial_states` parameter was not set.
    #[error("`initial_states` not set")]
    InitialStatesNotSet,

    /// The `state_size` parameter was not set.
    #[error("`state_size` not set")]
    StateSizeNotSet,

    /// The `root_directories` parameter was not set.
    #[error("`root_directories` not set")]
    RootDirectoriesNotSet,

    /// The `initial_memory_limit` parameter was not set.
    #[error("`initial_memory_limit` not set")]
    InitialMemoryLimitNotSet,

    /// The `available_disk_space_limit` parameter was not set.
    #[error("`available_disk_space_limit` not set")]
    AvailableDiskSpaceLimitNotSet,

    /// The `update_array_threshold` parameter was not set.
    #[error("`update_array_threshold` not set")]
    UpdateArrayThresholdNotSet,

    /// The `use_locked_io` parameter was not set.
    #[error("`use_locked_io` not set")]
    UseLockedIoNotSet,

    /// The `sync_filesystem` parameter was not set.
    #[error("`sync_filesystem` not set")]
    SyncFilesystemNotSet,

    /// The `compute_checksums` parameter was not set.
    #[error("`compute_checksums` not set")]
    ComputeChecksumsNotSet,

    /// The `use_compression` parameter was not set.
    #[error("`use_compression` not set")]
    UseCompressionNotSet,

    /// The `expander` parameter was not set.
    #[error("`expander` not set")]
    ExpanderNotSet,

    /// The `callback` parameter was not set.
    #[error("`callback` not set")]
    CallbackNotSet,

    /// The `settings_provider` parameter was not set.
    #[error("`settings_provider` not set")]
    SettingsProviderNotSet,

    /// The `chunk_size_bytes` parameter was too large.
    #[error("Chunk size ({chunk_size_bytes}) must be at most 2^29 = 536870912 bytes")]
    ChunkSizeTooLarge {
        /// The provided chunk size in bytes.
        chunk_size_bytes: usize,
    },

    /// Not all chunks of the bit array are the same size.
    #[error(
        "State size ({state_size}) must be divisible by 8 * chunk size ({})",
        8 * chunk_size_bytes,
    )]
    ChunksNotSameSize {
        /// The provided state size.
        state_size: u64,

        /// The provided chunk size in bytes.
        chunk_size_bytes: usize,
    },

    /// Not enough update blocks for the given number of threads and chunks.
    #[error(
        "Number of update blocks ({num_update_blocks}) must be greater than threads * num chunks \
        ({})",
        threads * num_chunks,
    )]
    NotEnoughUpdateBlocks {
        /// The provided number of update blocks.
        num_update_blocks: usize,

        /// The provided number of threads.
        threads: usize,

        /// The number of chunks, calculated from the state size and chunk size.
        num_chunks: usize,
    },
}

/// Used to define all the parameters of a BFS and run it.
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
    Expander: BfsExpander<EXPANSION_NODES> + Clone + Sync,
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
    Expander: BfsExpander<EXPANSION_NODES> + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    Provider: BfsSettingsProvider + Sync,
{
    /// Creates a default builder with no parameters set.
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

    /// The number of threads to use in the search.
    #[must_use]
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// The size of each bit array chunk in bytes. Must be at most 2^29.
    #[must_use]
    pub fn chunk_size_bytes(mut self, chunk_size_bytes: usize) -> Self {
        self.chunk_size_bytes = Some(chunk_size_bytes);
        self
    }

    /// The amount of memory to use for storing updates (adjacent nodes) in bytes.
    #[must_use]
    pub fn update_memory(mut self, update_memory: usize) -> Self {
        self.update_memory = Some(update_memory);
        self
    }

    /// The number of blocks in which to divide the update memory.
    #[must_use]
    pub fn num_update_blocks(mut self, num_update_blocks: usize) -> Self {
        self.num_update_blocks = Some(num_update_blocks);
        self
    }

    /// The frequency (one check per `capacity_check_frequency` new nodes) at which to check the
    /// capacity of an update block. When capacity is reached, the block will be stored and a new
    /// block taken.
    #[must_use]
    pub fn capacity_check_frequency(mut self, capacity_check_frequency: usize) -> Self {
        self.capacity_check_frequency = Some(capacity_check_frequency);
        self
    }

    /// The nodes at depth 0, from which the search starts.
    #[must_use]
    pub fn initial_states(mut self, initial_states: &[u64]) -> Self {
        self.initial_states = Some(initial_states.to_vec());
        self
    }

    /// The number of nodes in the graph.
    #[must_use]
    pub fn state_size(mut self, state_size: u64) -> Self {
        self.state_size = Some(state_size);
        self
    }

    /// The working directories that will contain all of the intermediate data. Each directory
    /// should be on a different hard drive.
    #[must_use]
    pub fn root_directories(mut self, root_directories: &[PathBuf]) -> Self {
        self.root_directories = Some(root_directories.to_vec());
        self
    }

    /// Approximate amount of memory in bytes to use for the initial in-memory breadth-first
    /// search, before switching to the disk-based search.
    #[must_use]
    pub fn initial_memory_limit(mut self, initial_memory_limit: usize) -> Self {
        self.initial_memory_limit = Some(initial_memory_limit);
        self
    }

    /// The minimum amount of available disk space (in bytes) that should be kept free on each
    /// drive.
    #[must_use]
    pub fn available_disk_space_limit(mut self, available_disk_space_limit: u64) -> Self {
        self.available_disk_space_limit = Some(available_disk_space_limit);
        self
    }

    /// The threshold at which to switch from writing an update file containing a list of node
    /// indices, to writing a bit array of updates.
    #[must_use]
    pub fn update_array_threshold(mut self, update_array_threshold: u64) -> Self {
        self.update_array_threshold = Some(update_array_threshold);
        self
    }

    /// Whether disk I/O should be locked using a mutex to prevent multiple threads from performing
    /// I/O on the same disk at the same time.
    #[must_use]
    pub fn use_locked_io(mut self, use_locked_io: bool) -> Self {
        self.use_locked_io = Some(use_locked_io);
        self
    }

    /// Whether to call `sync` on the filesystem before deleting files. If this is set to false and
    /// a system failure occurs, it's possible that data could be lost and the search would need to
    /// be restarted from the beginning.
    #[must_use]
    pub fn sync_filesystem(mut self, sync_filesystem: bool) -> Self {
        self.sync_filesystem = Some(sync_filesystem);
        self
    }

    /// Whether to compute and verify checksums for all data written to disk.
    #[must_use]
    pub fn compute_checksums(mut self, compute_checksums: bool) -> Self {
        self.compute_checksums = Some(compute_checksums);
        self
    }

    /// Whether update files and bit arrays should be compressed.
    #[must_use]
    pub fn use_compression(mut self, use_compression: bool) -> Self {
        self.use_compression = Some(use_compression);
        self
    }

    /// The implementor of the [`BfsExpander`] trait that defines the graph to traverse.
    ///
    /// [`BfsExpander`]: ../expander/trait.BfsExpander.html
    #[must_use]
    pub fn expander(mut self, expander: Expander) -> Self {
        self.expander = Some(expander);
        self
    }

    /// The implementor of the [`BfsCallback`] trait that defines callbacks to run during the
    /// search.
    ///
    /// [`BfsCallback`]: ../callback/trait.BfsCallback.html
    #[must_use]
    pub fn callback(mut self, callback: Callback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// The implementor of the [`BfsSettingsProvider`] trait that provides various additional
    /// settings for the search.
    ///
    /// [`BfsSettingsProvider`]: ../provider/trait.BfsSettingsProvider.html
    #[must_use]
    pub fn settings_provider(mut self, settings_provider: Provider) -> Self {
        self.settings_provider = Some(settings_provider);
        self
    }

    /// Runs the BFS with the given settings, requiring all fields of the [`BfsBuilder`] to be set
    /// explicitly, without using any default values.
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

    /// Runs the BFS with the given settings, using default values for any parameters that were not
    /// set, if possible.
    ///
    /// Default values are:
    /// - `threads`: 1
    /// - `update_memory`: 1 GiB
    /// - `capacity_check_frequency`: 256
    /// - `initial_memory_limit`: 64 MiB
    /// - `use_locked_io`: false
    /// - `sync_filesystem`: true
    /// - `compute_checksums`: true
    /// - `use_compression`: true
    /// - `update_array_threshold`: `chunk_size_bytes`
    /// - `num_update_blocks`: `2 * threads * num_chunks`, where
    ///   `num_chunks = state_size / (8 * chunk_size_bytes)`
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
