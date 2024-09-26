use std::path::{Path, PathBuf};

use crate::{callback::BfsCallback, expander::BfsExpander, provider::BfsSettingsProvider};

#[derive(Debug)]
pub(crate) struct BfsSettings<Expander, Callback, Provider, const EXPANSION_NODES: usize> {
    pub(crate) threads: usize,
    pub(crate) chunk_size_bytes: usize,
    pub(crate) update_memory: usize,
    pub(crate) num_update_blocks: usize,
    pub(crate) capacity_check_frequency: usize,
    pub(crate) initial_states: Vec<u64>,
    pub(crate) state_size: u64,
    pub(crate) root_directories: Vec<PathBuf>,
    pub(crate) initial_memory_limit: usize,
    pub(crate) available_disk_space_limit: u64,
    pub(crate) update_array_threshold: u64,
    pub(crate) use_locked_io: bool,
    pub(crate) sync_filesystem: bool,
    pub(crate) compute_checksums: bool,
    pub(crate) use_compression: bool,
    pub(crate) expander: Expander,
    pub(crate) callback: Callback,
    pub(crate) settings_provider: Provider,
}

impl<Expander, Callback, Provider, const EXPANSION_NODES: usize>
    BfsSettings<Expander, Callback, Provider, EXPANSION_NODES>
where
    Expander: BfsExpander<EXPANSION_NODES> + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    Provider: BfsSettingsProvider + Sync,
{
    fn array_bytes(&self) -> usize {
        self.state_size as usize / 8
    }

    pub(crate) fn num_array_chunks(&self) -> usize {
        self.array_bytes() / self.chunk_size_bytes
    }

    pub(crate) fn states_per_chunk(&self) -> usize {
        self.chunk_size_bytes * 8
    }

    pub(crate) fn update_block_capacity(&self) -> usize {
        self.update_memory / (self.num_update_blocks * std::mem::size_of::<u32>())
    }

    fn root_dir(&self, chunk_idx: usize) -> &Path {
        let chunk_root_idx = self.settings_provider.chunk_root_idx(chunk_idx);
        &self.root_directories[chunk_root_idx]
    }

    pub(crate) fn state_file_path(&self) -> PathBuf {
        self.root_dir(0).join("state.dat")
    }

    pub(crate) fn update_files_size_file_path(&self) -> PathBuf {
        self.root_dir(0).join("update-files-size.dat")
    }

    pub(crate) fn update_depth_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("update")
            .join(format!("depth-{depth}"))
    }

    pub(crate) fn update_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.update_depth_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}"))
    }

    pub(crate) fn chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("array")
            .join(format!("depth-{depth}"))
    }

    pub(crate) fn chunk_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.chunk_dir_path(depth, chunk_idx)
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    pub(crate) fn backup_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("backup-array")
            .join(format!("depth-{depth}"))
    }

    pub(crate) fn backup_chunk_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.backup_chunk_dir_path(depth, chunk_idx)
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    pub(crate) fn update_array_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("update-array")
            .join(format!("depth-{depth}"))
    }

    pub(crate) fn update_array_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.update_array_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}"))
    }

    fn backup_update_array_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("backup-update-array")
            .join(format!("depth-{depth}"))
    }

    pub(crate) fn backup_update_array_chunk_dir_path(
        &self,
        depth: usize,
        chunk_idx: usize,
    ) -> PathBuf {
        self.backup_update_array_dir_path(depth, chunk_idx)
            .join(format!("update-chunk-{chunk_idx}"))
    }

    pub(crate) fn new_positions_data_dir_path(&self, depth: usize) -> PathBuf {
        self.root_dir(0)
            .join("new-positions")
            .join(format!("depth-{depth}"))
    }

    pub(crate) fn new_positions_data_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.new_positions_data_dir_path(depth)
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    pub(crate) fn exhausted_chunk_dir_path(&self) -> PathBuf {
        self.root_dir(0).join("exhausted-chunks")
    }

    pub(crate) fn exhausted_chunk_file_path(&self, chunk_idx: usize) -> PathBuf {
        self.exhausted_chunk_dir_path()
            .join(format!("chunk-{chunk_idx}.dat"))
    }
}
