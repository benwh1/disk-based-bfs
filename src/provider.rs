#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateFilesBehavior {
    DontCompress,
    CompressAndDelete,
    CompressAndKeep,
}

impl UpdateFilesBehavior {
    pub(crate) fn should_compress(self) -> bool {
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
