//! Defines the `BfsSettingsProvider` trait and related types.

/// Defines the behavior of update files and update bit arrays at the end of each iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateFilesBehavior {
    /// Don't merge update files into a bit array at the end of the iteration.
    DontMerge,

    /// Merge update files into a bit array at the end of the iteration, and delete them after they
    /// have been used.
    MergeAndDelete,

    /// Merge update files into a bit array at the end of the iteration, and move them to a backup
    /// directory after they have been used.
    MergeAndKeep,
}

impl UpdateFilesBehavior {
    pub(crate) fn should_merge(self) -> bool {
        matches!(self, Self::MergeAndDelete | Self::MergeAndKeep)
    }
}

/// Defines the behavior of chunk files at the end of each iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkFilesBehavior {
    /// Delete chunk files at the end of the iteration.
    Delete,

    /// Move chunk files to a backup directory at the end of the iteration.
    Keep,
}

/// Provider for some additional settings for the BFS.
pub trait BfsSettingsProvider {
    /// Returns an index into [`BfsBuilder::root_directories`] that defines which hard drive the
    /// given chunk should be stored on.
    ///
    /// [`BfsBuilder::root_directories`]: ../builder/struct.BfsBuilder.html
    fn chunk_root_idx(&self, chunk_idx: usize) -> usize;

    /// Returns the behavior of update files at the end of the depth `depth` iteration.
    fn update_files_behavior(&self, depth: usize) -> UpdateFilesBehavior;

    /// Returns the behavior of chunk files at the end of the depth `depth` iteration.
    fn chunk_files_behavior(&self, depth: usize) -> ChunkFilesBehavior;
}
