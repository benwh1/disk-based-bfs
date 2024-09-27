//! Defines the `BfsCallback` trait.

/// Defines callback functions that run during the BFS.
pub trait BfsCallback {
    /// Called when a new node is visited.
    fn new_state(&mut self, depth: usize, state: u64);

    /// Called when a chunk of the bit array has finished processing.
    fn end_of_chunk(&self, depth: usize, chunk_idx: usize);
}
