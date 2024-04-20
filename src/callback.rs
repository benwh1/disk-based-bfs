pub trait BfsCallback {
    fn new_state(&mut self, depth: usize, state: u64);
    fn end_of_chunk(&self, depth: usize, chunk_idx: usize);
}
