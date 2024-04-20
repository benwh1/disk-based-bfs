pub trait BfsCallback {
    fn new_state(&mut self, state: u64);
    fn end_of_chunk(&self, depth: usize, chunk_idx: usize);
}
