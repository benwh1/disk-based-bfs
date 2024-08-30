pub trait ChunkAllocator {
    fn chunk_root_idx(&self, chunk_idx: usize) -> usize;
}

pub struct Uniform(pub usize);

impl ChunkAllocator for Uniform {
    fn chunk_root_idx(&self, chunk_idx: usize) -> usize {
        chunk_idx % self.0
    }
}
