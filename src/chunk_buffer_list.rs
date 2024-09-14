use std::sync::Arc;

use parking_lot::Mutex;

#[derive(Clone)]
pub(crate) struct ChunkBufferList {
    buffers: Arc<Mutex<Vec<Option<Vec<u8>>>>>,
}

impl ChunkBufferList {
    pub(crate) fn new_empty(num_buffers: usize) -> Self {
        let buffers = vec![None; num_buffers];
        let buffers = Arc::new(Mutex::new(buffers));
        Self { buffers }
    }

    pub(crate) fn fill(&self, buffer_size: usize) {
        let mut lock = self.buffers.lock();
        for buf in lock.iter_mut() {
            if buf.is_none() {
                buf.replace(vec![0; buffer_size]);
            }
        }
    }

    pub(crate) fn take(&self) -> Option<Vec<u8>> {
        let mut lock = self.buffers.lock();
        lock.iter_mut().find_map(|buf| buf.take())
    }

    pub(crate) fn put(&self, buffer: Vec<u8>) {
        let mut lock = self.buffers.lock();
        for buf in lock.iter_mut() {
            if buf.is_none() {
                buf.replace(buffer);
                return;
            }
        }
    }
}
