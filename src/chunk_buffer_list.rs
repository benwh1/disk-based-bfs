use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ChunkBufferList {
    buffers: Arc<Mutex<Vec<Option<Vec<u8>>>>>,
}

impl ChunkBufferList {
    pub fn new_empty(num_buffers: usize) -> Self {
        let buffers = vec![None; num_buffers];
        let buffers = Arc::new(Mutex::new(buffers));
        Self { buffers }
    }

    pub fn fill(&self, buffer_size: usize) {
        let mut lock = self.buffers.lock().unwrap();
        for buf in lock.iter_mut() {
            if buf.is_none() {
                buf.replace(vec![0; buffer_size]);
            }
        }
    }

    pub fn take(&self) -> Option<Vec<u8>> {
        let mut lock = self.buffers.lock().unwrap();
        lock.iter_mut().find_map(|buf| buf.take())
    }

    pub fn put(&self, buffer: Vec<u8>) {
        let mut lock = self.buffers.lock().unwrap();
        for buf in lock.iter_mut() {
            if buf.is_none() {
                buf.replace(buffer);
                return;
            }
        }
    }
}
