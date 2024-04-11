use std::{marker::PhantomData, path::PathBuf};

pub struct TwoBitBfsBuilder<T, Encoder: Fn(&T) -> u64, Decoder: Fn(&mut T, u64)> {
    encoder: Option<Encoder>,
    decoder: Option<Decoder>,
    threads: u64,
    chunk_size_bytes: Option<usize>,
    initial_state: Option<T>,
    state_size: Option<u64>,
    array_file_directory: Option<PathBuf>,
    update_file_directory: Option<PathBuf>,
    phantom_t: PhantomData<T>,
}

impl<T, Encoder: Fn(&T) -> u64, Decoder: Fn(&mut T, u64)> TwoBitBfsBuilder<T, Encoder, Decoder> {
    pub fn new() -> Self {
        TwoBitBfsBuilder {
            encoder: None,
            decoder: None,
            threads: 1,
            chunk_size_bytes: None,
            initial_state: None,
            state_size: None,
            array_file_directory: None,
            update_file_directory: None,
            phantom_t: PhantomData,
        }
    }

    pub fn encoder(mut self, encoder: Encoder) -> Self {
        self.encoder = Some(encoder);
        self
    }

    pub fn decoder(mut self, decoder: Decoder) -> Self {
        self.decoder = Some(decoder);
        self
    }

    pub fn threads(mut self, threads: u64) -> Self {
        self.threads = threads;
        self
    }

    pub fn chunk_size_bytes(mut self, chunk_size_bytes: usize) -> Self {
        self.chunk_size_bytes = Some(chunk_size_bytes);
        self
    }

    pub fn initial_state(mut self, initial_state: T) -> Self {
        self.initial_state = Some(initial_state);
        self
    }

    pub fn state_size(mut self, state_size: u64) -> Self {
        self.state_size = Some(state_size);
        self
    }

    pub fn array_file_directory(mut self, array_file_directory: PathBuf) -> Self {
        self.array_file_directory = Some(array_file_directory);
        self
    }

    pub fn update_file_directory(mut self, update_file_directory: PathBuf) -> Self {
        self.update_file_directory = Some(update_file_directory);
        self
    }

    pub fn build(self) -> Option<TwoBitBfs<T, Encoder, Decoder>> {
        Some(TwoBitBfs {
            encoder: self.encoder?,
            decoder: self.decoder?,
            threads: self.threads,
            chunk_size_bytes: self.chunk_size_bytes?,
            initial_state: self.initial_state?,
            state_size: self.state_size?,
            array_file_directory: self.array_file_directory?,
            update_file_directory: self.update_file_directory?,
            phantom_t: PhantomData,
        })
    }
}

pub struct TwoBitBfs<T, Encoder: Fn(&T) -> u64, Decoder: Fn(&mut T, u64)> {
    encoder: Encoder,
    decoder: Decoder,
    threads: u64,
    chunk_size_bytes: usize,
    initial_state: T,
    state_size: u64,
    array_file_directory: PathBuf,
    update_file_directory: PathBuf,
    phantom_t: PhantomData<T>,
}
