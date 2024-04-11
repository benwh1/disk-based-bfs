use std::{marker::PhantomData, path::PathBuf};

pub struct TwoBitBfsBuilder<T, Encoder: Fn(&T) -> usize, Decoder: FnMut(&mut T, usize)> {
    encoder: Option<Encoder>,
    decoder: Option<Decoder>,
    threads: usize,
    chunk_size_bytes: Option<usize>,
    initial_state: Option<usize>,
    state_size: Option<usize>,
    array_file_directories: Option<Vec<PathBuf>>,
    update_file_directories: Option<Vec<PathBuf>>,
    phantom_t: PhantomData<T>,
}

impl<T, Encoder: Fn(&T) -> usize, Decoder: FnMut(&mut T, usize)>
    TwoBitBfsBuilder<T, Encoder, Decoder>
{
    pub fn new() -> Self {
        TwoBitBfsBuilder {
            encoder: None,
            decoder: None,
            threads: 1,
            chunk_size_bytes: None,
            initial_state: None,
            state_size: None,
            array_file_directories: None,
            update_file_directories: None,
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

    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn chunk_size_bytes(mut self, chunk_size_bytes: usize) -> Self {
        self.chunk_size_bytes = Some(chunk_size_bytes);
        self
    }

    pub fn initial_state(mut self, initial_state: &T) -> Self {
        self.initial_state = Some(self
            .encoder
            .as_ref()
            .expect("Encoder is required to set initial state")(
            initial_state
        ));
        self
    }

    pub fn state_size(mut self, state_size: usize) -> Self {
        self.state_size = Some(state_size);
        self
    }

    pub fn array_file_directories(mut self, array_file_directories: Vec<PathBuf>) -> Self {
        self.array_file_directories = Some(array_file_directories);
        self
    }

    pub fn update_file_directories(mut self, update_file_directories: Vec<PathBuf>) -> Self {
        self.update_file_directories = Some(update_file_directories);
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
            array_file_directories: self.array_file_directories?,
            update_file_directories: self.update_file_directories?,
            phantom_t: PhantomData,
        })
    }
}

pub struct TwoBitBfs<T, Encoder: Fn(&T) -> usize, Decoder: FnMut(&mut T, usize)> {
    encoder: Encoder,
    decoder: Decoder,
    threads: usize,
    chunk_size_bytes: usize,
    initial_state: usize,
    state_size: usize,
    array_file_directories: Vec<PathBuf>,
    update_file_directories: Vec<PathBuf>,
    phantom_t: PhantomData<T>,
}
