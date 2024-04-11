use std::{marker::PhantomData, path::PathBuf};

pub struct TwoBitBfsBuilder<
    T,
    Encoder: Fn(&T) -> u64,
    Decoder: Fn(&mut T, u64),
    Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]),
    const EXPANSION_NODES: usize,
> {
    encoder: Option<Encoder>,
    decoder: Option<Decoder>,
    expander: Option<Expander>,
    threads: u64,
    chunk_size_bytes: Option<usize>,
    initial_state: Option<T>,
    state_size: Option<u64>,
    array_file_directory: Option<PathBuf>,
    update_file_directory: Option<PathBuf>,
    initial_memory_limit: Option<usize>,
    phantom_t: PhantomData<T>,
}

impl<
        T,
        Encoder: Fn(&T) -> u64,
        Decoder: Fn(&mut T, u64),
        Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]),
        const EXPANSION_NODES: usize,
    > TwoBitBfsBuilder<T, Encoder, Decoder, Expander, EXPANSION_NODES>
{
    pub fn new() -> Self {
        TwoBitBfsBuilder {
            encoder: None,
            decoder: None,
            expander: None,
            threads: 1,
            chunk_size_bytes: None,
            initial_state: None,
            state_size: None,
            array_file_directory: None,
            update_file_directory: None,
            initial_memory_limit: None,
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

    pub fn expander(mut self, expander: Expander) -> Self {
        self.expander = Some(expander);
        self
    }

    pub fn expansion_nodes(self, _: [(); EXPANSION_NODES]) -> Self {
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

    pub fn initial_memory_limit(mut self, initial_memory_limit: usize) -> Self {
        self.initial_memory_limit = Some(initial_memory_limit);
        self
    }

    pub fn build(self) -> Option<TwoBitBfs<T, Encoder, Decoder, Expander, EXPANSION_NODES>> {
        // Require that all chunks are the same size
        let chunk_size_bytes = self.chunk_size_bytes?;
        let state_size = self.state_size? as usize;
        if state_size % (4 * chunk_size_bytes) != 0 {
            return None;
        }

        Some(TwoBitBfs {
            encoder: self.encoder?,
            decoder: self.decoder?,
            expander: self.expander?,
            threads: self.threads,
            chunk_size_bytes: self.chunk_size_bytes?,
            initial_state: self.initial_state?,
            state_size: self.state_size?,
            array_file_directory: self.array_file_directory?,
            update_file_directory: self.update_file_directory?,
            initial_memory_limit: self.initial_memory_limit?,
            phantom_t: PhantomData,
        })
    }
}

pub struct TwoBitBfs<
    T,
    Encoder: Fn(&T) -> u64,
    Decoder: Fn(&mut T, u64),
    Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]),
    const EXPANSION_NODES: usize,
> {
    encoder: Encoder,
    decoder: Decoder,
    expander: Expander,
    threads: u64,
    chunk_size_bytes: usize,
    initial_state: T,
    state_size: u64,
    array_file_directory: PathBuf,
    update_file_directory: PathBuf,
    initial_memory_limit: usize,
    phantom_t: PhantomData<T>,
}
