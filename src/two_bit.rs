use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    marker::PhantomData,
    path::PathBuf,
};

pub struct TwoBitBfsBuilder<
    T: Clone,
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
    info_directory: Option<PathBuf>,
    initial_memory_limit: Option<usize>,
    phantom_t: PhantomData<T>,
}

impl<
        T: Clone,
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
            info_directory: None,
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

    pub fn info_directory(mut self, info_directory: PathBuf) -> Self {
        self.info_directory = Some(info_directory);
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
            info_directory: self.info_directory?,
            initial_memory_limit: self.initial_memory_limit?,
            phantom_t: PhantomData,
        })
    }
}

const UNSEEN: u8 = 0b00;
const CURRENT: u8 = 0b01;
const NEXT: u8 = 0b10;
const OLD: u8 = 0b11;

pub struct TwoBitBfs<
    T: Clone,
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
    info_directory: PathBuf,
    initial_memory_limit: usize,
    phantom_t: PhantomData<T>,
}

impl<
        T: Clone,
        Encoder: Fn(&T) -> u64,
        Decoder: Fn(&mut T, u64),
        Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]),
        const EXPANSION_NODES: usize,
    > TwoBitBfs<T, Encoder, Decoder, Expander, EXPANSION_NODES>
{
    fn encode(&self, state: &T) -> u64 {
        (self.encoder)(state)
    }

    fn decode(&self, state: &mut T, encoded: u64) {
        (self.decoder)(state, encoded);
    }

    fn expand(&self, state: &mut T, nodes: &mut [u64; EXPANSION_NODES]) {
        (self.expander)(state, nodes);
    }

    fn array_bytes(&self) -> usize {
        self.state_size.div_ceil(4) as usize
    }

    fn num_array_chunks(&self) -> usize {
        self.array_bytes() / self.chunk_size_bytes
    }

    /// Updates a chunk from depth `depth` to depth `depth + 1`
    fn update_chunk(&self, chunk_bytes: &mut [u8], chunk_idx: usize, depth: usize, next: u8) {
        // Read the chunk from disk
        let dir_path = self.array_file_directory.join(format!("depth-{depth}"));
        let file_path = dir_path.join(format!("chunk-{chunk_idx}.dat"));
        let mut file = File::open(file_path).unwrap();

        // Check that the file size is correct
        let expected_size = self.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        file.read_exact(chunk_bytes).unwrap();

        for i in 0..self.num_array_chunks() {
            let file_path = self
                .update_file_directory
                .join(format!("depth-{}", depth + 1))
                .join(format!("update-chunk-{chunk_idx}"))
                .join(format!("from-chunk-{i}.dat"));
            let file = File::open(file_path).unwrap();
            let expected_entries = file.metadata().unwrap().len() / 8;
            let mut reader = BufReader::new(file);

            // Read 8 bytes at a time, and update the current chunk
            let mut buf = [0u8; 8];
            let mut entries = 0;

            let mut new_positions = 0u64;

            while let Ok(bytes_read) = reader.read(&mut buf) {
                if bytes_read == 0 {
                    break;
                }

                let val = u64::from_le_bytes(buf);
                let (_, chunk_offset, byte_offset) = self.to_chunk_idx(val);
                let byte = chunk_bytes[chunk_offset];

                if byte >> (byte_offset * 2) == UNSEEN {
                    let mask = 0b11 << (byte_offset * 2);
                    let new_byte = (byte & !mask) | next << (byte_offset * 2);
                    chunk_bytes[chunk_offset] = new_byte;

                    new_positions += 1;
                }

                entries += 1;
            }

            assert_eq!(entries, expected_entries);

            // Write info file containing number of new positions
            let info_dir_path = self
                .info_directory
                .join(format!("depth-{}", depth + 1))
                .join(format!("update-chunk-{chunk_idx}"));

            std::fs::create_dir_all(&info_dir_path).unwrap();

            let info_file_path = info_dir_path.join(format!("from-chunk-{i}.info"));

            let mut info_file = File::create_new(info_file_path).unwrap();
            info_file.write_all(&new_positions.to_le_bytes()).unwrap();
        }

        // Write new chunk
        let new_chunk_dir = self
            .array_file_directory
            .join(format!("depth-{}", depth + 1));

        std::fs::create_dir_all(&new_chunk_dir).unwrap();

        let new_chunk_path = new_chunk_dir.join(format!("chunk-{chunk_idx}.dat"));
        let mut new_chunk_file = File::create_new(new_chunk_path).unwrap();

        new_chunk_file.write_all(&chunk_bytes).unwrap();

        // Delete the old chunk file
        let old_chunk_path = self
            .array_file_directory
            .join(format!("depth-{depth}"))
            .join(format!("chunk-{chunk_idx}.dat"));
        std::fs::remove_file(old_chunk_path).unwrap();

        // Delete all the update files
        let update_dir_path = self
            .update_file_directory
            .join(format!("depth-{}", depth + 1))
            .join(format!("update-chunk-{chunk_idx}"));
        std::fs::remove_dir_all(update_dir_path).unwrap();
    }

    fn count_new_positions(&self, depth: usize) -> u64 {
        let mut new_positions = 0;

        for chunk_idx in 0..self.num_array_chunks() {
            for i in 0..self.num_array_chunks() {
                let info_file_path = self
                    .info_directory
                    .join(format!("depth-{depth}"))
                    .join(format!("update-chunk-{chunk_idx}"))
                    .join(format!("from-chunk-{i}.info"));
                let mut info_file = File::open(info_file_path).unwrap();
                let mut buf = [0u8; 8];
                info_file.read_exact(&mut buf).unwrap();
                let count = u64::from_le_bytes(buf);
                new_positions += count;
            }
        }

        new_positions
    }

    /// Converts an encoded node value to (chunk_idx, chunk_offset, byte_offset)
    fn to_chunk_idx(&self, encoded: u64) -> (usize, usize, usize) {
        let encoded = encoded as usize;
        let chunk_idx = (encoded / 4) / self.chunk_size_bytes;
        let chunk_offset = (encoded / 4) % self.chunk_size_bytes;
        let byte_offset = encoded % 4;
        (chunk_idx, chunk_offset, byte_offset)
    }

    fn from_chunk_idx(&self, chunk_idx: usize, chunk_offset: usize, byte_offset: usize) -> u64 {
        (chunk_idx * self.chunk_size_bytes * 4 + chunk_offset * 4 + byte_offset) as u64
    }

    pub fn run(&self) {
        let max_capacity = self.initial_memory_limit / 8;

        // Do the initial iterations of BFS in memory
        let mut old = HashSet::<u64>::with_capacity(max_capacity / 2);
        let mut current = HashSet::<u64>::new();
        let mut next = HashSet::<u64>::new();

        let mut state = self.initial_state.clone();
        let mut expanded = [0u64; EXPANSION_NODES];
        let mut depth = 0;

        current.insert(self.encode(&state));

        let mut new;
        let mut total = 1;


        loop {
            new = 0;

            for &encoded in &current {
                self.decode(&mut state, encoded);
                self.expand(&mut state, &mut expanded);
                for node in expanded {
                    if !old.contains(&node) && !current.contains(&node) {
                        if next.insert(node) {
                            new += 1;
                        }
                    }
                }
            }
            depth += 1;
            total += new;

            println!("depth {depth} new {new} total {total}");

            // No new nodes, we are done already.
            if new == 0 {
                return;
            }

            if total > max_capacity {
                break;
            }

            for val in current.drain() {
                old.insert(val);
            }
            std::mem::swap(&mut current, &mut next);
        }

        // We ran out of memory. Continue BFS using disk.

        for chunk_idx in 0..self.num_array_chunks() {
            const UNSEEN_BYTE: u8 = UNSEEN * 0b01010101;
            let mut chunk_bytes = vec![UNSEEN_BYTE; self.chunk_size_bytes];

            // Update values from `old`
            for (_, chunk_offset, byte_offset) in old
                .iter()
                .map(|&val| self.to_chunk_idx(val))
                .filter(|&(i, _, _)| chunk_idx == i)
            {
                let byte = chunk_bytes[chunk_offset];
                let mask = 0b11 << (byte_offset * 2);
                let new_byte = (byte & !mask) | OLD << (byte_offset * 2);
                chunk_bytes[chunk_offset] = new_byte;
            }

            // Update values from `next` and make them current
            for (_, chunk_offset, byte_offset) in next
                .iter()
                .map(|&val| self.to_chunk_idx(val))
                .filter(|&(i, _, _)| chunk_idx == i)
            {
                let byte = chunk_bytes[chunk_offset];
                let mask = 0b11 << (byte_offset * 2);
                let new_byte = (byte & !mask) | CURRENT << (byte_offset * 2);
                chunk_bytes[chunk_offset] = new_byte;
            }

            // Write the updated chunk to disk
            let dir_path = self.array_file_directory.join(format!("depth-{depth}"));
            std::fs::create_dir_all(&dir_path).unwrap();

            let file_path = dir_path.join(format!("chunk-{chunk_idx}.dat"));
            let mut file = File::create_new(file_path).unwrap();
            file.write_all(&chunk_bytes).unwrap();
        }

        drop(old);

        // Continue with BFS using disk

        let mut chunk_bytes = vec![0u8; self.chunk_size_bytes];

        // Because of how chunks get updated, the values of `NEXT` and `CURRENT` get swapped every
        // iteration (see Korf's paper, in the paragraph "Eliminating the Conversion Scan").
        let mut next_and_current_swapped = false;

        loop {
            let (current, next) = if next_and_current_swapped {
                (NEXT, CURRENT)
            } else {
                (CURRENT, NEXT)
            };

            for chunk_idx in 0..self.num_array_chunks() {
                // Read the chunk from disk
                let dir_path = self.array_file_directory.join(format!("depth-{depth}"));
                let file_path = dir_path.join(format!("chunk-{chunk_idx}.dat"));
                let mut file = File::open(file_path).unwrap();

                // Check that the file size is correct
                let expected_size = self.chunk_size_bytes;
                let actual_size = file.metadata().unwrap().len();
                assert_eq!(expected_size, actual_size as usize);

                file.read_exact(&mut chunk_bytes).unwrap();

                // Create update files
                let mut update_files = (0..self.num_array_chunks())
                    .map(|i| {
                        let dir_path = self
                            .update_file_directory
                            .join(format!("depth-{}", depth + 1))
                            .join(format!("update-chunk-{i}"));

                        std::fs::create_dir_all(&dir_path).unwrap();

                        let file_path = dir_path.join(format!("from-chunk-{chunk_idx}.dat"));
                        let file = File::create_new(file_path).unwrap();

                        BufWriter::new(file)
                    })
                    .collect::<Vec<_>>();

                // Expand current nodes
                for chunk_offset in 0..self.chunk_size_bytes {
                    for byte_offset in 0..4 {
                        let byte = chunk_bytes[chunk_offset];
                        let val = (byte >> (byte_offset * 2)) & 0b11;

                        if val == current {
                            let encoded = self.from_chunk_idx(chunk_idx, chunk_offset, byte_offset);
                            self.decode(&mut state, encoded);
                            self.expand(&mut state, &mut expanded);
                            for node in expanded {
                                let (chunk_idx, _, _) = self.to_chunk_idx(node);
                                update_files[chunk_idx].write(&node.to_le_bytes()).unwrap();
                            }

                            // Set the val to `OLD` after expanding
                            let mask = 0b11 << (byte_offset * 2);
                            let new_byte = (byte & !mask) | OLD << (byte_offset * 2);
                            chunk_bytes[chunk_offset] = new_byte;
                        }
                    }
                }
            }

            // Read the update files and update the array chunks
            for chunk_idx in 0..self.num_array_chunks() {
                self.update_chunk(&mut chunk_bytes, chunk_idx, depth, next);
            }

            // Read the info files and count all new positions
            let new_positions = self.count_new_positions(depth + 1);

            println!("depth {} new {new_positions}", depth + 1);

            if new_positions == 0 {
                break;
            }

            depth += 1;
            next_and_current_swapped = !next_and_current_swapped;
        }
    }
}
