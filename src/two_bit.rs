use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    marker::PhantomData,
    path::PathBuf,
};

pub struct TwoBitBfsBuilder<
    T: Clone + Sync,
    Encoder: Fn(&T) -> u64 + Sync,
    Decoder: Fn(&mut T, u64) + Sync,
    Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]) + Sync,
    const EXPANSION_NODES: usize,
> {
    encoder: Option<Encoder>,
    decoder: Option<Decoder>,
    expander: Option<Expander>,
    threads: u64,
    chunk_size_bytes: Option<usize>,
    update_set_capacity: Option<usize>,
    capacity_check_frequency: Option<usize>,
    initial_state: Option<T>,
    state_size: Option<u64>,
    array_file_directory: Option<PathBuf>,
    update_file_directory: Option<PathBuf>,
    initial_memory_limit: Option<usize>,
    phantom_t: PhantomData<T>,
}

impl<
        T: Clone + Sync,
        Encoder: Fn(&T) -> u64 + Sync,
        Decoder: Fn(&mut T, u64) + Sync,
        Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]) + Sync,
        const EXPANSION_NODES: usize,
    > Default for TwoBitBfsBuilder<T, Encoder, Decoder, Expander, EXPANSION_NODES>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        T: Clone + Sync,
        Encoder: Fn(&T) -> u64 + Sync,
        Decoder: Fn(&mut T, u64) + Sync,
        Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]) + Sync,
        const EXPANSION_NODES: usize,
    > TwoBitBfsBuilder<T, Encoder, Decoder, Expander, EXPANSION_NODES>
{
    pub fn new() -> Self {
        Self {
            encoder: None,
            decoder: None,
            expander: None,
            threads: 1,
            chunk_size_bytes: None,
            update_set_capacity: None,
            capacity_check_frequency: None,
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

    pub fn threads(mut self, threads: u64) -> Self {
        self.threads = threads;
        self
    }

    pub fn chunk_size_bytes(mut self, chunk_size_bytes: usize) -> Self {
        // Limit to 2^30 bytes so that we can store 32 bit values in the update files
        if chunk_size_bytes < 1 << 30 {
            self.chunk_size_bytes = Some(chunk_size_bytes);
        }
        self
    }

    pub fn update_set_capacity(mut self, update_set_capacity: usize) -> Self {
        self.update_set_capacity = Some(update_set_capacity);
        self
    }

    pub fn capacity_check_frequency(mut self, capacity_check_frequency: usize) -> Self {
        self.capacity_check_frequency = Some(capacity_check_frequency);
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
            update_set_capacity: self.update_set_capacity?,
            capacity_check_frequency: self.capacity_check_frequency?,
            initial_state: self.initial_state?,
            state_size: self.state_size?,
            array_file_directory: self.array_file_directory?,
            update_file_directory: self.update_file_directory?,
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
    T: Clone + Sync,
    Encoder: Fn(&T) -> u64 + Sync,
    Decoder: Fn(&mut T, u64) + Sync,
    Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]) + Sync,
    const EXPANSION_NODES: usize,
> {
    encoder: Encoder,
    decoder: Decoder,
    expander: Expander,
    threads: u64,
    chunk_size_bytes: usize,
    update_set_capacity: usize,
    capacity_check_frequency: usize,
    initial_state: T,
    state_size: u64,
    array_file_directory: PathBuf,
    update_file_directory: PathBuf,
    initial_memory_limit: usize,
    phantom_t: PhantomData<T>,
}

impl<
        T: Clone + Sync,
        Encoder: Fn(&T) -> u64 + Sync,
        Decoder: Fn(&mut T, u64) + Sync,
        Expander: Fn(&mut T, &mut [u64; EXPANSION_NODES]) + Sync,
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

    fn states_per_chunk(&self) -> usize {
        self.chunk_size_bytes * 4
    }

    /// Updates a chunk from depth `depth` to depth `depth + 1`
    fn update_chunk(
        &self,
        chunk_bytes: &mut [u8],
        chunk_idx: usize,
        depth: usize,
        next: u8,
    ) -> u64 {
        // Read the chunk from disk
        let dir_path = self.array_file_directory.join(format!("depth-{depth}"));
        let file_path = dir_path.join(format!("chunk-{chunk_idx}.dat"));
        let mut file = File::open(file_path).unwrap();

        // Check that the file size is correct
        let expected_size = self.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        file.read_exact(chunk_bytes).unwrap();

        // Set current positions to old
        let current = if next == NEXT { CURRENT } else { NEXT };
        for byte in chunk_bytes.iter_mut() {
            for bit_idx in (0..8).step_by(2) {
                if (*byte >> bit_idx) & 0b11 == current {
                    let mask = 0b11 << bit_idx;
                    *byte = (*byte & !mask) | OLD << bit_idx;
                }
            }
        }

        let mut new_positions = 0u64;

        for i in 0..self.num_array_chunks() {
            let dir_path = self
                .update_file_directory
                .join(format!("depth-{}", depth + 1))
                .join(format!("update-chunk-{chunk_idx}"))
                .join(format!("from-chunk-{i}"));

            for file_path in std::fs::read_dir(&dir_path)
                .unwrap()
                .flatten()
                .map(|entry| entry.path())
            {
                let file = File::open(file_path).unwrap();
                let expected_entries = file.metadata().unwrap().len() / 4;
                let mut reader = BufReader::new(file);

                // Read 4 bytes at a time, and update the current chunk
                let mut buf = [0u8; 4];
                let mut entries = 0;

                while let Ok(bytes_read) = reader.read(&mut buf) {
                    if bytes_read == 0 {
                        break;
                    }

                    let chunk_offset = u32::from_le_bytes(buf);
                    let (byte_idx, bit_idx) = self.chunk_offset_to_bit_coords(chunk_offset);
                    let byte = chunk_bytes[byte_idx];

                    if (byte >> bit_idx) & 0b11 == UNSEEN {
                        let mask = 0b11 << bit_idx;
                        let new_byte = (byte & !mask) | next << bit_idx;
                        chunk_bytes[byte_idx] = new_byte;

                        new_positions += 1;
                    }

                    entries += 1;
                }

                assert_eq!(entries, expected_entries);
            }
        }

        tracing::info!("depth {} chunk {chunk_idx} new {new_positions}", depth + 1);

        // Write new chunk
        let new_chunk_dir = self
            .array_file_directory
            .join(format!("depth-{}", depth + 1));

        std::fs::create_dir_all(&new_chunk_dir).unwrap();

        let new_chunk_path = new_chunk_dir.join(format!("chunk-{chunk_idx}.dat"));
        let mut new_chunk_file = File::create_new(new_chunk_path).unwrap();

        new_chunk_file.write_all(chunk_bytes).unwrap();

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

        new_positions
    }

    fn write_update_file(
        &self,
        depth: usize,
        updated_chunk_idx: usize,
        from_chunk_idx: usize,
        update_set: &mut HashSet<u32>,
    ) {
        let dir_path = self
            .update_file_directory
            .join(format!("depth-{}", depth + 1))
            .join(format!("update-chunk-{updated_chunk_idx}"))
            .join(format!("from-chunk-{from_chunk_idx}"));

        std::fs::create_dir_all(&dir_path).unwrap();

        let part = std::fs::read_dir(&dir_path).unwrap().flatten().count();
        let file_path = dir_path.join(format!("part-{}.dat", part));
        let file = File::create_new(file_path).unwrap();
        let mut writer = BufWriter::new(file);

        for val in update_set.drain() {
            if writer.write(&val.to_le_bytes()).unwrap() != 4 {
                panic!("failed to write to update file");
            }
        }
    }

    fn expand_chunk(&self, chunk_buffer: &mut [u8], chunk_idx: usize, depth: usize, current: u8) {
        let mut state = self.initial_state.clone();
        let mut expanded = [0u64; EXPANSION_NODES];

        // Read the chunk from disk
        let dir_path = self.array_file_directory.join(format!("depth-{depth}"));
        let file_path = dir_path.join(format!("chunk-{chunk_idx}.dat"));
        let mut file = File::open(file_path).unwrap();

        // Check that the file size is correct
        let expected_size = self.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        file.read_exact(chunk_buffer).unwrap();

        // Create update sets
        let mut update_sets =
            vec![HashSet::<u32>::with_capacity(self.update_set_capacity); self.num_array_chunks()];

        // Expand current nodes
        for (chunk_offset, byte) in chunk_buffer.iter().enumerate() {
            if chunk_offset % self.capacity_check_frequency == 0 {
                // Check if any of the update sets may go over capacity
                let max_new_nodes = self.capacity_check_frequency * EXPANSION_NODES;

                for (idx, set) in update_sets.iter_mut().enumerate() {
                    if set.len() + max_new_nodes > set.capacity() {
                        // Possible to reach capacity on the next block of expansions, so
                        // write update file to disk
                        self.write_update_file(depth, idx, chunk_idx, set);
                    }
                }
            }

            for bit_idx in (0..8).step_by(2) {
                let val = (byte >> bit_idx) & 0b11;

                if val == current {
                    let encoded = self.bit_coords_to_node(chunk_idx, chunk_offset, bit_idx);
                    self.decode(&mut state, encoded);
                    self.expand(&mut state, &mut expanded);
                    for node in expanded {
                        let (idx, offset) = self.node_to_chunk_coords(node);
                        update_sets[idx].insert(offset);
                    }
                }
            }
        }

        // Write remaining update files
        for (idx, set) in update_sets.iter_mut().enumerate() {
            self.write_update_file(depth, idx, chunk_idx, set);
        }
    }

    /// Converts an encoded node value to (chunk_idx, byte_idx, bit_idx)
    fn node_to_bit_coords(&self, node: u64) -> (usize, usize, usize) {
        let node = node as usize;
        let chunk_idx = (node / 4) / self.chunk_size_bytes;
        let byte_idx = (node / 4) % self.chunk_size_bytes;
        let bit_idx = 2 * (node % 4);
        (chunk_idx, byte_idx, bit_idx)
    }

    fn bit_coords_to_node(&self, chunk_idx: usize, byte_idx: usize, bit_idx: usize) -> u64 {
        (chunk_idx * self.chunk_size_bytes * 4 + byte_idx * 4 + bit_idx / 2) as u64
    }

    /// Converts an encoded node value to (chunk_idx, chunk_offset)
    fn node_to_chunk_coords(&self, node: u64) -> (usize, u32) {
        let node = node as usize;
        let n = self.states_per_chunk();
        (node / n, (node % n) as u32)
    }

    fn chunk_offset_to_bit_coords(&self, chunk_offset: u32) -> (usize, usize) {
        let byte_idx = (chunk_offset / 4) as usize;
        let bit_idx = 2 * (chunk_offset % 4) as usize;
        (byte_idx, bit_idx)
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

        tracing::info!("starting in-memory BFS");

        loop {
            new = 0;

            for &encoded in &current {
                self.decode(&mut state, encoded);
                self.expand(&mut state, &mut expanded);
                for node in expanded {
                    if !old.contains(&node) && !current.contains(&node) && next.insert(node) {
                        new += 1;
                    }
                }
            }
            depth += 1;
            total += new;

            tracing::info!("depth {depth} new {new} total {total}");

            // No new nodes, we are done already.
            if new == 0 {
                tracing::info!("no new nodes, done");
                return;
            }

            if total > max_capacity {
                tracing::info!("exceeded memory limit");
                break;
            }

            for val in current.drain() {
                old.insert(val);
            }
            std::mem::swap(&mut current, &mut next);
        }

        // We ran out of memory. Continue BFS using disk.

        tracing::info!("starting disk BFS");

        for chunk_idx in 0..self.num_array_chunks() {
            tracing::info!("creating chunk {chunk_idx}");

            const UNSEEN_BYTE: u8 = UNSEEN * 0b01010101;
            let mut chunk_bytes = vec![UNSEEN_BYTE; self.chunk_size_bytes];

            // Update values from `old`
            for (_, byte_idx, bit_idx) in old
                .iter()
                .map(|&val| self.node_to_bit_coords(val))
                .filter(|&(i, _, _)| chunk_idx == i)
            {
                let byte = chunk_bytes[byte_idx];
                let mask = 0b11 << bit_idx;
                let new_byte = (byte & !mask) | OLD << bit_idx;
                chunk_bytes[byte_idx] = new_byte;
            }

            // Update values from `next` and make them current
            for (_, byte_idx, bit_idx) in next
                .iter()
                .map(|&val| self.node_to_bit_coords(val))
                .filter(|&(i, _, _)| chunk_idx == i)
            {
                let byte = chunk_bytes[byte_idx];
                let mask = 0b11 << bit_idx;
                let new_byte = (byte & !mask) | CURRENT << bit_idx;
                chunk_bytes[byte_idx] = new_byte;
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

        // Because of how chunks get updated, the values of `NEXT` and `CURRENT` get swapped every
        // iteration (see Korf's paper, in the paragraph "Eliminating the Conversion Scan").
        let mut next_and_current_swapped = false;

        loop {
            let (current, next) = if next_and_current_swapped {
                (NEXT, CURRENT)
            } else {
                (CURRENT, NEXT)
            };

            let new_positions = std::thread::scope(|s| {
                // Expand chunks
                let threads = (0..self.threads as usize)
                    .map(|thread_idx| {
                        s.spawn(move || {
                            let mut chunk_bytes = vec![0u8; self.chunk_size_bytes];

                            for chunk_idx in (0..self.num_array_chunks())
                                .skip(thread_idx)
                                .step_by(self.threads as usize)
                            {
                                tracing::info!("[Thread {thread_idx}] expanding chunk {chunk_idx}");
                                self.expand_chunk(&mut chunk_bytes, chunk_idx, depth, current);
                            }
                        })
                    })
                    .collect::<Vec<_>>();

                threads.into_iter().for_each(|t| t.join().unwrap());

                let threads = (0..self.threads as usize)
                    .map(|thread_idx| {
                        s.spawn(move || {
                            let mut new_positions = 0;

                            // Read the update files and update the array chunks
                            let mut chunk_bytes = vec![0u8; self.chunk_size_bytes];

                            for chunk_idx in (0..self.num_array_chunks())
                                .skip(thread_idx)
                                .step_by(self.threads as usize)
                            {
                                tracing::info!("[Thread {thread_idx}] updating chunk {chunk_idx}");
                                new_positions +=
                                    self.update_chunk(&mut chunk_bytes, chunk_idx, depth, next);
                            }

                            new_positions
                        })
                    })
                    .collect::<Vec<_>>();

                threads.into_iter().map(|t| t.join().unwrap()).sum::<u64>()
            });

            tracing::info!("depth {} new {new_positions}", depth + 1);

            if new_positions == 0 {
                break;
            }

            depth += 1;
            next_and_current_swapped = !next_and_current_swapped;
        }
    }
}
