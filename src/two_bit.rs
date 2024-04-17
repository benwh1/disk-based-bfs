use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
};

pub struct TwoBitBfsBuilder<
    T: Default + Sync,
    Expander: Fn(&mut T, u64, &mut [u64; EXPANSION_NODES]) + Sync,
    const EXPANSION_NODES: usize,
> {
    expander: Option<Expander>,
    threads: usize,
    chunk_size_bytes: Option<usize>,
    update_set_capacity: Option<usize>,
    capacity_check_frequency: Option<usize>,
    initial_states: Option<Vec<u64>>,
    state_size: Option<u64>,
    root_directories: Option<Vec<PathBuf>>,
    initial_memory_limit: Option<usize>,
    phantom_t: PhantomData<T>,
}

impl<
        T: Default + Sync,
        Expander: Fn(&mut T, u64, &mut [u64; EXPANSION_NODES]) + Sync,
        const EXPANSION_NODES: usize,
    > Default for TwoBitBfsBuilder<T, Expander, EXPANSION_NODES>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        T: Default + Sync,
        Expander: Fn(&mut T, u64, &mut [u64; EXPANSION_NODES]) + Sync,
        const EXPANSION_NODES: usize,
    > TwoBitBfsBuilder<T, Expander, EXPANSION_NODES>
{
    pub fn new() -> Self {
        Self {
            expander: None,
            threads: 1,
            chunk_size_bytes: None,
            update_set_capacity: None,
            capacity_check_frequency: None,
            initial_states: None,
            state_size: None,
            root_directories: None,
            initial_memory_limit: None,
            phantom_t: PhantomData,
        }
    }

    pub fn expander(mut self, expander: Expander) -> Self {
        self.expander = Some(expander);
        self
    }

    pub fn threads(mut self, threads: usize) -> Self {
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

    pub fn initial_states(mut self, initial_states: &[u64]) -> Self {
        self.initial_states = Some(initial_states.to_vec());
        self
    }

    pub fn state_size(mut self, state_size: u64) -> Self {
        self.state_size = Some(state_size);
        self
    }

    pub fn root_directories(mut self, root_directories: &[PathBuf]) -> Self {
        self.root_directories = Some(root_directories.to_vec());
        self
    }

    pub fn initial_memory_limit(mut self, initial_memory_limit: usize) -> Self {
        self.initial_memory_limit = Some(initial_memory_limit);
        self
    }

    pub fn build(self) -> Option<TwoBitBfs<T, Expander, EXPANSION_NODES>> {
        // Require that all chunks are the same size
        let chunk_size_bytes = self.chunk_size_bytes?;
        let state_size = self.state_size? as usize;
        if state_size % (4 * chunk_size_bytes) != 0 {
            return None;
        }

        Some(TwoBitBfs {
            expander: self.expander?,
            threads: self.threads,
            chunk_size_bytes: self.chunk_size_bytes?,
            update_set_capacity: self.update_set_capacity?,
            capacity_check_frequency: self.capacity_check_frequency?,
            initial_states: self.initial_states?,
            state_size: self.state_size?,
            root_directories: self.root_directories?,
            initial_memory_limit: self.initial_memory_limit?,
            phantom_t: PhantomData,
        })
    }
}

const UNSEEN: u8 = 0b00;
const CURRENT: u8 = 0b01;
const NEXT: u8 = 0b10;
const OLD: u8 = 0b11;

pub enum InMemoryBfsResult {
    Complete,
    OutOfMemory {
        old: HashSet<u64>,
        next: HashSet<u64>,
        depth: usize,
    },
}

pub struct TwoBitBfs<
    T: Default + Sync,
    Expander: Fn(&mut T, u64, &mut [u64; EXPANSION_NODES]) + Sync,
    const EXPANSION_NODES: usize,
> {
    expander: Expander,
    threads: usize,
    chunk_size_bytes: usize,
    update_set_capacity: usize,
    capacity_check_frequency: usize,
    initial_states: Vec<u64>,
    state_size: u64,
    root_directories: Vec<PathBuf>,
    initial_memory_limit: usize,
    phantom_t: PhantomData<T>,
}

impl<
        T: Default + Sync,
        Expander: Fn(&mut T, u64, &mut [u64; EXPANSION_NODES]) + Sync,
        const EXPANSION_NODES: usize,
    > TwoBitBfs<T, Expander, EXPANSION_NODES>
{
    fn expand(&self, state: &mut T, encoded: u64, nodes: &mut [u64; EXPANSION_NODES]) {
        (self.expander)(state, encoded, nodes);
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

    fn root_dir(&self, chunk_idx: usize) -> &Path {
        &self.root_directories[chunk_idx % self.root_directories.len()]
    }

    fn update_chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("update")
            .join(format!("depth-{depth}"))
            .join(format!("update-chunk-{chunk_idx}"))
    }

    fn update_chunk_from_chunk_dir_path(
        &self,
        depth: usize,
        updated_chunk_idx: usize,
        from_chunk_idx: usize,
    ) -> PathBuf {
        self.update_chunk_dir_path(depth, updated_chunk_idx)
            .join(format!("from-chunk-{from_chunk_idx}"))
    }

    fn chunk_dir_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.root_dir(chunk_idx)
            .join("array")
            .join(format!("depth-{depth}"))
    }

    fn chunk_file_path(&self, depth: usize, chunk_idx: usize) -> PathBuf {
        self.chunk_dir_path(depth, chunk_idx)
            .join(format!("chunk-{chunk_idx}.dat"))
    }

    fn read_chunk(&self, chunk_buffer: &mut [u8], chunk_idx: usize, depth: usize) {
        let file_path = self.chunk_file_path(depth, chunk_idx);
        let mut file = File::open(file_path).unwrap();

        // Check that the file size is correct
        let expected_size = self.chunk_size_bytes;
        let actual_size = file.metadata().unwrap().len();
        assert_eq!(expected_size, actual_size as usize);

        file.read_exact(chunk_buffer).unwrap();
    }

    fn write_update_file(
        &self,
        depth: usize,
        updated_chunk_idx: usize,
        from_chunk_idx: usize,
        update_set: &mut HashSet<u32>,
    ) {
        let dir_path =
            self.update_chunk_from_chunk_dir_path(depth + 1, updated_chunk_idx, from_chunk_idx);

        std::fs::create_dir_all(&dir_path).unwrap();

        let part = std::fs::read_dir(&dir_path).unwrap().flatten().count();
        let file_path_tmp = dir_path.join(format!("part-{part}.dat.tmp"));
        let file = File::create(&file_path_tmp).unwrap();
        let mut writer = BufWriter::new(file);

        for val in update_set.drain() {
            if writer.write(&val.to_le_bytes()).unwrap() != 4 {
                panic!("failed to write to update file");
            }
        }

        drop(writer);

        let file_path = dir_path.join(format!("part-{part}.dat"));
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn write_chunk(&self, chunk_buffer: &[u8], chunk_idx: usize, depth: usize) {
        let dir_path = self.chunk_dir_path(depth, chunk_idx);

        std::fs::create_dir_all(&dir_path).unwrap();

        let file_path_tmp = dir_path.join(format!("chunk-{chunk_idx}.dat.tmp"));
        let mut file = File::create(&file_path_tmp).unwrap();

        file.write_all(chunk_buffer).unwrap();
        drop(file);

        let file_path = self.chunk_file_path(depth, chunk_idx);
        std::fs::rename(file_path_tmp, file_path).unwrap();
    }

    fn delete_chunk_file(&self, depth: usize, chunk_idx: usize) {
        let file_path = self.chunk_file_path(depth, chunk_idx);
        if file_path.exists() {
            std::fs::remove_file(file_path).unwrap();
        }
    }

    fn delete_update_files(&self, depth: usize, chunk_idx: usize) {
        let dir_path = self.update_chunk_dir_path(depth, chunk_idx);
        if dir_path.exists() {
            std::fs::remove_dir_all(dir_path).unwrap();
        }
    }

    fn demote_chunk(&self, chunk_buffer: &mut [u8], depth: usize) {
        let current = if depth % 2 == 0 { CURRENT } else { NEXT };

        // Set current positions to old
        for byte in chunk_buffer.iter_mut() {
            for bit_idx in (0..8).step_by(2) {
                if (*byte >> bit_idx) & 0b11 == current {
                    let mask = 0b11 << bit_idx;
                    *byte = (*byte & !mask) | OLD << bit_idx;
                }
            }
        }
    }

    fn update_chunk(&self, chunk_buffer: &mut [u8], chunk_idx: usize, depth: usize) -> u64 {
        let next = if depth % 2 == 0 { NEXT } else { CURRENT };

        let mut new_positions = 0u64;

        for i in 0..self.num_array_chunks() {
            let dir_path = self.update_chunk_from_chunk_dir_path(depth + 1, chunk_idx, i);

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
                    let byte = chunk_buffer[byte_idx];

                    if (byte >> bit_idx) & 0b11 == UNSEEN {
                        let mask = 0b11 << bit_idx;
                        let new_byte = (byte & !mask) | next << bit_idx;
                        chunk_buffer[byte_idx] = new_byte;

                        new_positions += 1;
                    }

                    entries += 1;
                }

                assert_eq!(entries, expected_entries);
            }
        }

        new_positions
    }

    fn expand_chunk(&self, chunk_buffer: &[u8], chunk_idx: usize, depth: usize) {
        let mut state = T::default();
        let mut expanded = [0u64; EXPANSION_NODES];

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
                let current = if depth % 2 == 0 { CURRENT } else { NEXT };

                if val == current {
                    let encoded = self.bit_coords_to_node(chunk_idx, chunk_offset, bit_idx);
                    self.expand(&mut state, encoded, &mut expanded);
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

    fn in_memory_bfs(&self) -> InMemoryBfsResult {
        let max_capacity = self.initial_memory_limit / 8;

        let mut old = HashSet::<u64>::with_capacity(max_capacity / 2);
        let mut current = HashSet::<u64>::new();
        let mut next = HashSet::<u64>::new();

        let mut state = T::default();
        let mut expanded = [0u64; EXPANSION_NODES];
        let mut depth = 0;

        for &state in &self.initial_states {
            current.insert(state);
        }

        let mut new;
        let mut total = 1;

        tracing::info!("starting in-memory BFS");

        loop {
            new = 0;

            for &encoded in &current {
                self.expand(&mut state, encoded, &mut expanded);
                for node in expanded {
                    if !old.contains(&node) && !current.contains(&node) && next.insert(node) {
                        new += 1;
                    }
                }
            }
            depth += 1;
            total += new;

            tracing::info!("depth {depth} new {new}");

            // No new nodes, we are done already.
            if new == 0 {
                tracing::info!("no new nodes, done");
                return InMemoryBfsResult::Complete;
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

        InMemoryBfsResult::OutOfMemory { old, next, depth }
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
        let (old, next, mut depth) = match self.in_memory_bfs() {
            InMemoryBfsResult::Complete => return,
            InMemoryBfsResult::OutOfMemory { old, next, depth } => (old, next, depth),
        };

        tracing::info!("starting disk BFS");

        // Create chunks and do the first expansion
        std::thread::scope(|s| {
            let threads = (0..self.threads)
                .map(|thread_idx| {
                    let old = &old;
                    let next = &next;

                    s.spawn(move || {
                        for chunk_idx in (0..self.num_array_chunks())
                            .skip(thread_idx)
                            .step_by(self.threads)
                        {
                            tracing::info!("[Thread {thread_idx}] creating chunk {chunk_idx}");

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
                                let current = if depth % 2 == 0 { CURRENT } else { NEXT };
                                let new_byte = (byte & !mask) | current << bit_idx;
                                chunk_bytes[byte_idx] = new_byte;
                            }

                            // Expand the chunk before writing to disk
                            self.expand_chunk(&chunk_bytes, chunk_idx, depth);

                            // Write the chunk to disk
                            let dir_path = self.chunk_dir_path(depth, chunk_idx);
                            std::fs::create_dir_all(&dir_path).unwrap();

                            let file_path_tmp = dir_path.join(format!("chunk-{chunk_idx}.dat.tmp"));
                            let mut file = File::create(&file_path_tmp).unwrap();

                            file.write_all(&chunk_bytes).unwrap();
                            drop(file);

                            let file_path = self.chunk_file_path(depth, chunk_idx);
                            std::fs::rename(file_path_tmp, file_path).unwrap();
                        }
                    })
                })
                .collect::<Vec<_>>();

            threads.into_iter().for_each(|t| t.join().unwrap());
        });

        drop(old);
        drop(next);

        let mut chunk_buffers = vec![vec![0u8; self.chunk_size_bytes]; self.threads];

        loop {
            let mut new_positions = 0;

            for group_idx in (0..self.num_array_chunks()).step_by(self.threads) {
                std::thread::scope(|s| {
                    let threads = (0..self.threads)
                        .map(|t| {
                            let mut chunk_buffer = std::mem::take(&mut chunk_buffers[t]);

                            s.spawn(move || {
                                let chunk_idx = group_idx + t;

                                // If the number of chunks isn't divisible by the number of
                                // threads, then `chunk_idx` may be out of bounds during the last
                                // group of threads.
                                if chunk_idx >= self.num_array_chunks() {
                                    return None;
                                }

                                tracing::info!("[Thread {t}] reading depth {depth} chunk {chunk_idx}");
                                self.read_chunk(&mut chunk_buffer, chunk_idx, depth);

                                tracing::info!("[Thread {t}] demoting depth {depth} chunk {chunk_idx}");
                                self.demote_chunk(&mut chunk_buffer, depth);

                                tracing::info!("[Thread {t}] updating depth {depth} -> {} chunk {chunk_idx}", depth + 1);
                                let new = self.update_chunk(&mut chunk_buffer, chunk_idx, depth);

                                tracing::info!("[Thread {t}] depth {} chunk {chunk_idx} new {new}", depth + 1);

                                tracing::info!("[Thread {t}] expanding depth {} -> {} chunk {chunk_idx}", depth + 1, depth + 2);
                                self.expand_chunk(&chunk_buffer, chunk_idx, depth + 1);

                                tracing::info!("[Thread {t}] writing depth {} chunk {chunk_idx}", depth + 1);
                                self.write_chunk(&mut chunk_buffer, chunk_idx, depth + 1);

                                tracing::info!("[Thread {t}] deleting depth {depth} chunk {chunk_idx}");
                                self.delete_chunk_file(depth, chunk_idx);

                                tracing::info!("[Thread {t}] deleting update files for depth {depth} -> {} chunk {chunk_idx}", depth + 1);
                                self.delete_update_files(depth + 1, chunk_idx);

                                Some((chunk_buffer, new))
                            })
                        })
                        .collect::<Vec<_>>();

                    for (t, thread) in threads.into_iter().enumerate() {
                        if let Some((mut chunk_buffer, new)) = thread.join().unwrap() {
                            new_positions += new;

                            // Swap the buffer back into place so the next group of threads can
                            // reuse it
                            std::mem::swap(&mut chunk_buffers[t], &mut chunk_buffer);
                        }
                    }
                });
            }

            tracing::info!("depth {} new {new_positions}", depth + 1);

            if new_positions == 0 {
                break;
            }

            depth += 1;
        }
    }
}
