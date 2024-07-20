use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    thread::Builder as ThreadBuilder,
};

use rand::distributions::{Alphanumeric, DistString as _};

use crate::{
    io::LockedIO,
    settings::BfsSettings,
    update::blocks::{AvailableUpdateBlock, FilledUpdateBlock},
};

pub struct UpdateBlockList<'a> {
    settings: &'a BfsSettings,
    locked_io: &'a LockedIO<'a>,
    available_blocks: Vec<AvailableUpdateBlock>,
    filled_blocks: Vec<FilledUpdateBlock>,
}

impl<'a> UpdateBlockList<'a> {
    pub fn new(settings: &'a BfsSettings, locked_io: &'a LockedIO) -> Self {
        let num_blocks = 2 * settings.threads * settings.num_array_chunks();

        tracing::debug!("creating {num_blocks} update blocks");

        let available_blocks = (0..num_blocks)
            .map(|_| AvailableUpdateBlock::new(settings.update_capacity_per_vec()))
            .collect::<Vec<_>>();
        let filled_blocks = Vec::new();

        Self {
            settings,
            locked_io,
            available_blocks,
            filled_blocks,
        }
    }

    pub fn num_available_blocks(&self) -> usize {
        self.available_blocks.len()
    }

    pub fn num_filled_blocks(&self) -> usize {
        self.filled_blocks.len()
    }

    /// Returns the number of bytes written for each chunk
    pub fn write_all(&mut self) -> HashMap<usize, Vec<u64>> {
        tracing::debug!("writing {} update blocks", self.filled_blocks.len());

        // Sort the updates so that all the blocks that belong in the same file are consecutive,
        // so that we can use `chunk_by_mut` to group them together
        self.filled_blocks
            .sort_unstable_by_key(|block| (block.depth(), block.chunk_idx()));

        let bytes_written = Arc::new(Mutex::new(HashMap::new()));

        let num_disks = self.locked_io.num_disks();
        std::thread::scope(|s| {
            (0..num_disks)
                .map(|t| {
                    let filled_blocks = &self.filled_blocks;
                    let settings = self.settings;
                    let locked_io = self.locked_io;

                    let bytes_written = bytes_written.clone();

                    ThreadBuilder::new()
                        .name(format!("write-update-blocks-{t}"))
                        .spawn_scoped(s, move || {
                            for chunk in filled_blocks
                                .chunk_by(|b1, b2| {
                                    (b1.depth(), b1.chunk_idx()) == (b2.depth(), b2.chunk_idx())
                                })
                                .filter(|chunk| chunk[0].chunk_idx() % num_disks == t)
                            {
                                let depth = chunk[0].depth();
                                let chunk_idx = chunk[0].chunk_idx();

                                let dir_path = settings.update_chunk_dir_path(depth, chunk_idx);
                                std::fs::create_dir_all(&dir_path).unwrap();

                                let mut rng = rand::thread_rng();
                                let file_name = Alphanumeric.sample_string(&mut rng, 16);
                                let file_path = dir_path.join(file_name);

                                let buffers = chunk
                                    .iter()
                                    .map(|block| bytemuck::cast_slice(block.updates()))
                                    .collect::<Vec<_>>();

                                let bytes_to_write =
                                    buffers.iter().map(|buf| buf.len() as u64).sum::<u64>();

                                let mut bytes_written_lock = bytes_written.lock().unwrap();
                                let sizes_for_depth = bytes_written_lock
                                    .entry(depth)
                                    .or_insert_with(|| vec![0; settings.num_array_chunks()]);
                                sizes_for_depth[chunk_idx] += bytes_to_write;
                                drop(bytes_written_lock);

                                locked_io.write_file_multiple_buffers(&file_path, &buffers);
                            }
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|t| t.join().unwrap());
        });

        for block in self.filled_blocks.drain(..) {
            self.available_blocks.push(block.clear());
        }

        tracing::debug!("finished writing update blocks");

        // We could try to move the `bytes_written` vector out of the `Arc` and `Mutex` but it's
        // easier to just clone it
        let lock = bytes_written.lock().unwrap();
        (*lock).clone()
    }

    pub(super) fn take_impl(
        &mut self,
        log: bool,
    ) -> (AvailableUpdateBlock, Option<HashMap<usize, Vec<u64>>>) {
        if log {
            tracing::debug!(
                "taking update block, {} blocks remaining",
                self.available_blocks.len(),
            );
        }

        if let Some(block) = self.available_blocks.pop() {
            return (block, None);
        }

        let bytes_written = self.write_all();

        // All blocks will be available now
        (self.available_blocks.pop().unwrap(), Some(bytes_written))
    }

    pub(super) fn take(&mut self) -> (AvailableUpdateBlock, Option<HashMap<usize, Vec<u64>>>) {
        self.take_impl(true)
    }

    pub(super) fn put(&mut self, block: FilledUpdateBlock) {
        self.filled_blocks.push(block);
    }
}
