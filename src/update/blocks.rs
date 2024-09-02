pub struct AvailableUpdateBlock {
    updates: Vec<u32>,
}

impl AvailableUpdateBlock {
    pub fn new(capacity: usize) -> Self {
        Self {
            updates: Vec::with_capacity(capacity),
        }
    }

    pub fn into_fillable(
        self,
        source_depth: usize,
        source_chunk_idx: usize,
    ) -> FillableUpdateBlock {
        FillableUpdateBlock {
            updates: self.updates,
            source_depth,
            source_chunk_idx,
        }
    }
}

pub struct FillableUpdateBlock {
    updates: Vec<u32>,

    // Depth and index of the chunk that these updates were generated from
    source_depth: usize,
    source_chunk_idx: usize,
}

impl FillableUpdateBlock {
    pub fn into_filled(self, depth: usize, chunk_idx: usize) -> FilledUpdateBlock {
        FilledUpdateBlock {
            updates: self.updates,
            depth,
            chunk_idx,
            source_depth: self.source_depth,
            source_chunk_idx: self.source_chunk_idx,
        }
    }

    pub fn source_depth(&self) -> usize {
        self.source_depth
    }

    pub fn source_chunk_idx(&self) -> usize {
        self.source_chunk_idx
    }

    pub fn push(&mut self, update: u32) {
        self.updates.push(update);
    }

    pub fn len(&self) -> usize {
        self.updates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.updates.capacity()
    }
}

pub struct FilledUpdateBlock {
    updates: Vec<u32>,
    depth: usize,
    chunk_idx: usize,
    source_depth: usize,
    source_chunk_idx: usize,
}

impl FilledUpdateBlock {
    pub fn clear(mut self) -> AvailableUpdateBlock {
        self.updates.clear();

        AvailableUpdateBlock {
            updates: self.updates,
        }
    }

    pub fn source_depth(&self) -> usize {
        self.source_depth
    }

    pub fn source_chunk_idx(&self) -> usize {
        self.source_chunk_idx
    }

    pub fn len(&self) -> usize {
        self.updates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    pub fn updates(&self) -> &[u32] {
        &self.updates
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn chunk_idx(&self) -> usize {
        self.chunk_idx
    }
}
