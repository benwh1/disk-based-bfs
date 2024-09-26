pub const NONE: u64 = u64::MAX;

pub trait BfsExpander<const EXPANSION_NODES: usize> {
    fn expand(&mut self, node: u64, expanded_nodes: &mut [u64; EXPANSION_NODES]);
}
