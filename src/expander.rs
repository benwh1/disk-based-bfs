//! Defines the `BfsExpander` trait.

/// A value that represents no node.
pub const NONE: u64 = u64::MAX;

/// Defines the graph that the BFS will traverse.
pub trait BfsExpander<const EXPANSION_NODES: usize> {
    /// Given a node `node` of the graph, populates `expanded_nodes` with the adjacent nodes in the
    /// graph.
    fn expand(&mut self, node: u64, expanded_nodes: &mut [u64; EXPANSION_NODES]);
}
