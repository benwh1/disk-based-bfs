//! Breadth-first search using hard drive space for storing intermediate data.
//!
//! The implementation is efficient, generic, parallel, and can use multiple hard drives. The
//! algorithm is based on the paper [Minimizing Disk I/O in Two-Bit Breadth-First Search](https://cdn.aaai.org/AAAI/2008/AAAI08-050.pdf)
//! of Richard Korf, with various improvements. It is suitable for very large implicit graphs
//! (~10^13 nodes), e.g. the 15 puzzle graph.

#![feature(buf_read_has_data_left)]
#![feature(once_cell_get_mut)]
#![deny(clippy::branches_sharing_code)]
#![deny(clippy::doc_markdown)]
#![deny(clippy::double_must_use)]
#![deny(clippy::explicit_into_iter_loop)]
#![deny(clippy::explicit_iter_loop)]
#![deny(clippy::flat_map_option)]
#![deny(clippy::if_not_else)]
#![deny(clippy::implicit_clone)]
#![deny(clippy::inconsistent_struct_constructor)]
#![deny(clippy::iter_not_returning_iterator)]
#![deny(clippy::iter_with_drain)]
#![deny(clippy::map_unwrap_or)]
#![deny(clippy::mod_module_files)]
#![deny(clippy::needless_pass_by_value)]
#![deny(clippy::partialeq_to_none)]
#![deny(clippy::redundant_clone)]
#![deny(clippy::semicolon_if_nothing_returned)]
#![deny(clippy::similar_names)]
#![deny(clippy::unused_trait_names)]
#![deny(clippy::use_self)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(clippy::must_use_candidate)]

mod bfs;
pub mod builder;
pub mod callback;
mod chunk_buffer_list;
pub mod expander;
mod io;
pub mod provider;
mod settings;
mod update;
