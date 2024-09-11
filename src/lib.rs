#![feature(array_chunks)]
#![feature(once_cell_get_mut)]
#![deny(clippy::use_self)]
#![deny(clippy::if_not_else)]
#![deny(clippy::inconsistent_struct_constructor)]
#![deny(clippy::map_unwrap_or)]
#![deny(clippy::semicolon_if_nothing_returned)]
#![deny(clippy::similar_names)]
#![deny(clippy::needless_pass_by_value)]
#![deny(clippy::partialeq_to_none)]
#![deny(clippy::flat_map_option)]
#![deny(clippy::doc_markdown)]

pub mod bfs;
pub mod callback;
pub mod chunk_buffer_list;
pub mod io;
pub mod settings;
pub mod update;
