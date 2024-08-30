#![feature(array_chunks)]
#![deny(clippy::use_self)]
#![deny(clippy::if_not_else)]
#![deny(clippy::inconsistent_struct_constructor)]
#![deny(clippy::map_unwrap_or)]
#![deny(clippy::semicolon_if_nothing_returned)]

pub mod bfs;
pub mod callback;
pub mod chunk_allocator;
pub mod chunk_buffer_list;
pub mod io;
pub mod settings;
pub mod update;
