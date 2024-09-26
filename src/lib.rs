#![feature(buf_read_has_data_left)]
#![feature(once_cell_get_mut)]
#![warn(clippy::must_use_candidate)]
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
#![deny(clippy::double_must_use)]
#![deny(clippy::iter_not_returning_iterator)]
#![deny(clippy::mod_module_files)]
#![deny(clippy::explicit_iter_loop)]
#![deny(clippy::implicit_clone)]
#![deny(clippy::iter_with_drain)]
#![deny(clippy::branches_sharing_code)]
#![deny(clippy::redundant_clone)]

mod bfs;
pub mod builder;
pub mod callback;
mod chunk_buffer_list;
pub mod expander;
mod io;
pub mod provider;
mod settings;
mod update;
