use std::{
    fs::File,
    io::{BufRead as _, Cursor, Error as IoError, ErrorKind, Read as _, Write as _},
    path::{Path, PathBuf},
    string::FromUtf8Error,
};

use parking_lot::Mutex;
use thiserror::Error;
use xxhash_rust::xxh3::Xxh3Default;
use zstd::{Decoder, Encoder};

use crate::{
    callback::BfsCallback, expander::BfsExpander, provider::BfsSettingsProvider,
    settings::BfsSettings,
};

#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error("ReadMetadata: failed to read metadata of file {path:?}: {err}")]
    ReadMetadata { path: PathBuf, err: IoError },

    #[error(
        "IncorrectFileLength: unexpected length of file {path:?} (expected {expected} bytes, \
        got {actual} bytes)"
    )]
    IncorrectFileLength {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },

    #[error("CreateFile: failed to create file {path:?}: {err}")]
    CreateFile { path: PathBuf, err: IoError },

    #[error("OpenFile: failed to open file {path:?}: {err}")]
    OpenFile { path: PathBuf, err: IoError },

    #[error("ReadFile: failed to read file {path:?}: {err}")]
    ReadFile { path: PathBuf, err: IoError },

    #[error("WriteFile: failed to write to file {path:?}: {err}")]
    WriteFile { path: PathBuf, err: IoError },

    #[error("DeleteFile: failed to delete file {path:?}: {err}")]
    DeleteFile { path: PathBuf, err: IoError },

    #[error("RenameFile: failed to rename file {old_path:?} to {new_path:?}: {err}")]
    RenameFile {
        old_path: PathBuf,
        new_path: PathBuf,
        err: IoError,
    },

    #[error("StringData: invalid UTF-8 data in file {path:?}: {err}")]
    StringData { path: PathBuf, err: FromUtf8Error },

    #[error(
        "ChecksumMismatch: checksum mismatch for file {path:?} (expected {expected:x}, got \
        {actual:x})"
    )]
    ChecksumMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },

    #[error("FileNotOnDisk: path {path:?} not on disk")]
    FileNotOnDisk { path: PathBuf },

    #[error("FileNotOnAnyDisk: path {path:?} not on any disk")]
    FileNotOnAnyDisk { path: PathBuf },

    #[error("FilesNotAllOnDisk: files {paths:?} not on disk")]
    FilesNotAllOnDisk { paths: Vec<PathBuf> },

    #[error("FilesNotOnSameDisk: files {paths:?} not on same disk")]
    FilesNotOnSameDisk { paths: Vec<PathBuf> },

    #[error("Compression: failed to write compressed data to file {path:?}: {err}")]
    Compression { path: PathBuf, err: IoError },

    #[error("Decompression: failed to read compressed data from file {path:?}: {err}")]
    Decompression { path: PathBuf, err: IoError },

    #[error("IncompleteDecompression: file {path:?} still has data left after decompression")]
    IncompleteDecompression { path: PathBuf },
}

impl Error {
    /// Helper to check whether an error is due to a non-existent file being read
    pub(crate) fn is_read_nonexistent_file_error(&self) -> bool {
        matches!(
            self,
            Self::ReadFile { err, .. } | Self::ReadMetadata { err, .. }
                if err.kind() == ErrorKind::NotFound
        )
    }
}

fn hash(bufs: &[&[u8]]) -> u64 {
    let mut hasher = Xxh3Default::new();
    for buf in bufs {
        hasher.update(buf);
    }
    hasher.digest()
}

fn write_compressed(
    disk_mutex: Option<&Mutex<()>>,
    path: &Path,
    data: &[&[u8]],
    with_hash: bool,
) -> Result<u64, Error> {
    let data_size: u64 = data.iter().map(|buf| buf.len() as u64).sum();
    let hash = if with_hash { Some(hash(data)) } else { None };

    tracing::trace!("writing {path:?}");

    let path_tmp = path.with_extension("tmp");

    let mut compressed_bytes = Vec::new();
    let mut encoder = Encoder::new(&mut compressed_bytes, 1).unwrap();
    for data in data {
        encoder.write_all(bytemuck::cast_slice(data)).unwrap();
    }
    encoder.finish().unwrap();

    let lock = disk_mutex.map(|m| m.lock());

    let mut file = File::create(&path_tmp).map_err(|err| Error::CreateFile {
        path: path_tmp.clone(),
        err,
    })?;

    let mut encoder = Encoder::new(&mut file, 1).unwrap();
    for data in data {
        encoder.write_all(data).map_err(|err| Error::Compression {
            path: path_tmp.clone(),
            err,
        })?;
    }
    encoder.finish().map_err(|err| Error::Compression {
        path: path_tmp.clone(),
        err,
    })?;

    if let Some(hash) = hash {
        file.write_all(&hash.to_le_bytes())
            .map_err(|err| Error::WriteFile {
                path: path_tmp.clone(),
                err,
            })?;
    }

    drop(file);

    let file_size = path_tmp
        .metadata()
        .map_err(|err| Error::ReadMetadata {
            path: path_tmp.clone(),
            err,
        })?
        .len();

    std::fs::rename(&path_tmp, path).map_err(|err| Error::RenameFile {
        old_path: path_tmp.clone(),
        new_path: path.to_owned(),
        err,
    })?;

    let bytes_to_write_uncompressed = if with_hash { data_size + 8 } else { data_size };

    tracing::trace!(
        "wrote {file_size} bytes ({bytes_to_write_uncompressed} bytes uncompressed) to {path:?}"
    );

    drop(lock);

    Ok(file_size)
}

fn write_uncompressed(
    disk_mutex: Option<&Mutex<()>>,
    path: &Path,
    data: &[&[u8]],
    with_hash: bool,
) -> Result<u64, Error> {
    let data_size: u64 = data.iter().map(|buf| buf.len() as u64).sum();
    let hash = if with_hash { Some(hash(data)) } else { None };

    tracing::trace!("writing {path:?}");

    let path_tmp = path.with_extension("tmp");

    let lock = disk_mutex.map(|m| m.lock());

    let mut file = File::create(&path_tmp).map_err(|err| Error::CreateFile {
        path: path_tmp.clone(),
        err,
    })?;

    for data in data {
        file.write_all(data).map_err(|err| Error::WriteFile {
            path: path_tmp.clone(),
            err,
        })?;
    }

    if let Some(hash) = hash {
        file.write_all(&hash.to_le_bytes())
            .map_err(|err| Error::WriteFile {
                path: path_tmp.clone(),
                err,
            })?;
    }

    drop(file);

    let file_size = path_tmp
        .metadata()
        .map_err(|err| Error::ReadMetadata {
            path: path_tmp.clone(),
            err,
        })?
        .len();

    let bytes_to_write_uncompressed = if with_hash { data_size + 8 } else { data_size };

    // Check that the number of bytes written is exactly the size of the file
    if file_size != bytes_to_write_uncompressed {
        return Err(Error::IncorrectFileLength {
            path: path_tmp,
            expected: bytes_to_write_uncompressed,
            actual: file_size,
        });
    }

    std::fs::rename(&path_tmp, path).map_err(|err| Error::RenameFile {
        old_path: path_tmp.clone(),
        new_path: path.to_owned(),
        err,
    })?;

    tracing::trace!("wrote {file_size} bytes to {path:?}");

    drop(lock);

    Ok(file_size)
}

fn write(
    disk_mutex: Option<&Mutex<()>>,
    path: &Path,
    data: &[&[u8]],
    with_hash: bool,
    compressed: bool,
) -> Result<u64, Error> {
    if compressed {
        write_compressed(disk_mutex, path, data, with_hash)
    } else {
        write_uncompressed(disk_mutex, path, data, with_hash)
    }
}

fn read_uncompressed_to_buf(
    disk_mutex: Option<&Mutex<()>>,
    path: &Path,
    buf: &mut [u8],
    with_hash: bool,
) -> Result<(), Error> {
    let buf_len = buf.len() as u64;

    tracing::trace!("reading file {path:?}");

    let lock = disk_mutex.map(|m| m.lock());

    let file_size = path
        .metadata()
        .map_err(|err| Error::ReadMetadata {
            path: path.to_owned(),
            err,
        })?
        .len();

    let expected_file_size = if with_hash { buf_len + 8 } else { buf_len };

    if file_size != expected_file_size {
        return Err(Error::IncorrectFileLength {
            path: path.to_owned(),
            expected: expected_file_size,
            actual: file_size,
        });
    }

    let mut file = File::open(path).map_err(|err| Error::OpenFile {
        path: path.to_owned(),
        err,
    })?;
    file.read_exact(buf).map_err(|err| Error::ReadFile {
        path: path.to_owned(),
        err,
    })?;

    if with_hash {
        let mut hash_buf = [0u8; 8];
        file.read_exact(&mut hash_buf)
            .map_err(|err| Error::ReadFile {
                path: path.to_owned(),
                err,
            })?;

        drop(lock);

        let read_hash = u64::from_le_bytes(hash_buf);
        let computed_hash = hash(&[buf]);

        if read_hash != computed_hash {
            return Err(Error::ChecksumMismatch {
                path: path.to_owned(),
                expected: read_hash,
                actual: computed_hash,
            });
        }
    } else {
        drop(lock);
    }

    tracing::trace!("read {file_size} bytes from file {path:?}");

    Ok(())
}

fn read_compressed_to_buf(
    disk_mutex: Option<&Mutex<()>>,
    path: &Path,
    buf: &mut [u8],
    with_hash: bool,
) -> Result<(), Error> {
    let lock = disk_mutex.map(|m| m.lock());

    let file_len = path
        .metadata()
        .map_err(|err| Error::ReadMetadata {
            path: path.to_owned(),
            err,
        })?
        .len();

    tracing::trace!("reading file {path:?}");

    let mut bytes = std::fs::read(path).map_err(|err| Error::ReadFile {
        path: path.to_owned(),
        err,
    })?;

    drop(lock);

    let bytes_len = bytes.len();

    if bytes_len as u64 != file_len {
        return Err(Error::IncorrectFileLength {
            path: path.to_owned(),
            expected: bytes_len as u64,
            actual: file_len,
        });
    }

    let read_hash = if with_hash {
        let hash_buf: [u8; 8] = bytes.split_off(bytes_len - 8).try_into().unwrap();
        let hash = u64::from_le_bytes(hash_buf);
        Some(hash)
    } else {
        None
    };

    let mut decoder = Decoder::new(Cursor::new(&bytes)).unwrap();
    decoder
        .read_exact(buf)
        .map_err(|err| Error::Decompression {
            path: path.to_owned(),
            err,
        })?;

    if decoder.finish().has_data_left().unwrap_or(true) {
        return Err(Error::IncompleteDecompression {
            path: path.to_owned(),
        });
    }

    let decompressed_len = if with_hash { buf.len() + 8 } else { buf.len() };
    tracing::trace!(
        "read {bytes_len} bytes ({decompressed_len} bytes uncompressed) from file {path:?}"
    );

    if let Some(read_hash) = read_hash {
        let computed_hash = hash(&[buf]);

        if read_hash != computed_hash {
            return Err(Error::ChecksumMismatch {
                path: path.to_owned(),
                expected: read_hash,
                actual: computed_hash,
            });
        }
    }

    Ok(())
}

fn read_to_buf(
    disk_mutex: Option<&Mutex<()>>,
    path: &Path,
    buf: &mut [u8],
    with_hash: bool,
    compressed: bool,
) -> Result<(), Error> {
    if compressed {
        read_compressed_to_buf(disk_mutex, path, buf, with_hash)
    } else {
        read_uncompressed_to_buf(disk_mutex, path, buf, with_hash)
    }
}

fn read_to_vec(
    disk_mutex: Option<&Mutex<()>>,
    path: &Path,
    with_hash: bool,
    compressed: bool,
) -> Result<Vec<u8>, Error> {
    let lock = disk_mutex.map(|m| m.lock());

    let file_len = path
        .metadata()
        .map_err(|err| Error::ReadMetadata {
            path: path.to_owned(),
            err,
        })?
        .len();

    tracing::trace!("reading file {path:?}");

    let mut bytes = std::fs::read(path).map_err(|err| Error::ReadFile {
        path: path.to_owned(),
        err,
    })?;

    drop(lock);

    let bytes_len = bytes.len();

    if bytes_len as u64 != file_len {
        return Err(Error::IncorrectFileLength {
            path: path.to_owned(),
            expected: bytes_len as u64,
            actual: file_len,
        });
    }

    let read_hash = if with_hash {
        let hash_buf: [u8; 8] = bytes.split_off(bytes_len - 8).try_into().unwrap();
        let hash = u64::from_le_bytes(hash_buf);
        Some(hash)
    } else {
        None
    };

    if compressed {
        let mut decoder = Decoder::new(Cursor::new(&bytes)).unwrap();
        let mut decompressed_buf = Vec::new();
        decoder
            .read_to_end(&mut decompressed_buf)
            .map_err(|err| Error::Decompression {
                path: path.to_owned(),
                err,
            })?;
        bytes = decompressed_buf;

        let decompressed_len = if with_hash {
            bytes.len() + 8
        } else {
            bytes.len()
        };

        tracing::trace!(
            "read {bytes_len} bytes ({decompressed_len} bytes uncompressed) from file {path:?}"
        );
    } else {
        tracing::trace!("read {bytes_len} bytes from file {path:?}");
    }

    if let Some(read_hash) = read_hash {
        let computed_hash = hash(&[&bytes]);

        if read_hash != computed_hash {
            return Err(Error::ChecksumMismatch {
                path: path.to_owned(),
                expected: read_hash,
                actual: computed_hash,
            });
        }
    }

    Ok(bytes)
}

fn read_to_string(
    disk_mutex: Option<&Mutex<()>>,
    path: &Path,
    with_hash: bool,
    compressed: bool,
) -> Result<String, Error> {
    String::from_utf8(read_to_vec(disk_mutex, path, with_hash, compressed)?).map_err(|err| {
        Error::StringData {
            path: path.to_owned(),
            err,
        }
    })
}

fn delete(disk_mutex: Option<&Mutex<()>>, paths: &[&Path]) -> Result<u64, Error> {
    let lock = disk_mutex.map(|m| m.lock());

    let mut bytes_deleted = 0;

    for &path in paths {
        let file_len = path
            .metadata()
            .map_err(|err| Error::ReadMetadata {
                path: path.to_owned(),
                err,
            })?
            .len();

        tracing::trace!("deleting {file_len} bytes from {path:?}");

        std::fs::remove_file(path).map_err(|err| Error::DeleteFile {
            path: path.to_owned(),
            err,
        })?;

        bytes_deleted += file_len;
    }

    drop(lock);

    Ok(bytes_deleted)
}

struct LockedDisk<'a, Expander, Callback, Provider, const EXPANSION_NODES: usize> {
    settings: &'a BfsSettings<Expander, Callback, Provider, EXPANSION_NODES>,
    lock: Mutex<()>,
    disk_path: PathBuf,
}

impl<'a, Expander, Callback, Provider, const EXPANSION_NODES: usize>
    LockedDisk<'a, Expander, Callback, Provider, EXPANSION_NODES>
{
    fn new(
        settings: &'a BfsSettings<Expander, Callback, Provider, EXPANSION_NODES>,
        disk_path: PathBuf,
    ) -> Self {
        Self {
            settings,
            lock: Mutex::new(()),
            disk_path,
        }
    }

    fn is_on_disk(&self, path: &Path) -> bool {
        path.starts_with(&self.disk_path)
    }

    fn mutex(&self) -> Option<&Mutex<()>> {
        self.settings.use_locked_io.then_some(&self.lock)
    }

    fn try_read_file(&self, path: &Path, buf: &mut [u8], compressed: bool) -> Result<(), Error> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk {
                path: path.to_owned(),
            });
        }

        read_to_buf(
            self.mutex(),
            path,
            buf,
            self.settings.compute_checksums,
            compressed,
        )
    }

    fn try_read_to_string(&self, path: &Path, compressed: bool) -> Result<String, Error> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk {
                path: path.to_owned(),
            });
        }

        read_to_string(
            self.mutex(),
            path,
            self.settings.compute_checksums,
            compressed,
        )
    }

    fn try_read_to_vec(&self, path: &Path, compressed: bool) -> Result<Vec<u8>, Error> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk {
                path: path.to_owned(),
            });
        }

        read_to_vec(
            self.mutex(),
            path,
            self.settings.compute_checksums,
            compressed,
        )
    }

    fn try_write_file_multiple_buffers(
        &self,
        path: &Path,
        data: &[&[u8]],
        compressed: bool,
    ) -> Result<u64, Error> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk {
                path: path.to_owned(),
            });
        }

        write(
            self.mutex(),
            path,
            data,
            self.settings.compute_checksums,
            compressed,
        )
    }

    fn try_delete_files(&self, paths: &[&Path]) -> Result<u64, Error> {
        if paths.iter().any(|path| !self.is_on_disk(path)) {
            return Err(Error::FilesNotAllOnDisk {
                paths: paths.iter().map(|path| (*path).to_owned()).collect(),
            });
        }

        delete(self.mutex(), paths)
    }
}

pub(crate) struct LockedIO<'a, Expander, Callback, Provider, const EXPANSION_NODES: usize> {
    disks: Vec<LockedDisk<'a, Expander, Callback, Provider, EXPANSION_NODES>>,
}

impl<'a, Expander, Callback, Provider, const EXPANSION_NODES: usize>
    LockedIO<'a, Expander, Callback, Provider, EXPANSION_NODES>
where
    Expander: BfsExpander<EXPANSION_NODES> + Clone + Sync,
    Callback: BfsCallback + Clone + Sync,
    Provider: BfsSettingsProvider + Sync,
{
    pub(crate) fn new(
        settings: &'a BfsSettings<Expander, Callback, Provider, EXPANSION_NODES>,
    ) -> Self {
        let disks = settings
            .root_directories
            .iter()
            .map(|disk_path| LockedDisk::new(settings, disk_path.to_owned()))
            .collect();

        Self { disks }
    }

    pub(crate) fn num_disks(&self) -> usize {
        self.disks.len()
    }

    pub(crate) fn try_read_file(
        &self,
        path: &Path,
        buf: &mut [u8],
        compressed: bool,
    ) -> Result<(), Error> {
        for disk in &self.disks {
            let result = disk.try_read_file(path, buf, compressed);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    pub(crate) fn try_read_to_string(
        &self,
        path: &Path,
        compressed: bool,
    ) -> Result<String, Error> {
        for disk in &self.disks {
            let result = disk.try_read_to_string(path, compressed);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    fn try_read_to_vec(&self, path: &Path, compressed: bool) -> Result<Vec<u8>, Error> {
        for disk in &self.disks {
            let result = disk.try_read_to_vec(path, compressed);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    fn try_write_file(&self, path: &Path, data: &[u8], compressed: bool) -> Result<u64, Error> {
        self.try_write_file_multiple_buffers(path, &[data], compressed)
    }

    fn try_write_file_multiple_buffers(
        &self,
        path: &Path,
        data: &[&[u8]],
        compressed: bool,
    ) -> Result<u64, Error> {
        for disk in &self.disks {
            let result = disk.try_write_file_multiple_buffers(path, data, compressed);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    pub(crate) fn try_delete_file(&self, path: &Path) -> Result<u64, Error> {
        self.try_delete_files(&[path])
    }

    pub(crate) fn try_delete_files(&self, paths: &[&Path]) -> Result<u64, Error> {
        for disk in &self.disks {
            let result = disk.try_delete_files(paths);
            match result {
                Err(Error::FilesNotAllOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FilesNotOnSameDisk {
            paths: paths.iter().map(|path| (*path).to_owned()).collect(),
        })
    }

    pub(crate) fn read_file(&self, path: &Path, buf: &mut [u8], compressed: bool) {
        self.try_read_file(path, buf, compressed)
            .inspect_err(|e| panic!("{e}"))
            .unwrap();
    }

    pub(crate) fn read_to_vec(&self, path: &Path, compressed: bool) -> Vec<u8> {
        self.try_read_to_vec(path, compressed)
            .inspect_err(|e| panic!("{e}"))
            .unwrap()
    }

    pub(crate) fn write_file(&self, path: &Path, data: &[u8], compressed: bool) -> u64 {
        self.try_write_file(path, data, compressed)
            .inspect_err(|e| panic!("{e}"))
            .unwrap()
    }

    pub(crate) fn write_file_multiple_buffers(
        &self,
        path: &Path,
        data: &[&[u8]],
        compressed: bool,
    ) -> u64 {
        self.try_write_file_multiple_buffers(path, data, compressed)
            .inspect_err(|e| panic!("{e}"))
            .unwrap()
    }
}

pub(crate) fn sync() {
    tracing::debug!("syncing filesystem");

    unsafe {
        libc::sync();
    }
}
