use std::{
    fs::File,
    io::{Error as IoError, Read, Write},
    path::{Path, PathBuf},
    string::FromUtf8Error,
    sync::{Mutex, MutexGuard},
};

use thiserror::Error;
use xxhash_rust::xxh3::Xxh3Default;

use crate::settings::{BfsSettings, BfsSettingsProvider};

#[derive(Debug, Error)]
pub enum Error {
    #[error("ReadMetadataError: failed to read metadata of file {path:?}: {err}")]
    ReadMetadataError { path: PathBuf, err: IoError },

    #[error(
        "IncorrectFileLength: unexpected length of file {path:?} (expected {expected} bytes, \
        got {actual} bytes)"
    )]
    IncorrectFileLength {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },

    #[error("CreateFileError: failed to create file {path:?}: {err}")]
    CreateFileError { path: PathBuf, err: IoError },

    #[error("OpenFileError: failed to open file {path:?}: {err}")]
    OpenFileError { path: PathBuf, err: IoError },

    #[error("ReadFileError: failed to read file {path:?}: {err}")]
    ReadFileError { path: PathBuf, err: IoError },

    #[error("WriteFileError: failed to write to file {path:?}: {err}")]
    WriteFileError { path: PathBuf, err: IoError },

    #[error("DeleteFileError: failed to delete file {path:?}: {err}")]
    DeleteFileError { path: PathBuf, err: IoError },

    #[error("RenameFileError: failed to rename file {old_path:?} to {new_path:?}: {err}")]
    RenameFileError {
        old_path: PathBuf,
        new_path: PathBuf,
        err: IoError,
    },

    #[error("StringDataError: invalid UTF-8 data in file {path:?}: {err}")]
    StringDataError { path: PathBuf, err: FromUtf8Error },

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
}

fn hash(bufs: &[&[u8]]) -> u64 {
    let mut hasher = Xxh3Default::new();
    for buf in bufs {
        hasher.update(buf);
    }
    hasher.digest()
}

fn verify_hash(path: &Path, buf: &[u8]) -> Result<(), Error> {
    let hash_path = path.with_extension("xxh3");
    let hash_file_len = hash_path
        .metadata()
        .map_err(|err| Error::ReadMetadataError {
            path: hash_path.to_owned(),
            err,
        })?
        .len();

    if hash_file_len != 8 {
        return Err(Error::IncorrectFileLength {
            path: hash_path.to_owned(),
            expected: 8,
            actual: hash_file_len,
        });
    }

    let mut hash_buf = [0u8; 8];
    let mut file = File::open(&hash_path).map_err(|err| Error::OpenFileError {
        path: hash_path.to_owned(),
        err,
    })?;
    file.read_exact(&mut hash_buf)
        .map_err(|err| Error::ReadFileError {
            path: hash_path.to_owned(),
            err,
        })?;

    drop(file);

    let read_hash = u64::from_le_bytes(hash_buf);
    let computed_hash = hash(&[buf]);

    if read_hash != computed_hash {
        return Err(Error::ChecksumMismatch {
            path: path.to_owned(),
            expected: read_hash,
            actual: computed_hash,
        });
    }

    Ok(())
}

fn write_unlocked<'a>(path: &Path, data: &'a [&[u8]], with_hash: bool) -> Result<u64, Error> {
    let data_size: u64 = data.iter().map(|buf| buf.len() as u64).sum();

    let hash = if with_hash { Some(hash(data)) } else { None };

    if let Some(hash) = hash {
        tracing::trace!("writing {data_size} bytes to {path:?}, hash = {hash:x}");
    } else {
        tracing::trace!("writing {data_size} bytes to {path:?}");
    }

    let path_tmp = path.with_extension("tmp");
    let mut file = File::create(&path_tmp).map_err(|err| Error::CreateFileError {
        path: path_tmp.to_owned(),
        err,
    })?;

    for data in data {
        file.write_all(data).map_err(|err| Error::WriteFileError {
            path: path_tmp.to_owned(),
            err,
        })?;
    }
    drop(file);

    // Check that the number of bytes written is exactly the size of the file
    let file_size = path_tmp
        .metadata()
        .map_err(|err| Error::ReadMetadataError {
            path: path_tmp.to_owned(),
            err,
        })?
        .len();

    if file_size != data_size {
        return Err(Error::IncorrectFileLength {
            path: path_tmp.to_owned(),
            expected: data_size,
            actual: file_size,
        });
    }

    if let Some(hash) = hash {
        let hash_path_tmp = path.with_extension("xxh3.tmp");
        let hash_path = path.with_extension("xxh3");

        // Write hash to file
        let mut file = File::create(&hash_path_tmp).map_err(|err| Error::CreateFileError {
            path: hash_path_tmp.to_owned(),
            err,
        })?;
        file.write_all(&hash.to_le_bytes())
            .map_err(|err| Error::WriteFileError {
                path: hash_path_tmp.to_owned(),
                err,
            })?;
        drop(file);

        std::fs::rename(&hash_path_tmp, &hash_path).map_err(|err| Error::RenameFileError {
            old_path: hash_path_tmp.to_owned(),
            new_path: hash_path.to_owned(),
            err,
        })?;
    }

    std::fs::rename(&path_tmp, &path).map_err(|err| Error::RenameFileError {
        old_path: path_tmp.to_owned(),
        new_path: path.to_owned(),
        err,
    })?;

    Ok(data_size)
}

fn read_unlocked(path: &Path, buf: &mut [u8], with_hash: bool) -> Result<(), Error> {
    let buf_len = buf.len();

    tracing::trace!("reading {buf_len} bytes from file {path:?}");

    let file_len = path
        .metadata()
        .map_err(|err| Error::ReadMetadataError {
            path: path.to_owned(),
            err,
        })?
        .len();

    if buf_len as u64 != file_len {
        return Err(Error::IncorrectFileLength {
            path: path.to_owned(),
            expected: buf_len as u64,
            actual: file_len,
        });
    }

    let mut file = File::open(path).map_err(|err| Error::OpenFileError {
        path: path.to_owned(),
        err,
    })?;
    file.read_exact(buf).map_err(|err| Error::ReadFileError {
        path: path.to_owned(),
        err,
    })?;

    if with_hash {
        verify_hash(path, buf)?;
    }

    Ok(())
}

fn read_unlocked_to_vec(path: &Path, with_hash: bool) -> Result<Vec<u8>, Error> {
    let file_len = path
        .metadata()
        .map_err(|err| Error::ReadMetadataError {
            path: path.to_owned(),
            err,
        })?
        .len();

    tracing::trace!("reading {file_len} bytes from file {path:?}");

    let buf = std::fs::read(path).map_err(|err| Error::ReadFileError {
        path: path.to_owned(),
        err,
    })?;
    let buf_len = buf.len();

    if buf_len as u64 != file_len {
        return Err(Error::IncorrectFileLength {
            path: path.to_owned(),
            expected: buf_len as u64,
            actual: file_len,
        });
    }

    if with_hash {
        verify_hash(path, &buf)?;
    }

    Ok(buf)
}

fn read_unlocked_to_string(path: &Path, with_hash: bool) -> Result<String, Error> {
    String::from_utf8(read_unlocked_to_vec(path, with_hash)?).map_err(|err| {
        Error::StringDataError {
            path: path.to_owned(),
            err,
        }
    })
}

fn delete_unlocked(paths: &[&Path], with_hash: bool) -> Result<u64, Error> {
    let mut bytes_deleted = 0;

    for &path in paths {
        let file_len = path
            .metadata()
            .map_err(|err| Error::ReadMetadataError {
                path: path.to_owned(),
                err,
            })?
            .len();

        tracing::trace!("deleting {file_len} bytes from {path:?}");

        std::fs::remove_file(path).map_err(|err| Error::DeleteFileError {
            path: path.to_owned(),
            err,
        })?;

        bytes_deleted += file_len;

        if with_hash {
            let hash_path = path.with_extension("xxh3");
            std::fs::remove_file(&hash_path).map_err(|err| Error::DeleteFileError {
                path: hash_path.to_owned(),
                err,
            })?;
        }
    }

    Ok(bytes_deleted)
}

pub struct LockedDisk<'a, P: BfsSettingsProvider> {
    settings: &'a BfsSettings<P>,
    lock: Mutex<()>,
    disk_path: PathBuf,
}

impl<'a, P: BfsSettingsProvider> LockedDisk<'a, P> {
    pub fn new(settings: &'a BfsSettings<P>, disk_path: PathBuf) -> Self {
        Self {
            settings,
            lock: Mutex::new(()),
            disk_path,
        }
    }

    fn is_on_disk(&self, path: &Path) -> bool {
        path.starts_with(&self.disk_path)
    }

    fn lock(&self) -> Option<MutexGuard<()>> {
        if self.settings.use_locked_io {
            Some(self.lock.lock().expect("failed to acquire lock"))
        } else {
            None
        }
    }

    pub fn try_read_file(&self, path: &Path, buf: &mut [u8]) -> Result<(), Error> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk {
                path: path.to_owned(),
            });
        }

        let _lock = self.lock();

        read_unlocked(path, buf, self.settings.compute_checksums)
    }

    pub fn try_read_to_string(&self, path: &Path) -> Result<String, Error> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk {
                path: path.to_owned(),
            });
        }

        let _lock = self.lock();

        read_unlocked_to_string(path, self.settings.compute_checksums)
    }

    pub fn try_read_to_vec(&self, path: &Path) -> Result<Vec<u8>, Error> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk {
                path: path.to_owned(),
            });
        }

        let _lock = self.lock();

        read_unlocked_to_vec(path, self.settings.compute_checksums)
    }

    pub fn try_write_file(&self, path: &Path, data: &[u8]) -> Result<u64, Error> {
        self.try_write_file_multiple_buffers(path, &[data])
    }

    pub fn try_write_file_multiple_buffers(
        &self,
        path: &Path,
        data: &[&[u8]],
    ) -> Result<u64, Error> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk {
                path: path.to_owned(),
            });
        }

        let _lock = self.lock();

        write_unlocked(path, data, self.settings.compute_checksums)
    }

    fn try_delete_file(&self, path: &Path) -> Result<u64, Error> {
        self.try_delete_files(&[path])
    }

    fn try_delete_files(&self, paths: &[&Path]) -> Result<u64, Error> {
        if paths.iter().any(|path| !self.is_on_disk(path)) {
            return Err(Error::FilesNotAllOnDisk {
                paths: paths.iter().map(|path| (*path).to_owned()).collect(),
            });
        }

        let _lock = self.lock();

        delete_unlocked(paths, self.settings.compute_checksums)
    }
}

pub struct LockedIO<'a, P: BfsSettingsProvider> {
    disks: Vec<LockedDisk<'a, P>>,
}

impl<'a, P: BfsSettingsProvider> LockedIO<'a, P> {
    pub fn new(settings: &'a BfsSettings<P>, disk_paths: Vec<PathBuf>) -> Self {
        let disks = disk_paths
            .into_iter()
            .map(|disk_path| LockedDisk::new(settings, disk_path))
            .collect();

        Self { disks }
    }

    pub fn num_disks(&self) -> usize {
        self.disks.len()
    }

    pub fn try_read_file(&self, path: &Path, buf: &mut [u8]) -> Result<(), Error> {
        for disk in &self.disks {
            let result = disk.try_read_file(path, buf);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    pub fn try_read_to_string(&self, path: &Path) -> Result<String, Error> {
        for disk in &self.disks {
            let result = disk.try_read_to_string(path);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    pub fn try_read_to_vec(&self, path: &Path) -> Result<Vec<u8>, Error> {
        for disk in &self.disks {
            let result = disk.try_read_to_vec(path);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    pub fn try_write_file(&self, path: &Path, data: &[u8]) -> Result<u64, Error> {
        for disk in &self.disks {
            let result = disk.try_write_file(path, data);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    pub fn try_write_file_multiple_buffers(
        &self,
        path: &Path,
        data: &[&[u8]],
    ) -> Result<u64, Error> {
        for disk in &self.disks {
            let result = disk.try_write_file_multiple_buffers(path, data);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    pub fn try_delete_file(&self, path: &Path) -> Result<u64, Error> {
        for disk in &self.disks {
            let result = disk.try_delete_file(path);
            match result {
                Err(Error::FileNotOnDisk { .. }) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk {
            path: path.to_owned(),
        })
    }

    pub fn try_delete_files(&self, paths: &[&Path]) -> Result<u64, Error> {
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

    pub fn read_file(&self, path: &Path, buf: &mut [u8]) {
        self.try_read_file(path, buf)
            .inspect_err(|e| panic!("{e}"))
            .unwrap();
    }

    pub fn read_to_string(&self, path: &Path) -> String {
        self.try_read_to_string(path)
            .inspect_err(|e| panic!("{e}"))
            .unwrap()
    }

    pub fn read_to_vec(&self, path: &Path) -> Vec<u8> {
        self.try_read_to_vec(path)
            .inspect_err(|e| panic!("{e}"))
            .unwrap()
    }

    pub fn write_file(&self, path: &Path, data: &[u8]) -> u64 {
        self.try_write_file(path, data)
            .inspect_err(|e| panic!("{e}"))
            .unwrap()
    }

    pub fn write_file_multiple_buffers(&self, path: &Path, data: &[&[u8]]) -> u64 {
        self.try_write_file_multiple_buffers(path, data)
            .inspect_err(|e| panic!("{e}"))
            .unwrap()
    }
}

pub fn sync() {
    tracing::debug!("syncing filesystem");

    unsafe {
        libc::sync();
    }
}
