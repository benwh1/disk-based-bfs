use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
};

use thiserror::Error;
use xxhash_rust::xxh3::Xxh3Default;

use crate::settings::{BfsSettings, BfsSettingsProvider};

fn hash(bufs: &[&[u8]]) -> u64 {
    let mut hasher = Xxh3Default::new();
    for buf in bufs {
        hasher.update(buf);
    }
    hasher.digest()
}

fn verify_hash(path: &Path, buf: &[u8]) {
    let hash_path = path.with_extension("xxh3");
    let hash_file_len = hash_path
        .metadata()
        .inspect_err(|_| panic!("failed to get metadata of hash file {hash_path:?}"))
        .unwrap()
        .len();
    assert_eq!(
        hash_file_len, 8,
        "unexpected length of hash file {path:?} (expected 8 bytes, got {hash_file_len} bytes)",
    );

    let mut hash_buf = [0u8; 8];
    let mut file = File::open(&hash_path)
        .inspect_err(|_| panic!("failed to open hash file {hash_path:?}"))
        .unwrap();
    file.read_exact(&mut hash_buf)
        .inspect_err(|_| panic!("failed to read hash file {hash_path:?}"))
        .unwrap();
    drop(file);

    let read_hash = u64::from_le_bytes(hash_buf);
    let computed_hash = hash(&[buf]);

    assert_eq!(
        read_hash, computed_hash,
        "hash mismatch for file {path:?} (expected {read_hash:x}, got {computed_hash:x})",
    );
}

fn write_unlocked<'a>(path: &Path, data: &'a [&[u8]], with_hash: bool) -> u64 {
    let data_size: u64 = data.iter().map(|buf| buf.len() as u64).sum();

    let hash = if with_hash { Some(hash(data)) } else { None };

    if let Some(hash) = hash {
        tracing::trace!("writing {data_size} bytes to {path:?}, hash = {hash:x}");
    } else {
        tracing::trace!("writing {data_size} bytes to {path:?}");
    }

    let path_tmp = path.with_extension("tmp");
    let mut file = File::create(&path_tmp)
        .inspect_err(|_| panic!("failed to create file {path_tmp:?}"))
        .unwrap();
    for data in data {
        file.write_all(data)
            .inspect_err(|_| panic!("failed to write to file {path_tmp:?}"))
            .unwrap();
    }
    drop(file);

    // Check that the number of bytes written is exactly the size of the file
    let file_size = path.metadata().unwrap().len();
    assert_eq!(file_size, data_size);

    if let Some(hash) = hash {
        let hash_path_tmp = path.with_extension("xxh3.tmp");
        let hash_path = path.with_extension("xxh3");

        // Write hash to file
        let mut file = File::create(&hash_path_tmp)
            .inspect_err(|_| panic!("failed to create hash file {hash_path_tmp:?}"))
            .unwrap();
        file.write_all(&hash.to_le_bytes())
            .inspect_err(|_| panic!("failed to write to hash file {hash_path_tmp:?}"))
            .unwrap();
        drop(file);

        std::fs::rename(&hash_path_tmp, hash_path)
            .inspect_err(|_| panic!("failed to rename tmp hash file {hash_path_tmp:?}"))
            .unwrap();
    }

    std::fs::rename(&path_tmp, path)
        .inspect_err(|_| panic!("failed to rename tmp file {path_tmp:?}"))
        .unwrap();

    data_size
}

fn read_unlocked(path: &Path, buf: &mut [u8], with_hash: bool) {
    let file_len = path
        .metadata()
        .inspect_err(|_| panic!("failed to get metadata of file {path:?}"))
        .unwrap()
        .len();
    let buf_len = buf.len();
    assert_eq!(
        buf_len, file_len as usize,
        "unexpected length of file {path:?} (expected {buf_len} bytes, got {file_len} bytes)",
    );

    tracing::trace!("reading {file_len} bytes from file {path:?}");

    let mut file = File::open(path)
        .inspect_err(|_| panic!("failed to open file {path:?}"))
        .unwrap();
    file.read_exact(buf)
        .inspect_err(|_| panic!("failed to read file {path:?}"))
        .unwrap();

    if with_hash {
        verify_hash(path, buf);
    }
}

fn read_unlocked_to_vec(path: &Path, with_hash: bool) -> Vec<u8> {
    let file_len = path
        .metadata()
        .inspect_err(|_| panic!("failed to get metadata of file {path:?}"))
        .unwrap()
        .len();

    tracing::trace!("reading {file_len} bytes from file {path:?}");

    let buf = std::fs::read(path)
        .inspect_err(|_| panic!("failed to read file {path:?}"))
        .unwrap();
    let buf_len = buf.len();

    assert_eq!(
        buf_len, file_len as usize,
        "unexpected length of file {path:?} (expected {file_len} bytes, got {buf_len} bytes)",
    );

    if with_hash {
        verify_hash(path, &buf);
    }

    buf
}

fn read_unlocked_to_string(path: &Path, with_hash: bool) -> String {
    String::from_utf8(read_unlocked_to_vec(path, with_hash))
        .inspect_err(|e| panic!("failed to read file {path:?} to string: {e}"))
        .unwrap()
}

fn delete_unlocked(paths: &[&Path], with_hash: bool) -> u64 {
    let mut bytes_deleted = 0;

    for path in paths {
        let file_len = path
            .metadata()
            .inspect_err(|_| panic!("failed to get metadata of file {path:?}"))
            .unwrap()
            .len();

        tracing::trace!("deleting {file_len} bytes from {path:?}");

        std::fs::remove_file(path)
            .inspect_err(|_| panic!("failed to delete file {path:?}"))
            .unwrap();

        bytes_deleted += file_len;

        if with_hash {
            let hash_path = path.with_extension("xxh3");
            std::fs::remove_file(&hash_path)
                .inspect_err(|_| panic!("failed to delete hash file {hash_path:?}"))
                .unwrap();
        }
    }

    bytes_deleted
}

#[derive(Debug, Error)]
pub enum Error<'a> {
    #[error("FileNotOnDisk: path {0:?} not on disk")]
    FileNotOnDisk(&'a Path),

    #[error("FileNotOnAnyDisk: path {0:?} not on any disk")]
    FileNotOnAnyDisk(&'a Path),

    #[error("FilesNotAllOnDisk: files {0:?} not on disk")]
    FilesNotAllOnDisk(&'a [&'a Path]),

    #[error("FilesNotOnSameDisk: files {0:?} not on same disk")]
    FilesNotOnSameDisk(&'a [&'a Path]),
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

    pub fn try_read_file<'b>(&self, path: &'b Path, buf: &mut [u8]) -> Result<(), Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        read_unlocked(path, buf, self.settings.compute_checksums);

        Ok(())
    }

    pub fn try_read_to_string<'b>(&self, path: &'b Path) -> Result<String, Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        Ok(read_unlocked_to_string(
            path,
            self.settings.compute_checksums,
        ))
    }

    pub fn try_read_to_vec<'b>(&self, path: &'b Path) -> Result<Vec<u8>, Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        Ok(read_unlocked_to_vec(path, self.settings.compute_checksums))
    }

    pub fn try_write_file<'b>(&self, path: &'b Path, data: &[u8]) -> Result<u64, Error<'b>> {
        self.try_write_file_multiple_buffers(path, &[data])
    }

    pub fn try_write_file_multiple_buffers<'b>(
        &self,
        path: &'b Path,
        data: &[&[u8]],
    ) -> Result<u64, Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        Ok(write_unlocked(path, data, self.settings.compute_checksums))
    }

    fn try_delete_file<'b>(&self, path: &'b Path) -> Result<u64, Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        Ok(delete_unlocked(&[path], self.settings.compute_checksums))
    }

    fn try_delete_files<'b>(&'b self, paths: &'b [&Path]) -> Result<u64, Error> {
        if paths.iter().any(|p| !self.is_on_disk(p)) {
            return Err(Error::FilesNotAllOnDisk(paths));
        }

        let _lock = self.lock();

        Ok(delete_unlocked(paths, self.settings.compute_checksums))
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

    pub fn try_read_file<'b>(&self, path: &'b Path, buf: &mut [u8]) -> Result<(), Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_read_file(path, buf);
            match result {
                Err(Error::FileNotOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk(path))
    }

    pub fn try_read_to_string<'b>(&self, path: &'b Path) -> Result<String, Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_read_to_string(path);
            match result {
                Err(Error::FileNotOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk(path))
    }

    pub fn try_read_to_vec<'b>(&self, path: &'b Path) -> Result<Vec<u8>, Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_read_to_vec(path);
            match result {
                Err(Error::FileNotOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk(path))
    }

    pub fn try_write_file<'b>(&self, path: &'b Path, data: &[u8]) -> Result<u64, Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_write_file(path, data);
            match result {
                Err(Error::FileNotOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk(path))
    }

    pub fn try_write_file_multiple_buffers<'b>(
        &self,
        path: &'b Path,
        data: &[&[u8]],
    ) -> Result<u64, Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_write_file_multiple_buffers(path, data);
            match result {
                Err(Error::FileNotOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk(path))
    }

    pub fn try_delete_file<'b>(&self, path: &'b Path) -> Result<u64, Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_delete_file(path);
            match result {
                Err(Error::FileNotOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk(path))
    }

    pub fn try_delete_files<'b>(&'b self, paths: &'b [&Path]) -> Result<u64, Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_delete_files(paths);
            match result {
                Err(Error::FilesNotAllOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FilesNotOnSameDisk(paths))
    }

    pub fn read_file(&self, path: &Path, buf: &mut [u8]) {
        self.try_read_file(path, buf)
            .inspect_err(|e| panic!("failed to read file: {e}"))
            .unwrap();
    }

    pub fn read_to_string(&self, path: &Path) -> String {
        self.try_read_to_string(path)
            .inspect_err(|e| panic!("failed to read file to string: {e}"))
            .unwrap()
    }

    pub fn read_to_vec(&self, path: &Path) -> Vec<u8> {
        self.try_read_to_vec(path)
            .inspect_err(|e| panic!("failed to read file to vec: {e}"))
            .unwrap()
    }

    pub fn write_file(&self, path: &Path, data: &[u8]) -> u64 {
        self.try_write_file(path, data)
            .inspect_err(|e| panic!("failed to write file: {e}"))
            .unwrap()
    }

    pub fn write_file_multiple_buffers(&self, path: &Path, data: &[&[u8]]) -> u64 {
        self.try_write_file_multiple_buffers(path, data)
            .inspect_err(|e| panic!("failed to write file: {e}"))
            .unwrap()
    }
}

pub fn sync() {
    tracing::debug!("syncing filesystem");

    unsafe {
        libc::sync();
    }
}
