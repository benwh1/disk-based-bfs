use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
};

use thiserror::Error;

use crate::settings::{BfsSettings, BfsSettingsProvider};

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

    #[error("IoError: {0}")]
    IoError(#[from] std::io::Error),
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

        tracing::trace!("reading file {path:?}");

        let mut file = File::open(path)?;
        file.read_exact(buf)?;

        Ok(())
    }

    pub fn try_read_to_string<'b>(&self, path: &'b Path) -> Result<String, Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        tracing::trace!("reading file {path:?}");

        let mut file = File::open(path)?;
        let mut buf = String::new();
        file.read_to_string(&mut buf)?;

        Ok(buf)
    }

    pub fn try_read_to_vec<'b>(&self, path: &'b Path) -> Result<Vec<u8>, Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        tracing::trace!("reading file {path:?}");

        Ok(std::fs::read(path)?)
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

        tracing::trace!("writing file {path:?}");

        let path_tmp = path.with_extension("tmp");
        let mut file = File::create(&path_tmp)?;
        for data in data {
            file.write_all(data)?;
        }

        // Check that the number of bytes written is exactly the size of the file
        let file_size = file.metadata().unwrap().len();
        let data_size: u64 = data.iter().map(|buf| buf.len() as u64).sum();
        assert_eq!(file_size, data_size);

        drop(file);

        std::fs::rename(path_tmp, path)?;

        Ok(file_size)
    }

    fn try_delete_file<'b>(&self, path: &'b Path) -> Result<u64, Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        tracing::trace!("deleting file {path:?}");

        let file_len = path.metadata()?.len();

        std::fs::remove_file(path)?;

        Ok(file_len)
    }

    fn try_delete_files<'b>(&'b self, paths: &'b [&Path]) -> Result<u64, Error> {
        if paths.iter().any(|p| !self.is_on_disk(p)) {
            return Err(Error::FilesNotAllOnDisk(paths));
        }

        let _lock = self.lock();

        let mut bytes_deleted = 0;
        for path in paths {
            tracing::trace!("deleting file {path:?}");
            bytes_deleted += path.metadata()?.len();
            std::fs::remove_file(path)?;
        }

        Ok(bytes_deleted)
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
            .map_err(|e| panic!("failed to read file: {e}"))
            .unwrap();
    }

    pub fn read_to_string(&self, path: &Path) -> String {
        self.try_read_to_string(path)
            .map_err(|e| panic!("failed to read file to string: {e}"))
            .unwrap()
    }

    pub fn read_to_vec(&self, path: &Path) -> Vec<u8> {
        self.try_read_to_vec(path)
            .map_err(|e| panic!("failed to read file to vec: {e}"))
            .unwrap()
    }

    pub fn write_file(&self, path: &Path, data: &[u8]) -> u64 {
        self.try_write_file(path, data)
            .map_err(|e| panic!("failed to write file: {e}"))
            .unwrap()
    }

    pub fn write_file_multiple_buffers(&self, path: &Path, data: &[&[u8]]) -> u64 {
        self.try_write_file_multiple_buffers(path, data)
            .map_err(|e| panic!("failed to write file: {e}"))
            .unwrap()
    }
}

pub fn sync() {
    tracing::debug!("syncing filesystem");

    unsafe {
        libc::sync();
    }
}
