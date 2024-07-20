use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
};

use thiserror::Error;

use crate::settings::BfsSettings;

#[derive(Debug, Error)]
pub enum Error<'a> {
    #[error("FileNotOnDisk: path {0:?} not on disk")]
    FileNotOnDisk(&'a Path),

    #[error("FileNotOnAnyDisk: path {0:?} not on any disk")]
    FileNotOnAnyDisk(&'a Path),

    #[error("IoError: {0}")]
    IoError(#[from] std::io::Error),
}

pub struct LockedDisk<'a> {
    settings: &'a BfsSettings,
    lock: Mutex<()>,
    disk_path: PathBuf,
}

impl<'a> LockedDisk<'a> {
    pub fn new(settings: &'a BfsSettings, disk_path: PathBuf) -> Self {
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

        let mut file = File::open(&path)?;
        file.read_exact(buf)?;

        Ok(())
    }

    pub fn try_read_to_string<'b>(&self, path: &'b Path) -> Result<String, Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        tracing::trace!("reading file {path:?}");

        let mut file = File::open(&path)?;
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

        Ok(std::fs::read(&path)?)
    }

    pub fn try_write_file<'b>(&self, path: &'b Path, data: &[u8]) -> Result<(), Error<'b>> {
        self.try_write_file_multiple_buffers(path, &[data])
    }

    pub fn try_write_file_multiple_buffers<'b>(
        &self,
        path: &'b Path,
        data: &[&[u8]],
    ) -> Result<(), Error<'b>> {
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

        drop(file);

        std::fs::rename(path_tmp, path)?;

        Ok(())
    }

    fn try_delete_file<'b>(&self, path: &'b Path) -> Result<(), Error<'b>> {
        if !self.is_on_disk(path) {
            return Err(Error::FileNotOnDisk(path));
        }

        let _lock = self.lock();

        tracing::trace!("deleting file {path:?}");

        std::fs::remove_file(path)?;

        Ok(())
    }
}

pub struct LockedIO<'a> {
    settings: &'a BfsSettings,
    disks: Vec<LockedDisk<'a>>,
    deletion_queue: Mutex<Vec<PathBuf>>,
}

impl<'a> LockedIO<'a> {
    pub fn new(settings: &'a BfsSettings, disk_paths: Vec<PathBuf>) -> Self {
        let disks = disk_paths
            .into_iter()
            .map(|disk_path| LockedDisk::new(settings, disk_path))
            .collect();

        Self {
            settings,
            disks,
            deletion_queue: Mutex::new(Vec::new()),
        }
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

    pub fn try_write_file<'b>(&self, path: &'b Path, data: &[u8]) -> Result<(), Error<'b>> {
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
    ) -> Result<(), Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_write_file_multiple_buffers(path, data);
            match result {
                Err(Error::FileNotOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk(path))
    }

    fn try_delete_file<'b>(&self, path: &'b Path) -> Result<(), Error<'b>> {
        for disk in &self.disks {
            let result = disk.try_delete_file(path);
            match result {
                Err(Error::FileNotOnDisk(_)) => continue,
                _ => return result,
            }
        }

        Err(Error::FileNotOnAnyDisk(path))
    }

    pub fn read_file<'b>(&self, path: &'b Path, buf: &mut [u8]) {
        self.try_read_file(path, buf).unwrap();
    }

    pub fn read_to_string(&self, path: &Path) -> String {
        self.try_read_to_string(path).unwrap()
    }

    pub fn read_to_vec(&self, path: &Path) -> Vec<u8> {
        self.try_read_to_vec(path).unwrap()
    }

    pub fn write_file(&self, path: &Path, data: &[u8]) {
        self.try_write_file(path, data).unwrap();
    }

    pub fn write_file_multiple_buffers(&self, path: &Path, data: &[&[u8]]) {
        self.try_write_file_multiple_buffers(path, data).unwrap();
    }

    pub fn queue_deletion(&self, path: PathBuf) {
        let mut deletion_queue_lock = self.deletion_queue.lock().expect("failed to acquire lock");
        deletion_queue_lock.push(path);

        let num_files = deletion_queue_lock.len();

        if num_files >= 256 {
            if self.settings.sync_filesystem {
                sync();
            }

            tracing::debug!("flushing deletion queue ({num_files} files)");

            for path in deletion_queue_lock.drain(..) {
                // It's possible that the file has already been deleted, e.g. by an end of depth
                // cleanup. It's not important if deleting a file fails, so ignore the result.
                _ = self.try_delete_file(&path);
            }

            tracing::debug!("finished flushing deletion queue");
        }
    }
}

pub fn sync() {
    tracing::debug!("syncing filesystem");

    unsafe {
        libc::sync();
    }
}
