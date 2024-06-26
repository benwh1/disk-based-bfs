use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

pub struct LockedDisk {
    lock: Mutex<()>,
    disk_path: PathBuf,
}

impl LockedDisk {
    pub fn new(disk_path: PathBuf) -> Self {
        Self {
            lock: Mutex::new(()),
            disk_path,
        }
    }

    fn is_on_disk(&self, path: &Path) -> bool {
        path.starts_with(&self.disk_path)
    }

    pub fn try_read_file(&self, path: &Path, buf: &mut [u8]) -> Option<()> {
        if !self.is_on_disk(path) {
            return None;
        }

        let _lock = self.lock.lock().expect("failed to acquire lock");

        tracing::info!("reading file {path:?}");

        let mut file = File::open(&path).ok()?;
        file.read_exact(buf).ok()?;

        Some(())
    }

    pub fn try_read_to_string(&self, path: &Path) -> Option<String> {
        if !self.is_on_disk(path) {
            return None;
        }

        let _lock = self.lock.lock().expect("failed to acquire lock");

        tracing::info!("reading file {path:?}");

        let mut file = File::open(&path).ok()?;
        let mut buf = String::new();
        file.read_to_string(&mut buf).ok()?;

        Some(buf)
    }

    pub fn try_read_to_vec(&self, path: &Path) -> Option<Vec<u8>> {
        if !self.is_on_disk(path) {
            return None;
        }

        let _lock = self.lock.lock().expect("failed to acquire lock");

        tracing::info!("reading file {path:?}");

        std::fs::read(&path).ok()
    }

    pub fn try_write_file(&self, path: &Path, data: &[u8]) -> Option<()> {
        if !self.is_on_disk(path) {
            return None;
        }

        let _lock = self.lock.lock().expect("failed to acquire lock");

        tracing::info!("writing file {path:?}");

        let path_tmp = path.with_extension("tmp");
        let mut file = File::create(&path_tmp).ok()?;
        file.write_all(&data).ok()?;

        drop(file);

        std::fs::rename(path_tmp, path).ok()?;

        Some(())
    }

    fn try_delete_file(&self, path: &PathBuf) -> Option<()> {
        if !self.is_on_disk(path) {
            return None;
        }

        let _lock = self.lock.lock().expect("failed to acquire lock");

        tracing::info!("deleting file {path:?}");

        std::fs::remove_file(path).ok()?;

        Some(())
    }
}

pub struct LockedIO {
    disks: Vec<LockedDisk>,
    deletion_queue: Mutex<Vec<PathBuf>>,
    sync_filesystem: bool,
}

impl LockedIO {
    pub fn new(disk_paths: Vec<PathBuf>, sync_filesystem: bool) -> Self {
        let disks = disk_paths.into_iter().map(LockedDisk::new).collect();
        Self {
            disks,
            deletion_queue: Mutex::new(Vec::new()),
            sync_filesystem,
        }
    }

    pub fn try_read_file(&self, path: &Path, buf: &mut [u8]) -> bool {
        for disk in &self.disks {
            if disk.try_read_file(path, buf).is_some() {
                return true;
            }
        }

        false
    }

    pub fn try_read_to_string(&self, path: &Path) -> Option<String> {
        for disk in &self.disks {
            if let Some(str) = disk.try_read_to_string(path) {
                return Some(str);
            }
        }

        None
    }

    pub fn try_read_to_vec(&self, path: &Path) -> Option<Vec<u8>> {
        for disk in &self.disks {
            if let Some(vec) = disk.try_read_to_vec(path) {
                return Some(vec);
            }
        }

        None
    }

    pub fn try_write_file(&self, path: &Path, data: &[u8]) -> bool {
        for disk in &self.disks {
            if disk.try_write_file(path, data).is_some() {
                return true;
            }
        }

        false
    }

    pub fn read_file(&self, path: &Path, buf: &mut [u8]) {
        if !self.try_read_file(path, buf) {
            panic!("file path {path:?} not on any disk");
        }
    }

    pub fn read_to_string(&self, path: &Path) -> String {
        match self.try_read_to_string(path) {
            Some(str) => str,
            None => panic!("file path {path:?} not on any disk"),
        }
    }

    pub fn read_to_vec(&self, path: &Path) -> Vec<u8> {
        match self.try_read_to_vec(path) {
            Some(vec) => vec,
            None => panic!("file path {path:?} not on any disk"),
        }
    }

    pub fn write_file(&self, path: &Path, data: &[u8]) {
        if !self.try_write_file(path, data) {
            panic!("file path {path:?} not on any disk");
        }
    }

    pub fn queue_deletion(&self, path: PathBuf) {
        let mut deletion_queue_lock = self.deletion_queue.lock().expect("failed to acquire lock");
        deletion_queue_lock.push(path);

        if deletion_queue_lock.len() >= 256 {
            if self.sync_filesystem {
                sync();
            }

            tracing::info!("flushing deletion queue");

            for path in deletion_queue_lock.drain(..) {
                for disk in &self.disks {
                    disk.try_delete_file(&path);
                }
            }
        }
    }
}

pub fn sync() {
    tracing::info!("syncing filesystem");

    unsafe {
        libc::sync();
    }
}
