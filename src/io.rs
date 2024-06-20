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

    pub fn try_read_file(&self, path: &Path, buf: &mut [u8]) -> bool {
        if !self.is_on_disk(path) {
            return false;
        }

        let _lock = self.lock.lock().unwrap();

        tracing::info!("reading file {path:?}");

        let mut file = File::open(&path).unwrap();
        file.read_exact(buf).unwrap();

        true
    }

    pub fn try_read_to_string(&self, path: &Path) -> Option<String> {
        if !self.is_on_disk(path) {
            return None;
        }

        let _lock = self.lock.lock().unwrap();

        tracing::info!("reading file {path:?}");

        let mut file = File::open(&path).unwrap();
        let mut buf = String::new();
        file.read_to_string(&mut buf).unwrap();

        Some(buf)
    }

    pub fn try_read_to_vec(&self, path: &Path) -> Option<Vec<u8>> {
        if !self.is_on_disk(path) {
            return None;
        }

        let _lock = self.lock.lock().unwrap();

        tracing::info!("reading file {path:?}");

        std::fs::read(&path).ok()
    }

    pub fn try_write_file(&self, path: &Path, data: &[u8]) -> bool {
        if !self.is_on_disk(path) {
            return false;
        }

        let _lock = self.lock.lock().unwrap();

        tracing::info!("writing file {path:?}");

        let path_tmp = path.with_extension("tmp");
        let mut file = File::create(&path_tmp).unwrap();
        file.write_all(&data).unwrap();

        drop(file);

        std::fs::rename(path_tmp, path).unwrap();

        true
    }
}

pub struct LockedIO {
    disks: Vec<LockedDisk>,
}

impl LockedIO {
    pub fn new(disk_paths: Vec<PathBuf>) -> Self {
        let disks = disk_paths.into_iter().map(LockedDisk::new).collect();
        Self { disks }
    }

    pub fn try_read_file(&self, path: &Path, buf: &mut [u8]) -> bool {
        for disk in &self.disks {
            if disk.try_read_file(path, buf) {
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
            if disk.try_write_file(path, data) {
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
}
