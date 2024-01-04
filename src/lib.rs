#![feature(allocator_api)]

use std::alloc::{AllocError, Allocator, Layout};
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::ErrorKind;
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::ptr::NonNull;
use std::thread::panicking;

use libc::{
    mmap, munmap, off_t, MAP_ANONYMOUS, MAP_FIXED, MAP_PRIVATE, MAP_SHARED, O_TMPFILE, PROT_NONE,
    PROT_READ, PROT_WRITE,
};
use linked_list_allocator::Heap;
pub use spin::Mutex;

const INITIAL_SIZE: usize = 4 << 20; // 4 MB

/// Trait which denotes that an allocator may be synchronised across forks.
pub trait SynchronizableAllocator {
    /// Perform synchronisation to ensure that the heap currently represents what is present across forks.
    /// You will not need to do this manually unless handling raw pointers in the synchronized heap.
    fn sync(&self) -> Result<(), AllocError>;
}

/// Metadata stored within the shared memory region regarding the heap.
struct HeapMetadata {
    /// Base address of the shared memory region.
    reserved: *mut c_void,
    /// Current size of the shared memory region.
    memsize: UnsafeCell<usize>,
    /// The heap, guarded by a spinlock mutex.
    heap: Mutex<Heap>,
}

/// A file-backed allocator for use in multiprocess programs or with very large allocations. Concurrent heap access is
/// managed by a spinlock mutex.
pub struct FileAllocator {
    /// The file which backs this allocator.
    file: File,
    /// The total size of the heap permitted.
    total: usize,
    /// The metadata of the heap as stored within the file.
    meta: &'static mut HeapMetadata,
    /// The current known size of the file. When this does not match the heap metadata, we are out of sync.
    /// Access to this field must be blocked by the mutex in the heap metadata.
    filesize: UnsafeCell<usize>,
}

impl FileAllocator {
    /// Create a new file-backed allocator with a given max total file size. The path provided must be a directory on
    /// a writable filesystem.
    pub fn new<P: AsRef<Path>>(within: P, total: usize) -> Result<Self, io::Error> {
        let file = OpenOptions::new()
            .custom_flags(O_TMPFILE)
            .write(true)
            .read(true)
            .open(within)?;

        file.set_len(INITIAL_SIZE as u64)?;

        // reserve the max potential heap size
        let reserved = unsafe {
            mmap(
                core::ptr::null_mut(),
                total,
                PROT_NONE,
                MAP_ANONYMOUS | MAP_PRIVATE,
                -1,
                0,
            )
        };
        if !reserved.is_null() {
            let base = unsafe {
                mmap(
                    reserved,
                    INITIAL_SIZE,
                    PROT_READ | PROT_WRITE,
                    MAP_FIXED | MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                ) as *mut u8
            };
            if base as usize != usize::MAX {
                let base = base as *mut MaybeUninit<HeapMetadata>;
                let meta = unsafe { base.as_mut() }.unwrap();

                let heap_base = unsafe { base.offset(1) as *mut u8 };
                let size = INITIAL_SIZE - unsafe { heap_base.byte_offset_from(base) as usize };

                let heap = unsafe { Heap::new(heap_base, size) };

                let meta = meta.write(HeapMetadata {
                    reserved,
                    memsize: UnsafeCell::new(INITIAL_SIZE),
                    heap: Mutex::new(heap),
                });

                let result = Self {
                    file,
                    total,
                    meta,
                    filesize: UnsafeCell::new(INITIAL_SIZE),
                };
                return Ok(result);
            }
        }
        Err(io::Error::new(
            ErrorKind::AddrNotAvailable,
            io::Error::last_os_error(),
        ))
    }

    fn sync_unlocked(&self) -> Result<(), AllocError> {
        let heap_filesize = unsafe { *self.meta.memsize.get() };
        let my_filesize = unsafe { self.filesize.get().as_mut().unwrap() };
        if *my_filesize != heap_filesize {
            let v = unsafe {
                mmap(
                    self.meta.reserved,
                    heap_filesize,
                    PROT_READ | PROT_WRITE,
                    MAP_FIXED | MAP_SHARED,
                    self.file.as_raw_fd(),
                    0,
                )
            } as isize;
            if v.is_negative() || v != self.meta.reserved as isize {
                return Err(AllocError);
            }
            *my_filesize = heap_filesize;
        }
        Ok(())
    }
}

impl SynchronizableAllocator for FileAllocator {
    fn sync(&self) -> Result<(), AllocError> {
        let heap = self.meta.heap.lock();
        self.sync_unlocked()?;
        drop(heap);
        Ok(())
    }
}

impl Drop for FileAllocator {
    fn drop(&mut self) {
        if !panicking() {
            assert_eq!(self.meta.heap.lock().used(), 0); // must allocate beforehand!
            if unsafe { munmap(self.meta.reserved, self.total) } < 0 {
                panic!("munmap failed: {}", io::Error::last_os_error())
            }
        }
    }
}

unsafe impl Allocator for FileAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let mut heap = self.meta.heap.lock();
        self.sync_unlocked()?;

        let mut allocation = heap.allocate_first_fit(layout);
        while allocation.is_err() {
            let filesize = unsafe { self.meta.memsize.get().as_mut() }
                .expect("contract with UnsafeCell violated");
            let old_filesize = *filesize;
            if old_filesize == self.total {
                return Err(AllocError);
            }
            let new_filesize = (old_filesize * 2).min(self.total);
            let extension = new_filesize - old_filesize;
            *filesize = new_filesize;

            if unsafe { libc::fallocate(self.file.as_raw_fd(), 0, 0, new_filesize as off_t) } < 0 {
                let e = io::Error::last_os_error();
                match e.kind() {
                    ErrorKind::OutOfMemory => return Err(AllocError),
                    _ => panic!("unrecoverable io error: {e}"),
                }
            }
            self.sync_unlocked()?;
            unsafe {
                heap.extend(extension);
            }
            allocation = heap.allocate_first_fit(layout);
        }
        allocation
            .map(|ptr| NonNull::slice_from_raw_parts(ptr, layout.size()))
            .map_err(|_| AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let mut heap = self.meta.heap.lock();
        heap.deallocate(ptr, layout);
    }
}

/// Box-like container for data in a shared heap that is accessed concurrently by multiple forks.
///
/// An Arc-like container is not provided, as counters would not be updated over [`libc::fork`].
pub struct ForkBox<'a, T, A>
where
    T: ?Sized,
{
    inner: *mut T,
    allocator: &'a A,
}

impl<'a, T, A> ForkBox<'a, T, A>
where
    T: ?Sized,
    A: Allocator + SynchronizableAllocator,
{
    /// Get a reference to the data. You are responsible for maintaining data safety!
    ///
    /// # Safety
    ///
    /// Use like [`UnsafeCell`]. This should be safe with inline spinlock mutexes, but other mutexes may have undefined
    /// behaviour.
    pub unsafe fn get(&self) -> Result<&'a T, AllocError> {
        self.allocator.sync()?;
        Ok(self.inner.as_mut().expect("invalid pointer provided"))
    }

    /// Get a mutable reference to the data. You are responsible for maintaining data safety!
    ///
    /// # Safety
    ///
    /// Use like [`UnsafeCell`]. Accessing mutably requires you to ensure that no other process currently is accessing
    /// the data.
    pub unsafe fn get_mut(&self) -> Result<&'a mut T, AllocError> {
        self.allocator.sync()?;
        Ok(self.inner.as_mut().expect("invalid pointer provided"))
    }

    /// Capture and own the data. You are responsible for maintaining data safety!
    ///
    /// # Safety
    ///
    /// Use like [`UnsafeCell`]. Taking ownership requires you to ensure that no other process currently is accessing
    /// the data.
    pub unsafe fn into_inner(self) -> Result<Box<T, &'a A>, AllocError> {
        self.allocator.sync()?;
        Ok(Box::from_raw_in(self.inner, self.allocator))
    }
}

impl<'a, T, A> From<Box<T, &'a A>> for ForkBox<'a, T, A>
where
    T: ?Sized,
    A: Allocator + SynchronizableAllocator,
{
    fn from(value: Box<T, &'a A>) -> Self {
        let (inner, allocator) = Box::into_raw_with_allocator(value);
        Self { inner, allocator }
    }
}

#[cfg(test)]
mod test {
    use std::{io, iter};

    use libc::{_exit, fork, waitpid};

    use crate::{FileAllocator, ForkBox};

    const WITHIN: &str = ".";

    #[test]
    fn basic_vec() -> Result<(), io::Error> {
        let allocator = FileAllocator::new(WITHIN, 32 << 30)?; // allow up to 32GB
        {
            let mut vec = Vec::new_in(&allocator);

            vec.extend_from_slice(&[0u8; 32]);
            for _ in 0..32 {
                vec.extend(iter::repeat(0u8).take(8 << 20)); // allocate 8MB
            }
        }

        Ok(())
    }

    fn many_vec_inner(allocator: &FileAllocator) {
        let mut vec = Vec::new();

        // allocate 4GB in 1MB chunks
        for _ in 0usize..(4 << 10) {
            let next = Vec::<u8, _>::with_capacity_in(1 << 20, allocator); // 1MB
            vec.push(next);
        }
    }

    #[test]
    fn many_vec() -> Result<(), io::Error> {
        let allocator = FileAllocator::new(WITHIN, 32 << 30)?; // allow up to 32GB
        many_vec_inner(&allocator);

        Ok(())
    }

    #[test]
    fn many_vec_dealloc() -> Result<(), io::Error> {
        let allocator = FileAllocator::new(WITHIN, 32 << 30)?; // allow up to 32GB
        for _ in 0usize..8 {
            many_vec_inner(&allocator);
        }

        Ok(())
    }

    #[test]
    fn fork_sanity_simple() -> Result<(), io::Error> {
        let allocator = FileAllocator::new(WITHIN, 32 << 30)?; // allow up to 32GB
        assert_eq!(allocator.meta.heap.lock().used(), 0);
        {
            let vec = Box::new_in(Vec::new_in(&allocator), &allocator);
            let vec = ForkBox::from(vec);
            assert_ne!(allocator.meta.heap.lock().used(), 0);

            match unsafe { fork() } {
                -1 => {
                    panic!("fork failed: {}", io::Error::last_os_error())
                }
                0 => {
                    // we are baby :)
                    let vec = unsafe { vec.get_mut() }.unwrap();

                    for _ in 0..32 {
                        vec.extend(iter::repeat(1u8).take(8 << 20)); // allocate 8MB
                    }

                    unsafe {
                        _exit(0);
                    }
                }
                pid => {
                    // we are parent
                    if unsafe { waitpid(pid, core::ptr::null_mut(), 0) } < 0 {
                        panic!("wait failed: {}", io::Error::last_os_error())
                    }
                    // the child process has ended, which means we can safely access the vec again
                    let vec = unsafe { vec.into_inner() }.unwrap();

                    assert_eq!(vec.len(), 32 * (8 << 20));
                    let heap_before = allocator.meta.heap.lock().used();
                    assert!(vec.into_iter().all(|e| e == 1));
                    assert!(allocator.meta.heap.lock().used() < heap_before);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn fork_sanity_chaos() -> Result<(), io::Error> {
        const ALLOCATIONS: usize = 1 << 10;

        let allocator = FileAllocator::new(WITHIN, 32 << 30)?; // allow up to 32GB
        assert_eq!(allocator.meta.heap.lock().used(), 0);
        {
            let vec = Box::new_in(Vec::new_in(&allocator), &allocator);
            let vec = ForkBox::from(vec);
            assert_ne!(allocator.meta.heap.lock().used(), 0);

            match unsafe { fork() } {
                -1 => {
                    panic!("fork failed: {}", io::Error::last_os_error())
                }
                0 => {
                    // we are baby :)
                    let vec = unsafe { vec.get_mut() }.unwrap();

                    for _ in 0..32 {
                        vec.extend(iter::repeat(Box::new_in(1u8, &allocator)).take(ALLOCATIONS));
                    }

                    unsafe {
                        _exit(0);
                    }
                }
                pid => {
                    // we are parent
                    let mut intermediary = Vec::new_in(&allocator);

                    for _ in 0..32 {
                        intermediary
                            .extend(iter::repeat(Box::new_in(1u8, &allocator)).take(ALLOCATIONS));
                    }

                    if unsafe { waitpid(pid, core::ptr::null_mut(), 0) } < 0 {
                        panic!("wait failed: {}", io::Error::last_os_error())
                    }
                    // the child process has ended, which means we can safely access the vec again
                    let mut vec = unsafe { vec.into_inner() }.unwrap();

                    vec.extend(intermediary);

                    assert_eq!(vec.len(), 2 * 32 * (ALLOCATIONS));
                    let heap_before = allocator.meta.heap.lock().used();
                    assert!(vec.into_iter().all(|e| *e == 1));
                    assert!(allocator.meta.heap.lock().used() < heap_before);
                }
            }
        }

        Ok(())
    }
}
