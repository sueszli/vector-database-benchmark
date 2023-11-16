import sys
import ctypes
import mmap as MMAP
mmap_function = None
munmap_function = None

def get_libc():
    if False:
        return 10
    osname = sys.platform.lower()
    if osname.startswith('darwin'):
        filename = 'libc.dylib'
    elif osname.startswith('linux'):
        filename = 'libc.so.6'
    elif osname.startswith('netbsd'):
        filename = 'libc.so'
    else:
        raise ValueError('Unsupported host OS: ' + osname)
    return ctypes.cdll.LoadLibrary(filename)
libc = get_libc()
mmap_function = libc.mmap
mmap_function.restype = ctypes.c_void_p
mmap_function.argtype = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
munmap_function = libc.munmap
munmap_function.restype = ctypes.c_int
munmap_function.argtype = [ctypes.c_void_p, ctypes.c_size_t]

def mmap(fd, offset, size):
    if False:
        return 10
    prot = MMAP.PROT_READ | MMAP.PROT_WRITE
    flags = MMAP.MAP_PRIVATE
    aligned_offset = offset & ~4095
    size += offset - aligned_offset
    if size & 4095 != 0:
        size = (size & ~4095) + 4096
    assert size > 0
    result = mmap_function(0, size, prot, flags, fd, aligned_offset)
    assert result != ctypes.c_void_p(-1).value
    return ctypes.cast(result + offset - aligned_offset, ctypes.POINTER(ctypes.c_char))

def munmap(address, size):
    if False:
        i = 10
        return i + 15
    address = ctypes.cast(address, ctypes.c_void_p).value
    aligned_address = address & ~4095
    size += address - aligned_address
    assert size > 0
    aligned_address = ctypes.cast(aligned_address, ctypes.POINTER(ctypes.c_char))
    result = munmap_function(aligned_address, size)
    assert result == 0