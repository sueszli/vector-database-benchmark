def is_manylinux1_compatible():
    if False:
        while True:
            i = 10
    from distutils.util import get_platform
    if get_platform() not in ['linux-x86_64', 'linux-i686']:
        return False
    try:
        import _manylinux
        return bool(_manylinux.manylinux1_compatible)
    except (ImportError, AttributeError):
        pass
    return have_compatible_glibc(2, 5)

def have_compatible_glibc(major, minimum_minor):
    if False:
        return 10
    import ctypes
    process_namespace = ctypes.CDLL(None)
    try:
        gnu_get_libc_version = process_namespace.gnu_get_libc_version
    except AttributeError:
        return False
    gnu_get_libc_version.restype = ctypes.c_char_p
    version_str = gnu_get_libc_version()
    if not isinstance(version_str, str):
        version_str = version_str.decode('ascii')
    version = [int(piece) for piece in version_str.split('.')]
    assert len(version) == 2
    if major != version[0]:
        return False
    if minimum_minor > version[1]:
        return False
    return True
import sys
if is_manylinux1_compatible():
    print(f'{sys.executable} is manylinux1 compatible')
    sys.exit(0)
else:
    print(f'{sys.executable} is NOT manylinux1 compatible')
    sys.exit(1)