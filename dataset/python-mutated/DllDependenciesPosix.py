"""DLL dependency scan methods for POSIX (Linux, *BSD, MSYS2).

"""
import os
import sys
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.PythonFlavors import isAnacondaPython
from nuitka.Tracing import inclusion_logger
from nuitka.utils.Execution import executeProcess, withEnvironmentPathAdded
from nuitka.utils.SharedLibraries import getSharedLibraryRPATH
from nuitka.utils.Utils import isAlpineLinux, isPosixWindows
from .DllDependenciesCommon import getLdLibraryPath
_detected_python_rpath = None
ldd_result_cache = {}

def detectBinaryPathDLLsPosix(dll_filename, package_name, original_dir):
    if False:
        for i in range(10):
            print('nop')
    if ldd_result_cache.get(dll_filename):
        return ldd_result_cache[dll_filename]
    global _detected_python_rpath
    if _detected_python_rpath is None and (not isPosixWindows()):
        _detected_python_rpath = getSharedLibraryRPATH(sys.executable) or False
        if _detected_python_rpath:
            _detected_python_rpath = _detected_python_rpath.replace('$ORIGIN', os.path.dirname(sys.executable))
    python_rpaths = (_detected_python_rpath,) if _detected_python_rpath else ()
    with withEnvironmentPathAdded('LD_LIBRARY_PATH', *getLdLibraryPath(package_name=package_name, python_rpaths=python_rpaths, original_dir=original_dir)):
        (stdout, stderr, _exit_code) = executeProcess(command=('ldd', dll_filename))
    stderr = b'\n'.join((line for line in stderr.splitlines() if not line.startswith(b'ldd: warning: you do not have execution permission for')))
    inclusion_logger.debug('ldd output for %s is:\n%s' % (dll_filename, stdout))
    if stderr:
        inclusion_logger.debug('ldd error for %s is:\n%s' % (dll_filename, stderr))
    result = OrderedSet()
    for line in stdout.split(b'\n'):
        if not line:
            continue
        if b'=>' not in line:
            continue
        part = line.split(b' => ', 2)[1]
        if b'(' in part:
            filename = part[:part.rfind(b'(') - 1]
        else:
            filename = part
        if not filename:
            continue
        if str is not bytes:
            filename = filename.decode('utf8')
        if filename in ('not found', 'ldd'):
            continue
        filename = os.path.normpath(filename)
        filename_base = os.path.basename(filename)
        if any((filename_base == entry or filename_base.startswith(entry + '.') for entry in _linux_dll_ignore_list)):
            continue
        if not os.path.isabs(filename):
            inclusion_logger.sysexit('Error: Found a dependency with a relative path. Was a dependency copied to dist early? ' + filename)
        result.add(filename)
    ldd_result_cache[dll_filename] = result
    sub_result = OrderedSet(result)
    for sub_dll_filename in result:
        sub_result = sub_result.union(detectBinaryPathDLLsPosix(dll_filename=sub_dll_filename, package_name=package_name, original_dir=original_dir))
    return sub_result
_linux_dll_ignore_list = ['linux-vdso.so.1', 'ld-linux-x86-64.so', 'libc.so', 'libpthread.so', 'libm.so', 'libdl.so', 'libBrokenLocale.so', 'libSegFault.so', 'libanl.so', 'libcidn.so', 'libcrypt.so', 'libmemusage.so', 'libmvec.so', 'libnsl.so', 'libnss3.so', 'libnssutil3.so', 'libnss_compat.so', 'libnss_db.so', 'libnss_dns.so', 'libnss_files.so', 'libnss_hesiod.so', 'libnss_nis.so', 'libnss_nisplus.so', 'libpcprofile.so', 'libresolv.so', 'librt.so', 'libthread_db-1.0.so', 'libthread_db.so', 'libutil.so', 'libstdc++.so', 'libdrm.so', 'libz.so']
if isAnacondaPython() or isAlpineLinux():
    _linux_dll_ignore_list.remove('libstdc++.so')