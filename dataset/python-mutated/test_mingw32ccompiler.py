import shutil
import subprocess
import sys
import pytest
from numpy.distutils import mingw32ccompiler

@pytest.mark.skipif(sys.platform != 'win32', reason='win32 only test')
def test_build_import():
    if False:
        i = 10
        return i + 15
    'Test the mingw32ccompiler.build_import_library, which builds a\n    `python.a` from the MSVC `python.lib`\n    '
    try:
        out = subprocess.check_output(['nm.exe', '--help'])
    except FileNotFoundError:
        pytest.skip("'nm.exe' not on path, is mingw installed?")
    supported = out[out.find(b'supported targets:'):]
    if sys.maxsize < 2 ** 32:
        if b'pe-i386' not in supported:
            raise ValueError("'nm.exe' found but it does not support 32-bit dlls when using 32-bit python. Supported formats: '%s'" % supported)
    elif b'pe-x86-64' not in supported:
        raise ValueError("'nm.exe' found but it does not support 64-bit dlls when using 64-bit python. Supported formats: '%s'" % supported)
    (has_import_lib, fullpath) = mingw32ccompiler._check_for_import_lib()
    if has_import_lib:
        shutil.move(fullpath, fullpath + '.bak')
    try:
        mingw32ccompiler.build_import_library()
    finally:
        if has_import_lib:
            shutil.move(fullpath + '.bak', fullpath)