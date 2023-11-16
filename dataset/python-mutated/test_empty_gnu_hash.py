import os
import stat
import ctypes
import lief
import pytest
from utils import get_sample, has_recent_glibc, is_linux, is_x86_64
SYMBOLS = {'myinstance': 4441, 'myinit': 4469, 'mycalc': 4505, 'mydelete': 4628}

@pytest.mark.skipif(not is_linux() or not is_x86_64(), reason='requires Linux x86-64')
@pytest.mark.skipif(not has_recent_glibc(), reason='needs a recent GLIBC version')
def test_gnu_hash(tmpdir):
    if False:
        print('Hello World!')
    target_path = get_sample('ELF/ELF64_x86-64_binary_empty-gnu-hash.bin')
    output = os.path.join(tmpdir, 'libnoempty.so')
    binary = lief.parse(target_path)
    binary[lief.ELF.DYNAMIC_TAGS.FLAGS_1].remove(lief.ELF.DYNAMIC_FLAGS_1.PIE)
    for (name, addr) in SYMBOLS.items():
        binary.add_exported_function(addr, name)
    binary.write(output)
    st = os.stat(output)
    os.chmod(output, st.st_mode | stat.S_IEXEC)
    print(output)
    lib = ctypes.cdll.LoadLibrary(output)
    print(lib.myinstance)
    assert lib.myinstance is not None