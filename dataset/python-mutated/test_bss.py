import os
import stat
import subprocess
import pytest
from pathlib import Path
from subprocess import Popen
import lief
from utils import get_sample, is_linux, is_x86_64, is_64bits_platform
lief.logging.set_level(lief.logging.LOGGING_LEVEL.INFO)

def test_issue_671(tmp_path: Path):
    if False:
        print('Hello World!')
    '\n    Test on the support of bss-like segments where virtual_address - imagebase != offset\n    cf. https://github.com/lief-project/LIEF/issues/671\n    '
    binary_name = 'nopie_bss_671.elf'
    target: lief.ELF.Binary = lief.parse(get_sample(f'ELF/{binary_name}'))
    for s in filter(lambda e: e.exported, target.static_symbols):
        target.add_dynamic_symbol(s)
    output = tmp_path / binary_name
    target.write(output.as_posix())
    built = lief.parse(output.as_posix())
    assert built[lief.ELF.SEGMENT_TYPES.PHDR].file_offset == 12288
    assert built[lief.ELF.SEGMENT_TYPES.PHDR].physical_size == 504
    assert built[lief.ELF.SEGMENT_TYPES.PHDR].virtual_address == 4206592
    if is_linux() and is_x86_64():
        st = os.stat(output)
        os.chmod(output, st.st_mode | stat.S_IEXEC)
        with Popen(output.as_posix(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as P:
            stdout = P.stdout.read().decode('utf8')
            print(stdout)
            assert len(stdout) > 0

def test_all(tmp_path: Path):
    if False:
        return 10
    if not is_64bits_platform():
        pytest.skip('requires a 64-bits platform')
    binary_name = '544ca2035a9c15e7756ed8d8067d860bd3157e4eeaa39b4ee932458eebe2434b.elf'
    target: lief.ELF.Binary = lief.parse(get_sample(f'ELF/{binary_name}'))
    bss = target.get_section('.bss')
    assert bss.virtual_address == 6661088
    assert bss.size == 483091248
    assert bss.file_offset == 369632
    assert len(bss.content) == 0
    target.add_library('libcap.so.2')
    new_segment = lief.ELF.Segment()
    new_segment.type = lief.ELF.SEGMENT_TYPES.LOAD
    new_segment.content = [204] * 80
    target.add(new_segment)
    output = tmp_path / f'{binary_name}.build'
    target.write(output.as_posix())
    if is_linux() and is_x86_64():
        st = os.stat(output)
        os.chmod(output, st.st_mode | stat.S_IEXEC)
        with Popen(output.as_posix(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as P:
            stdout = P.stdout.read().decode('utf8')
            print(stdout)
            assert len(stdout) > 0
    new: lief.ELF.Binary = lief.parse(output.as_posix())
    assert new.get_library('libcap.so.2').name == 'libcap.so.2'