import os
import shlex
import stat
import subprocess
import pytest
from pathlib import Path
import lief
from utils import get_compiler, is_linux, is_x86_64, is_aarch64
if not is_linux():
    pytest.skip('requires Linux', allow_module_level=True)
lief.logging.set_level(lief.logging.LOGGING_LEVEL.INFO)

class CommandResult:

    def __init__(self, output, error, retcode, process=None):
        if False:
            while True:
                i = 10
        self.output = output
        self.error = error
        self.retcode = retcode
        self.process = process

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        return not self.retcode

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if bool(self):
            return self.output
        return self.error
LIBADD_C = '#include <stdlib.h>\n#include <stdio.h>\n#define LOCAL __attribute__ ((visibility ("hidden")))\n\nLOCAL int add_hidden(int a, int b) {\n  printf("[LOCAL] %d + %d = %d\\n", a, b, a+b);\n  return a + b;\n}\n\n\nint main(int argc, char** argv) {\n\n  if (argc != 3) {\n    printf("Usage: %s <a> <b>\\n", argv[0]);\n    exit(-1);\n  }\n\n  printf("Hello\\n");\n  int res = add_hidden(atoi(argv[1]), atoi(argv[2]));\n  printf("From add_hidden@libadd.so a + b = %d\\n", res);\n  return 0;\n}\n'
BINADD_C = '#include <stdio.h>\n#include <stdlib.h>\nextern int add_hidden(int a, int b);\n\nint main(int argc, char **argv) {\n  if (argc != 3) {\n    printf("Usage: %s <a> <b>\\n", argv[0]);\n    exit(-1);\n  }\n\n  printf("Hello\\n");\n  int res = add_hidden(atoi(argv[1]), atoi(argv[2]));\n  printf("From add_hidden@libadd.so a + b = %d\\n", res);\n  return 0;\n}\n'

def run_cmd(cmd):
    if False:
        while True:
            i = 10
    print(f"Running: '{cmd}'")
    cmd = shlex.split(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    (stdout, stderr) = p.communicate()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    return CommandResult(stdout, stderr, p.returncode)

def modif_1(libadd: lief.ELF.Binary, output: Path):
    if False:
        return 10
    libadd_hidden = libadd.get_symbol('add_hidden')
    libadd_hidden.binding = lief.ELF.SYMBOL_BINDINGS.GLOBAL
    libadd_hidden.visibility = lief.ELF.SYMBOL_VISIBILITY.DEFAULT
    libadd_hidden = libadd.add_dynamic_symbol(libadd_hidden, lief.ELF.SymbolVersion.global_)
    if lief.ELF.DYNAMIC_TAGS.FLAGS_1 in libadd and libadd[lief.ELF.DYNAMIC_TAGS.FLAGS_1].has(lief.ELF.DYNAMIC_FLAGS_1.PIE):
        libadd[lief.ELF.DYNAMIC_TAGS.FLAGS_1].remove(lief.ELF.DYNAMIC_FLAGS_1.PIE)
    print(libadd_hidden)
    libadd.add(lief.ELF.DynamicSharedObject(output.name))
    libadd.write(output.as_posix())

def modif_2(libadd: lief.ELF.Binary, output: Path):
    if False:
        for i in range(10):
            print('nop')
    libadd.export_symbol('add_hidden')
    if lief.ELF.DYNAMIC_TAGS.FLAGS_1 in libadd and libadd[lief.ELF.DYNAMIC_TAGS.FLAGS_1].has(lief.ELF.DYNAMIC_FLAGS_1.PIE):
        libadd[lief.ELF.DYNAMIC_TAGS.FLAGS_1].remove(lief.ELF.DYNAMIC_FLAGS_1.PIE)
    libadd.write(output.as_posix())

def modif_3(libadd: lief.ELF.Binary, output: Path):
    if False:
        for i in range(10):
            print('nop')
    add_hidden_static = libadd.get_static_symbol('add_hidden')
    libadd.add_exported_function(add_hidden_static.value, add_hidden_static.name)
    if lief.ELF.DYNAMIC_TAGS.FLAGS_1 in libadd and libadd[lief.ELF.DYNAMIC_TAGS.FLAGS_1].has(lief.ELF.DYNAMIC_FLAGS_1.PIE):
        libadd[lief.ELF.DYNAMIC_TAGS.FLAGS_1].remove(lief.ELF.DYNAMIC_FLAGS_1.PIE)
    libadd.write(output.as_posix())

@pytest.mark.parametrize('modifier', [modif_1, modif_2, modif_3])
def test_libadd(tmp_path: Path, modifier):
    if False:
        print('Hello World!')
    if not is_linux():
        pytest.skip('unsupported system')
    libadd_src = tmp_path / 'libadd.c'
    binadd_src = tmp_path / 'binadd.c'
    libadd_src.write_text(LIBADD_C)
    binadd_src.write_text(BINADD_C)
    binadd_bin = tmp_path / 'binadd.bin'
    libadd_so = tmp_path / 'libadd.so'
    libadd2_so = tmp_path / 'libadd2.so'
    compiler = get_compiler()
    fmt = ''
    if is_x86_64():
        fmt = '{compiler} -Wl,--export-dynamic -mcmodel=large -fPIE -pie -o {output} {input}'
    if is_aarch64():
        fmt = '{compiler} -Wl,--export-dynamic -fPIE -pie -o {output} {input}'
    r = run_cmd(fmt.format(compiler=compiler, output=libadd_so, input=libadd_src))
    assert r
    libadd = lief.parse(libadd_so.as_posix())
    modifier(libadd, libadd2_so)
    lib_directory = libadd2_so.parent
    libname = libadd2_so.stem[3:]
    fmt = ''
    if is_x86_64():
        fmt = '{compiler} -Wl,--export-dynamic -mcmodel=large -fPIE -pie -Wl,-rpath={libdir} -L{libdir} -o {output} {input} -l{libadd2}'
    if is_aarch64():
        fmt = '{compiler} -Wl,--export-dynamic -fPIE -pie -Wl,-rpath={libdir} -L{libdir} -o {output} {input} -l{libadd2}'
    r = run_cmd(fmt.format(compiler=compiler, libdir=lib_directory, libadd2=libname, output=binadd_bin, input=binadd_src))
    assert r
    st = os.stat(binadd_bin)
    os.chmod(binadd_bin, st.st_mode | stat.S_IEXEC)
    r = run_cmd(f'{binadd_bin} 1 2')
    assert r
    assert 'From add_hidden@libadd.so a + b = 3' in r.output