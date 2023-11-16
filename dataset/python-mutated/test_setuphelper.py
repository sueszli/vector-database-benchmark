import os
import subprocess
import sys
from textwrap import dedent
import pytest
DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(os.path.dirname(DIR))
WIN = sys.platform.startswith('win32') or sys.platform.startswith('cygwin')

@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('std', [11, 0])
def test_simple_setup_py(monkeypatch, tmpdir, parallel, std):
    if False:
        while True:
            i = 10
    monkeypatch.chdir(tmpdir)
    monkeypatch.syspath_prepend(MAIN_DIR)
    (tmpdir / 'setup.py').write_text(dedent(f'            import sys\n            sys.path.append({MAIN_DIR!r})\n\n            from setuptools import setup, Extension\n            from pybind11.setup_helpers import build_ext, Pybind11Extension\n\n            std = {std}\n\n            ext_modules = [\n                Pybind11Extension(\n                    "simple_setup",\n                    sorted(["main.cpp"]),\n                    cxx_std=std,\n                ),\n            ]\n\n            cmdclass = dict()\n            if std == 0:\n                cmdclass["build_ext"] = build_ext\n\n\n            parallel = {parallel}\n            if parallel:\n                from pybind11.setup_helpers import ParallelCompile\n                ParallelCompile().install()\n\n            setup(\n                name="simple_setup_package",\n                cmdclass=cmdclass,\n                ext_modules=ext_modules,\n            )\n            '), encoding='ascii')
    (tmpdir / 'main.cpp').write_text(dedent('            #include <pybind11/pybind11.h>\n\n            int f(int x) {\n                return x * 3;\n            }\n            PYBIND11_MODULE(simple_setup, m) {\n                m.def("f", &f);\n            }\n            '), encoding='ascii')
    out = subprocess.check_output([sys.executable, 'setup.py', 'build_ext', '--inplace'])
    if not WIN:
        assert b'-g0' in out
    out = subprocess.check_output([sys.executable, 'setup.py', 'build_ext', '--inplace', '--force'], env=dict(os.environ, CFLAGS='-g'))
    if not WIN:
        assert b'-g0' not in out
    print(out)
    for item in tmpdir.listdir():
        print(item.basename)
    assert len([f for f in tmpdir.listdir() if f.basename.startswith('simple_setup')]) == 1
    assert len(list(tmpdir.listdir())) == 4
    (tmpdir / 'test.py').write_text(dedent('            import simple_setup\n            assert simple_setup.f(3) == 9\n            '), encoding='ascii')
    subprocess.check_call([sys.executable, 'test.py'], stdout=sys.stdout, stderr=sys.stderr)

def test_intree_extensions(monkeypatch, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.syspath_prepend(MAIN_DIR)
    from pybind11.setup_helpers import intree_extensions
    monkeypatch.chdir(tmpdir)
    root = tmpdir
    root.ensure_dir()
    subdir = root / 'dir'
    subdir.ensure_dir()
    src = subdir / 'ext.cpp'
    src.ensure()
    relpath = src.relto(tmpdir)
    (ext,) = intree_extensions([relpath])
    assert ext.name == 'ext'
    subdir.ensure('__init__.py')
    (ext,) = intree_extensions([relpath])
    assert ext.name == 'dir.ext'

def test_intree_extensions_package_dir(monkeypatch, tmpdir):
    if False:
        return 10
    monkeypatch.syspath_prepend(MAIN_DIR)
    from pybind11.setup_helpers import intree_extensions
    monkeypatch.chdir(tmpdir)
    root = tmpdir / 'src'
    root.ensure_dir()
    subdir = root / 'dir'
    subdir.ensure_dir()
    src = subdir / 'ext.cpp'
    src.ensure()
    (ext,) = intree_extensions([src.relto(tmpdir)], package_dir={'': 'src'})
    assert ext.name == 'dir.ext'
    (ext,) = intree_extensions([src.relto(tmpdir)], package_dir={'foo': 'src'})
    assert ext.name == 'foo.dir.ext'
    subdir.ensure('__init__.py')
    (ext,) = intree_extensions([src.relto(tmpdir)], package_dir={'': 'src'})
    assert ext.name == 'dir.ext'
    (ext,) = intree_extensions([src.relto(tmpdir)], package_dir={'foo': 'src'})
    assert ext.name == 'foo.dir.ext'