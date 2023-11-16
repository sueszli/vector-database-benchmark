"""Tests for numpy.distutils.build_ext."""
import os
import subprocess
import sys
from textwrap import indent, dedent
import pytest
from numpy.testing import IS_WASM

@pytest.mark.skipif(IS_WASM, reason='cannot start subprocess in wasm')
@pytest.mark.slow
def test_multi_fortran_libs_link(tmp_path):
    if False:
        print('Hello World!')
    '\n    Ensures multiple "fake" static libraries are correctly linked.\n    see gh-18295\n    '
    from numpy.f2py.tests.util import has_f77_compiler
    if not has_f77_compiler():
        pytest.skip('No F77 compiler found')
    with open(tmp_path / '_dummy1.f', 'w') as fid:
        fid.write(indent(dedent('            FUNCTION dummy_one()\n            RETURN\n            END FUNCTION'), prefix=' ' * 6))
    with open(tmp_path / '_dummy2.f', 'w') as fid:
        fid.write(indent(dedent('            FUNCTION dummy_two()\n            RETURN\n            END FUNCTION'), prefix=' ' * 6))
    with open(tmp_path / '_dummy.c', 'w') as fid:
        fid.write('int PyInit_dummyext;')
    with open(tmp_path / 'setup.py', 'w') as fid:
        srctree = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        fid.write(dedent(f'            def configuration(parent_package="", top_path=None):\n                from numpy.distutils.misc_util import Configuration\n                config = Configuration("", parent_package, top_path)\n                config.add_library("dummy1", sources=["_dummy1.f"])\n                config.add_library("dummy2", sources=["_dummy2.f"])\n                config.add_extension("dummyext", sources=["_dummy.c"], libraries=["dummy1", "dummy2"])\n                return config\n\n\n            if __name__ == "__main__":\n                import sys\n                sys.path.insert(0, r"{srctree}")\n                from numpy.distutils.core import setup\n                setup(**configuration(top_path="").todict())'))
    build_dir = tmp_path
    subprocess.check_call([sys.executable, 'setup.py', 'build', 'install', '--prefix', str(tmp_path / 'installdir'), '--record', str(tmp_path / 'tmp_install_log.txt')], cwd=str(build_dir))
    so = None
    with open(tmp_path / 'tmp_install_log.txt') as fid:
        for line in fid:
            if 'dummyext' in line:
                so = line.strip()
                break
    assert so is not None