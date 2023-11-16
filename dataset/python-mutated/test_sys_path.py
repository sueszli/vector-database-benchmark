import os
from glob import glob
import sys
import shutil
from pathlib import Path
import pytest
from ..helpers import skip_if_windows, skip_if_not_windows, get_example_dir
from jedi.inference import sys_path
from jedi.api.environment import create_environment

def test_paths_from_assignment(Script):
    if False:
        for i in range(10):
            print('nop')

    def paths(src):
        if False:
            while True:
                i = 10
        script = Script(src, path='/foo/bar.py')
        expr_stmt = script._module_node.children[0]
        return set(sys_path._paths_from_assignment(script._get_module_context(), expr_stmt))
    path_a = Path('/foo/a').absolute()
    path_b = Path('/foo/b').absolute()
    path_c = Path('/foo/c').absolute()
    assert paths('sys.path[0:0] = ["a"]') == {path_a}
    assert paths('sys.path = ["b", 1, x + 3, y, "c"]') == {path_b, path_c}
    assert paths('sys.path = a = ["a"]') == {path_a}
    assert paths('sys.path, other = ["a"], 2') == set()

def test_venv_and_pths(venv_path, environment):
    if False:
        print('Hello World!')
    pjoin = os.path.join
    if os.name == 'nt':
        if environment.version_info < (3, 11):
            site_pkg_path = pjoin(venv_path, 'lib', 'site-packages')
        else:
            site_pkg_path = pjoin(venv_path, 'Lib', 'site-packages')
    else:
        site_pkg_path = glob(pjoin(venv_path, 'lib', 'python*', 'site-packages'))[0]
    shutil.rmtree(site_pkg_path)
    shutil.copytree(get_example_dir('sample_venvs', 'pth_directory'), site_pkg_path)
    virtualenv = create_environment(venv_path)
    venv_paths = virtualenv.get_sys_path()
    ETALON = [site_pkg_path, pjoin(site_pkg_path, 'dir-from-foo-pth'), '/foo/smth.py:module', '/foo/smth.py:from_func', '/foo/smth.py:from_func']
    assert venv_paths[-len(ETALON):] == ETALON
    assert not set(sys.path).intersection(ETALON)
_s = ['/a', '/b', '/c/d/']

@pytest.mark.parametrize('sys_path_, module_path, expected, is_package', [(_s, '/a/b', ('b',), False), (_s, '/a/b/c', ('b', 'c'), False), (_s, '/a/b.py', ('b',), False), (_s, '/a/b/c.py', ('b', 'c'), False), (_s, '/x/b.py', None, False), (_s, '/c/d/x.py', ('x',), False), (_s, '/c/d/x.py', ('x',), False), (_s, '/c/d/x/y.py', ('x', 'y'), False), (_s, '/a/b.c.py', ('b.c',), False), (_s, '/a/b.d/foo.bar.py', ('b.d', 'foo.bar'), False), (_s, '/a/.py', None, False), (_s, '/a/c/.py', None, False), (['/foo'], '/foo/bar/__init__.py', ('bar',), True), (['/foo'], '/foo/bar/baz/__init__.py', ('bar', 'baz'), True), skip_if_windows(['/foo'], '/foo/bar.so', ('bar',), False), skip_if_windows(['/foo'], '/foo/bar/__init__.so', ('bar',), True), skip_if_not_windows(['/foo'], '/foo/bar.pyd', ('bar',), False), skip_if_not_windows(['/foo'], '/foo/bar/__init__.pyd', ('bar',), True), (['/foo'], '/x/bar.py', None, False), (['/foo'], '/foo/bar.xyz', ('bar.xyz',), False), (['/foo', '/foo/bar'], '/foo/bar/baz', ('baz',), False), (['/foo/bar', '/foo'], '/foo/bar/baz', ('baz',), False), (['/'], '/bar/baz.py', ('bar', 'baz'), False)])
def test_transform_path_to_dotted(sys_path_, module_path, expected, is_package):
    if False:
        return 10
    sys_path_ = [os.path.abspath(path) for path in sys_path_]
    module_path = os.path.abspath(module_path)
    assert sys_path.transform_path_to_dotted(sys_path_, Path(module_path)) == (expected, is_package)