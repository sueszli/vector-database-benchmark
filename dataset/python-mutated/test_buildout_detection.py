import os
from textwrap import dedent
from pathlib import Path
from jedi.inference.sys_path import _get_parent_dir_with_file, _get_buildout_script_paths, check_sys_path_modifications
from ..helpers import get_example_dir

def check_module_test(Script, code):
    if False:
        for i in range(10):
            print('nop')
    module_context = Script(code)._get_module_context()
    return check_sys_path_modifications(module_context)

def test_parent_dir_with_file(Script):
    if False:
        while True:
            i = 10
    path = Path(get_example_dir('buildout_project', 'src', 'proj_name'))
    parent = _get_parent_dir_with_file(path, 'buildout.cfg')
    assert parent is not None
    assert str(parent).endswith(os.path.join('test', 'examples', 'buildout_project'))

def test_buildout_detection(Script):
    if False:
        while True:
            i = 10
    path = Path(get_example_dir('buildout_project', 'src', 'proj_name'))
    paths = list(_get_buildout_script_paths(path.joinpath('module_name.py')))
    assert len(paths) == 1
    appdir_path = os.path.normpath(os.path.join(path, '../../bin/app'))
    assert str(paths[0]) == appdir_path

def test_append_on_non_sys_path(Script):
    if False:
        return 10
    code = dedent("\n        class Dummy(object):\n            path = []\n\n        d = Dummy()\n        d.path.append('foo')")
    paths = check_module_test(Script, code)
    assert not paths
    assert 'foo' not in paths

def test_path_from_invalid_sys_path_assignment(Script):
    if False:
        i = 10
        return i + 15
    code = dedent("\n        import sys\n        sys.path = 'invalid'")
    paths = check_module_test(Script, code)
    assert not paths
    assert 'invalid' not in paths

def test_sys_path_with_modifications(Script):
    if False:
        print('Hello World!')
    path = get_example_dir('buildout_project', 'src', 'proj_name', 'module_name.py')
    code = dedent('\n        import os\n    ')
    paths = Script(code, path=path)._inference_state.get_sys_path()
    assert os.path.abspath('/tmp/.buildout/eggs/important_package.egg') in paths

def test_path_from_sys_path_assignment(Script):
    if False:
        i = 10
        return i + 15
    code = dedent(f"\n        #!/usr/bin/python\n\n        import sys\n        sys.path[0:0] = [\n          {os.path.abspath('/usr/lib/python3.8/site-packages')!r},\n          {os.path.abspath('/home/test/.buildout/eggs/important_package.egg')!r},\n          ]\n\n        path[0:0] = [1]\n\n        import important_package\n\n        if __name__ == '__main__':\n            sys.exit(important_package.main())")
    paths = check_module_test(Script, code)
    assert 1 not in paths
    assert os.path.abspath('/home/test/.buildout/eggs/important_package.egg') in map(str, paths)