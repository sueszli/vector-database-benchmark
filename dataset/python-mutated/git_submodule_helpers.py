import os
import textwrap
from pathlib import Path
from typing import Tuple
from tests.lib import PipTestEnvironment, _create_main_file, _git_commit

def _create_test_package_submodule(env: PipTestEnvironment) -> Path:
    if False:
        for i in range(10):
            print('nop')
    env.scratch_path.joinpath('version_pkg_submodule').mkdir()
    submodule_path = env.scratch_path / 'version_pkg_submodule'
    env.run('touch', 'testfile', cwd=submodule_path)
    env.run('git', 'init', cwd=submodule_path)
    env.run('git', 'add', '.', cwd=submodule_path)
    _git_commit(env, submodule_path, message='initial version / submodule')
    return submodule_path

def _change_test_package_submodule(env: PipTestEnvironment, submodule_path: Path) -> None:
    if False:
        i = 10
        return i + 15
    submodule_path.joinpath('testfile').write_text('this is a changed file')
    submodule_path.joinpath('testfile2').write_text('this is an added file')
    env.run('git', 'add', '.', cwd=submodule_path)
    _git_commit(env, submodule_path, message='submodule change')

def _pull_in_submodule_changes_to_module(env: PipTestEnvironment, module_path: Path, rel_path: str) -> None:
    if False:
        return 10
    '\n    Args:\n      rel_path: the location of the submodule relative to the superproject.\n    '
    submodule_path = module_path / rel_path
    env.run('git', 'pull', '-q', 'origin', 'master', cwd=submodule_path)
    _git_commit(env, module_path, message='submodule change', stage_modified=True)

def _create_test_package_with_submodule(env: PipTestEnvironment, rel_path: str) -> Tuple[Path, Path]:
    if False:
        print('Hello World!')
    '\n    Args:\n      rel_path: the location of the submodule relative to the superproject.\n    '
    env.scratch_path.joinpath('version_pkg').mkdir()
    version_pkg_path = env.scratch_path / 'version_pkg'
    version_pkg_path.joinpath('testpkg').mkdir()
    pkg_path = version_pkg_path / 'testpkg'
    pkg_path.joinpath('__init__.py').write_text('# hello there')
    _create_main_file(pkg_path, name='version_pkg', output='0.1')
    version_pkg_path.joinpath('setup.py').write_text(textwrap.dedent("                        from setuptools import setup, find_packages\n                        setup(name='version_pkg',\n                              version='0.1',\n                              packages=find_packages(),\n                             )\n                        "))
    env.run('git', 'init', cwd=version_pkg_path)
    env.run('git', 'add', '.', cwd=version_pkg_path)
    _git_commit(env, version_pkg_path, message='initial version')
    submodule_path = _create_test_package_submodule(env)
    env.run('git', 'submodule', 'add', os.fspath(submodule_path), rel_path, cwd=version_pkg_path)
    _git_commit(env, version_pkg_path, message='initial version w submodule')
    return (version_pkg_path, submodule_path)