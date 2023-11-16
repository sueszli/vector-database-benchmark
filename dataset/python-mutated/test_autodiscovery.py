import os
import sys
import pytest
from dagster import DagsterInvariantViolationError, RepositoryDefinition
from dagster._core.code_pointer import CodePointer
from dagster._core.definitions.reconstruct import repository_def_from_pointer
from dagster._core.definitions.repository_definition import PendingRepositoryDefinition
from dagster._core.errors import DagsterImportError
from dagster._core.workspace.autodiscovery import LOAD_ALL_ASSETS, loadable_targets_from_python_file, loadable_targets_from_python_module, loadable_targets_from_python_package
from dagster._utils import alter_sys_path, file_relative_path, restore_sys_modules

def test_single_repository():
    if False:
        while True:
            i = 10
    single_repo_path = file_relative_path(__file__, 'single_repository.py')
    loadable_targets = loadable_targets_from_python_file(single_repo_path)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == 'single_repository'
    repo_def = CodePointer.from_python_file(single_repo_path, symbol, None).load_target()
    assert isinstance(repo_def, RepositoryDefinition)
    assert repo_def.name == 'single_repository'

def test_double_repository():
    if False:
        return 10
    loadable_repos = loadable_targets_from_python_file(file_relative_path(__file__, 'double_repository.py'))
    found_names = set()
    for lr in loadable_repos:
        assert isinstance(lr.target_definition, RepositoryDefinition)
        found_names.add(lr.target_definition.name)
    assert found_names == {'repo_one', 'repo_two'}

def test_single_job():
    if False:
        return 10
    single_job_path = file_relative_path(__file__, 'single_job.py')
    loadable_targets = loadable_targets_from_python_file(single_job_path)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == 'a_job'
    repo_def = repository_def_from_pointer(CodePointer.from_python_file(single_job_path, symbol, None))
    assert isinstance(repo_def, RepositoryDefinition)
    assert repo_def.get_job('a_job')

def test_double_job():
    if False:
        for i in range(10):
            print('nop')
    double_job_path = file_relative_path(__file__, 'double_job.py')
    with pytest.raises(DagsterInvariantViolationError) as exc_info:
        loadable_targets_from_python_file(double_job_path)
    assert str(exc_info.value) == 'No repository and more than one job found in "double_job". If you load a file or module directly it must have only one job in scope. Found jobs defined in variables or decorated functions: [\'pipe_one\', \'pipe_two\'].'

def test_single_graph():
    if False:
        print('Hello World!')
    single_graph_path = file_relative_path(__file__, 'single_graph.py')
    loadable_targets = loadable_targets_from_python_file(single_graph_path)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == 'graph_one'
    repo_def = repository_def_from_pointer(CodePointer.from_python_file(single_graph_path, symbol, None))
    assert isinstance(repo_def, RepositoryDefinition)
    assert repo_def.get_job('graph_one')

def test_double_graph():
    if False:
        for i in range(10):
            print('nop')
    double_job_path = file_relative_path(__file__, 'double_graph.py')
    with pytest.raises(DagsterInvariantViolationError) as exc_info:
        loadable_targets_from_python_file(double_job_path)
    assert str(exc_info.value) == 'More than one graph found in "double_graph". If you load a file or module directly and it has no repositories, jobs, or pipelines in scope, it must have no more than one graph in scope. Found graphs defined in variables or decorated functions: [\'graph_one\', \'graph_two\'].'

def test_multiple_assets():
    if False:
        return 10
    path = file_relative_path(__file__, 'multiple_assets.py')
    loadable_targets = loadable_targets_from_python_file(path)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == LOAD_ALL_ASSETS
    repo_def = repository_def_from_pointer(CodePointer.from_python_file(path, symbol, None))
    assert isinstance(repo_def, RepositoryDefinition)
    the_job = repo_def.get_implicit_global_asset_job_def()
    assert len(the_job.graph.node_defs) == 2

def test_no_loadable_targets():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvariantViolationError) as exc_info:
        loadable_targets_from_python_file(file_relative_path(__file__, 'nada.py'))
    assert str(exc_info.value) == 'No repositories, jobs, pipelines, graphs, or asset definitions found in "nada".'

def test_single_pending_repository():
    if False:
        i = 10
        return i + 15
    single_pending_repo_path = file_relative_path(__file__, 'single_pending_repository.py')
    loadable_targets = loadable_targets_from_python_file(single_pending_repo_path)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == 'single_pending_repository'
    repo_def = CodePointer.from_python_file(single_pending_repo_path, symbol, None).load_target()
    assert isinstance(repo_def, PendingRepositoryDefinition)
    assert repo_def.name == 'single_pending_repository'

def test_single_repository_in_module():
    if False:
        for i in range(10):
            print('nop')
    loadable_targets = loadable_targets_from_python_module('dagster.utils.test.toys.single_repository', working_directory=None)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == 'single_repository'
    repo_def = CodePointer.from_module('dagster.utils.test.toys.single_repository', symbol, working_directory=None).load_target()
    assert isinstance(repo_def, RepositoryDefinition)
    assert repo_def.name == 'single_repository'

def test_single_repository_in_package():
    if False:
        i = 10
        return i + 15
    loadable_targets = loadable_targets_from_python_package('dagster.utils.test.toys.single_repository', working_directory=None)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == 'single_repository'
    repo_def = CodePointer.from_python_package('dagster.utils.test.toys.single_repository', symbol, working_directory=None).load_target()
    assert isinstance(repo_def, RepositoryDefinition)
    assert repo_def.name == 'single_repository'

def test_single_defs_in_file():
    if False:
        print('Hello World!')
    dagster_defs_path = file_relative_path(__file__, 'single_defs.py')
    loadable_targets = loadable_targets_from_python_file(dagster_defs_path)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == 'defs'
    repo_def = repository_def_from_pointer(CodePointer.from_python_file(dagster_defs_path, symbol, None))
    assert isinstance(repo_def, RepositoryDefinition)

def test_single_def_any_name():
    if False:
        for i in range(10):
            print('nop')
    dagster_defs_path = file_relative_path(__file__, 'single_defs_any_name.py')
    loadable_targets = loadable_targets_from_python_file(dagster_defs_path)
    assert len(loadable_targets) == 1
    symbol = loadable_targets[0].attribute
    assert symbol == 'not_defs'

def test_double_defs_in_file():
    if False:
        while True:
            i = 10
    dagster_defs_path = file_relative_path(__file__, 'double_defs.py')
    with pytest.raises(DagsterInvariantViolationError, match='Cannot have more than one Definitions object defined at module scope'):
        loadable_targets_from_python_file(dagster_defs_path)

def _current_test_directory_paths():
    if False:
        while True:
            i = 10
    return [os.path.dirname(__file__)]

def test_local_directory_module():
    if False:
        i = 10
        return i + 15
    assert not os.path.exists(file_relative_path(__file__, '__init__.py'))
    assert os.path.dirname(__file__) in sys.path
    if 'autodiscover_in_module' in sys.modules:
        del sys.modules['autodiscover_in_module']
    with pytest.raises(DagsterImportError):
        loadable_targets_from_python_module('complete_bogus_module', working_directory=os.path.dirname(__file__), remove_from_path_fn=_current_test_directory_paths)
    with pytest.raises(DagsterImportError):
        loadable_targets_from_python_module('autodiscover_in_module', working_directory=None, remove_from_path_fn=_current_test_directory_paths)
    loadable_targets = loadable_targets_from_python_module('autodiscover_in_module', working_directory=os.path.dirname(__file__), remove_from_path_fn=_current_test_directory_paths)
    assert len(loadable_targets) == 1

def test_local_directory_file():
    if False:
        while True:
            i = 10
    path = file_relative_path(__file__, 'autodiscover_file_in_directory/repository.py')
    with restore_sys_modules():
        with pytest.raises(DagsterImportError) as exc_info:
            loadable_targets_from_python_file(path)
        assert "No module named 'autodiscover_src'" in str(exc_info.value)
    with alter_sys_path(to_add=[os.path.dirname(path)], to_remove=[]):
        loadable_targets_from_python_file(path, working_directory=os.path.dirname(path))