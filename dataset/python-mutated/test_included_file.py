from __future__ import annotations
import os
import pytest
from unittest.mock import MagicMock
from units.mock.loader import DictDataLoader
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.playbook.task_include import TaskInclude
from ansible.playbook.role_include import IncludeRole
from ansible.executor import task_result
from ansible.playbook.included_file import IncludedFile
from ansible.errors import AnsibleParserError

@pytest.fixture
def mock_iterator():
    if False:
        i = 10
        return i + 15
    mock_iterator = MagicMock(name='MockIterator')
    mock_iterator._play = MagicMock(name='MockPlay')
    return mock_iterator

@pytest.fixture
def mock_variable_manager():
    if False:
        i = 10
        return i + 15
    mock_variable_manager = MagicMock(name='MockVariableManager')
    mock_variable_manager.get_vars.return_value = dict()
    return mock_variable_manager

def test_equals_ok():
    if False:
        while True:
            i = 10
    uuid = '111-111'
    parent = MagicMock(name='MockParent')
    parent._uuid = uuid
    task = MagicMock(name='MockTask')
    task._uuid = uuid
    task._parent = parent
    inc_a = IncludedFile('a.yml', {}, {}, task)
    inc_b = IncludedFile('a.yml', {}, {}, task)
    assert inc_a == inc_b

def test_equals_different_tasks():
    if False:
        return 10
    parent = MagicMock(name='MockParent')
    parent._uuid = '111-111'
    task_a = MagicMock(name='MockTask')
    task_a._uuid = '11-11'
    task_a._parent = parent
    task_b = MagicMock(name='MockTask')
    task_b._uuid = '22-22'
    task_b._parent = parent
    inc_a = IncludedFile('a.yml', {}, {}, task_a)
    inc_b = IncludedFile('a.yml', {}, {}, task_b)
    assert inc_a != inc_b

def test_equals_different_parents():
    if False:
        i = 10
        return i + 15
    parent_a = MagicMock(name='MockParent')
    parent_a._uuid = '111-111'
    parent_b = MagicMock(name='MockParent')
    parent_b._uuid = '222-222'
    task_a = MagicMock(name='MockTask')
    task_a._uuid = '11-11'
    task_a._parent = parent_a
    task_b = MagicMock(name='MockTask')
    task_b._uuid = '11-11'
    task_b._parent = parent_b
    inc_a = IncludedFile('a.yml', {}, {}, task_a)
    inc_b = IncludedFile('a.yml', {}, {}, task_b)
    assert inc_a != inc_b

def test_included_file_instantiation():
    if False:
        for i in range(10):
            print('nop')
    filename = 'somefile.yml'
    inc_file = IncludedFile(filename=filename, args={}, vars={}, task=None)
    assert isinstance(inc_file, IncludedFile)
    assert inc_file._filename == filename
    assert inc_file._args == {}
    assert inc_file._vars == {}
    assert inc_file._task is None

def test_process_include_tasks_results(mock_iterator, mock_variable_manager):
    if False:
        return 10
    hostname = 'testhost1'
    hostname2 = 'testhost2'
    parent_task_ds = {'debug': 'msg=foo'}
    parent_task = Task.load(parent_task_ds)
    parent_task._play = None
    task_ds = {'include_tasks': 'include_test.yml'}
    loaded_task = TaskInclude.load(task_ds, task_include=parent_task)
    return_data = {'include': 'include_test.yml'}
    result1 = task_result.TaskResult(host=hostname, task=loaded_task, return_data=return_data)
    result2 = task_result.TaskResult(host=hostname2, task=loaded_task, return_data=return_data)
    results = [result1, result2]
    fake_loader = DictDataLoader({'include_test.yml': ''})
    res = IncludedFile.process_include_results(results, mock_iterator, fake_loader, mock_variable_manager)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]._filename == os.path.join(os.getcwd(), 'include_test.yml')
    assert res[0]._hosts == ['testhost1', 'testhost2']
    assert res[0]._args == {}
    assert res[0]._vars == {}

def test_process_include_tasks_diff_files(mock_iterator, mock_variable_manager):
    if False:
        print('Hello World!')
    hostname = 'testhost1'
    hostname2 = 'testhost2'
    parent_task_ds = {'debug': 'msg=foo'}
    parent_task = Task.load(parent_task_ds)
    parent_task._play = None
    task_ds = {'include_tasks': 'include_test.yml'}
    loaded_task = TaskInclude.load(task_ds, task_include=parent_task)
    loaded_task._play = None
    child_task_ds = {'include_tasks': 'other_include_test.yml'}
    loaded_child_task = TaskInclude.load(child_task_ds, task_include=loaded_task)
    loaded_child_task._play = None
    return_data = {'include': 'include_test.yml'}
    result1 = task_result.TaskResult(host=hostname, task=loaded_task, return_data=return_data)
    return_data = {'include': 'other_include_test.yml'}
    result2 = task_result.TaskResult(host=hostname2, task=loaded_child_task, return_data=return_data)
    results = [result1, result2]
    fake_loader = DictDataLoader({'include_test.yml': '', 'other_include_test.yml': ''})
    res = IncludedFile.process_include_results(results, mock_iterator, fake_loader, mock_variable_manager)
    assert isinstance(res, list)
    assert res[0]._filename == os.path.join(os.getcwd(), 'include_test.yml')
    assert res[1]._filename == os.path.join(os.getcwd(), 'other_include_test.yml')
    assert res[0]._hosts == ['testhost1']
    assert res[1]._hosts == ['testhost2']
    assert res[0]._args == {}
    assert res[1]._args == {}
    assert res[0]._vars == {}
    assert res[1]._vars == {}

def test_process_include_tasks_simulate_free(mock_iterator, mock_variable_manager):
    if False:
        i = 10
        return i + 15
    hostname = 'testhost1'
    hostname2 = 'testhost2'
    parent_task_ds = {'debug': 'msg=foo'}
    parent_task1 = Task.load(parent_task_ds)
    parent_task2 = Task.load(parent_task_ds)
    parent_task1._play = None
    parent_task2._play = None
    task_ds = {'include_tasks': 'include_test.yml'}
    loaded_task1 = TaskInclude.load(task_ds, task_include=parent_task1)
    loaded_task2 = TaskInclude.load(task_ds, task_include=parent_task2)
    return_data = {'include': 'include_test.yml'}
    result1 = task_result.TaskResult(host=hostname, task=loaded_task1, return_data=return_data)
    result2 = task_result.TaskResult(host=hostname2, task=loaded_task2, return_data=return_data)
    results = [result1, result2]
    fake_loader = DictDataLoader({'include_test.yml': ''})
    res = IncludedFile.process_include_results(results, mock_iterator, fake_loader, mock_variable_manager)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0]._filename == os.path.join(os.getcwd(), 'include_test.yml')
    assert res[1]._filename == os.path.join(os.getcwd(), 'include_test.yml')
    assert res[0]._hosts == ['testhost1']
    assert res[1]._hosts == ['testhost2']
    assert res[0]._args == {}
    assert res[1]._args == {}
    assert res[0]._vars == {}
    assert res[1]._vars == {}

def test_process_include_simulate_free_block_role_tasks(mock_iterator, mock_variable_manager):
    if False:
        i = 10
        return i + 15
    'Test loading the same role returns different included files\n\n    In the case of free, we may end up with included files from roles that\n    have the same parent but are different tasks. Previously the comparison\n    for equality did not check if the tasks were the same and only checked\n    that the parents were the same. This lead to some tasks being run\n    incorrectly and some tasks being silient dropped.'
    fake_loader = DictDataLoader({'include_test.yml': '', '/etc/ansible/roles/foo_role/tasks/task1.yml': '\n            - debug: msg=task1\n        ', '/etc/ansible/roles/foo_role/tasks/task2.yml': '\n            - debug: msg=task2\n        '})
    hostname = 'testhost1'
    hostname2 = 'testhost2'
    role1_ds = {'name': 'task1 include', 'include_role': {'name': 'foo_role', 'tasks_from': 'task1.yml'}}
    role2_ds = {'name': 'task2 include', 'include_role': {'name': 'foo_role', 'tasks_from': 'task2.yml'}}
    parent_task_ds = {'block': [role1_ds, role2_ds]}
    parent_block = Block.load(parent_task_ds, loader=fake_loader)
    parent_block._play = None
    include_role1_ds = {'include_args': {'name': 'foo_role', 'tasks_from': 'task1.yml'}}
    include_role2_ds = {'include_args': {'name': 'foo_role', 'tasks_from': 'task2.yml'}}
    include_role1 = IncludeRole.load(role1_ds, block=parent_block, loader=fake_loader)
    include_role2 = IncludeRole.load(role2_ds, block=parent_block, loader=fake_loader)
    result1 = task_result.TaskResult(host=hostname, task=include_role1, return_data=include_role1_ds)
    result2 = task_result.TaskResult(host=hostname2, task=include_role2, return_data=include_role2_ds)
    results = [result1, result2]
    res = IncludedFile.process_include_results(results, mock_iterator, fake_loader, mock_variable_manager)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0]._filename == 'foo_role'
    assert res[1]._filename == 'foo_role'
    assert res[0]._task != res[1]._task
    assert res[0]._hosts == ['testhost1']
    assert res[1]._hosts == ['testhost2']
    assert res[0]._args == {}
    assert res[1]._args == {}
    assert res[0]._vars == {}
    assert res[1]._vars == {}

def test_empty_raw_params():
    if False:
        print('Hello World!')
    parent_task_ds = {'debug': 'msg=foo'}
    parent_task = Task.load(parent_task_ds)
    parent_task._play = None
    task_ds_list = [{'include': ''}, {'include_tasks': ''}, {'import_tasks': ''}]
    for task_ds in task_ds_list:
        with pytest.raises(AnsibleParserError):
            TaskInclude.load(task_ds, task_include=parent_task)