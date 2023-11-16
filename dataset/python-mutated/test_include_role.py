from __future__ import annotations
import unittest
from unittest.mock import patch
from ansible.playbook import Play
from ansible.playbook.role_include import IncludeRole
from ansible.playbook.task import Task
from ansible.vars.manager import VariableManager
from units.mock.loader import DictDataLoader
from units.mock.path import mock_unfrackpath_noop

class TestIncludeRole(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.loader = DictDataLoader({'/etc/ansible/roles/l1/tasks/main.yml': "\n                - shell: echo 'hello world from l1'\n                - include_role: name=l2\n            ", '/etc/ansible/roles/l1/tasks/alt.yml': "\n                - shell: echo 'hello world from l1 alt'\n                - include_role: name=l2 tasks_from=alt defaults_from=alt\n            ", '/etc/ansible/roles/l1/defaults/main.yml': '\n                test_variable: l1-main\n                l1_variable: l1-main\n            ', '/etc/ansible/roles/l1/defaults/alt.yml': '\n                test_variable: l1-alt\n                l1_variable: l1-alt\n            ', '/etc/ansible/roles/l2/tasks/main.yml': "\n                - shell: echo 'hello world from l2'\n                - include_role: name=l3\n            ", '/etc/ansible/roles/l2/tasks/alt.yml': "\n                - shell: echo 'hello world from l2 alt'\n                - include_role: name=l3 tasks_from=alt defaults_from=alt\n            ", '/etc/ansible/roles/l2/defaults/main.yml': '\n                test_variable: l2-main\n                l2_variable: l2-main\n            ', '/etc/ansible/roles/l2/defaults/alt.yml': '\n                test_variable: l2-alt\n                l2_variable: l2-alt\n            ', '/etc/ansible/roles/l3/tasks/main.yml': "\n                - shell: echo 'hello world from l3'\n            ", '/etc/ansible/roles/l3/tasks/alt.yml': "\n                - shell: echo 'hello world from l3 alt'\n            ", '/etc/ansible/roles/l3/defaults/main.yml': '\n                test_variable: l3-main\n                l3_variable: l3-main\n            ', '/etc/ansible/roles/l3/defaults/alt.yml': '\n                test_variable: l3-alt\n                l3_variable: l3-alt\n            '})
        self.var_manager = VariableManager(loader=self.loader)

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def flatten_tasks(self, tasks):
        if False:
            for i in range(10):
                print('nop')
        for task in tasks:
            if isinstance(task, IncludeRole):
                (blocks, handlers) = task.get_block_list(loader=self.loader)
                for block in blocks:
                    for t in self.flatten_tasks(block.block):
                        yield t
            elif isinstance(task, Task):
                yield task
            else:
                for t in self.flatten_tasks(task.block):
                    yield t

    def get_tasks_vars(self, play, tasks):
        if False:
            return 10
        for task in self.flatten_tasks(tasks):
            if task.implicit:
                continue
            role = task._role
            yield (role.get_name(), self.var_manager.get_vars(play=play, task=task))

    @patch('ansible.playbook.role.definition.unfrackpath', mock_unfrackpath_noop)
    def test_simple(self):
        if False:
            return 10
        'Test one-level include with default tasks and variables'
        play = Play.load(dict(name='test play', hosts=['foo'], gather_facts=False, tasks=[{'include_role': 'name=l3'}]), loader=self.loader, variable_manager=self.var_manager)
        tasks = play.compile()
        tested = False
        for (role, task_vars) in self.get_tasks_vars(play, tasks):
            tested = True
            self.assertEqual(task_vars.get('l3_variable'), 'l3-main')
            self.assertEqual(task_vars.get('test_variable'), 'l3-main')
        self.assertTrue(tested)

    @patch('ansible.playbook.role.definition.unfrackpath', mock_unfrackpath_noop)
    def test_simple_alt_files(self):
        if False:
            return 10
        'Test one-level include with alternative tasks and variables'
        play = Play.load(dict(name='test play', hosts=['foo'], gather_facts=False, tasks=[{'include_role': 'name=l3 tasks_from=alt defaults_from=alt'}]), loader=self.loader, variable_manager=self.var_manager)
        tasks = play.compile()
        tested = False
        for (role, task_vars) in self.get_tasks_vars(play, tasks):
            tested = True
            self.assertEqual(task_vars.get('l3_variable'), 'l3-alt')
            self.assertEqual(task_vars.get('test_variable'), 'l3-alt')
        self.assertTrue(tested)

    @patch('ansible.playbook.role.definition.unfrackpath', mock_unfrackpath_noop)
    def test_nested(self):
        if False:
            while True:
                i = 10
        '\n        Test nested includes with default tasks and variables.\n\n        Variables from outer roles should be inherited, but overridden in inner\n        roles.\n        '
        play = Play.load(dict(name='test play', hosts=['foo'], gather_facts=False, tasks=[{'include_role': 'name=l1'}]), loader=self.loader, variable_manager=self.var_manager)
        tasks = play.compile()
        expected_roles = ['l1', 'l2', 'l3']
        for (role, task_vars) in self.get_tasks_vars(play, tasks):
            expected_roles.remove(role)
            if role == 'l1':
                self.assertEqual(task_vars.get('l1_variable'), 'l1-main')
                self.assertEqual(task_vars.get('l2_variable'), None)
                self.assertEqual(task_vars.get('l3_variable'), None)
                self.assertEqual(task_vars.get('test_variable'), 'l1-main')
            elif role == 'l2':
                self.assertEqual(task_vars.get('l1_variable'), 'l1-main')
                self.assertEqual(task_vars.get('l2_variable'), 'l2-main')
                self.assertEqual(task_vars.get('l3_variable'), None)
                self.assertEqual(task_vars.get('test_variable'), 'l2-main')
            elif role == 'l3':
                self.assertEqual(task_vars.get('l1_variable'), 'l1-main')
                self.assertEqual(task_vars.get('l2_variable'), 'l2-main')
                self.assertEqual(task_vars.get('l3_variable'), 'l3-main')
                self.assertEqual(task_vars.get('test_variable'), 'l3-main')
            else:
                self.fail()
        self.assertFalse(expected_roles)

    @patch('ansible.playbook.role.definition.unfrackpath', mock_unfrackpath_noop)
    def test_nested_alt_files(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test nested includes with alternative tasks and variables.\n\n        Variables from outer roles should be inherited, but overridden in inner\n        roles.\n        '
        play = Play.load(dict(name='test play', hosts=['foo'], gather_facts=False, tasks=[{'include_role': 'name=l1 tasks_from=alt defaults_from=alt'}]), loader=self.loader, variable_manager=self.var_manager)
        tasks = play.compile()
        expected_roles = ['l1', 'l2', 'l3']
        for (role, task_vars) in self.get_tasks_vars(play, tasks):
            expected_roles.remove(role)
            if role == 'l1':
                self.assertEqual(task_vars.get('l1_variable'), 'l1-alt')
                self.assertEqual(task_vars.get('l2_variable'), None)
                self.assertEqual(task_vars.get('l3_variable'), None)
                self.assertEqual(task_vars.get('test_variable'), 'l1-alt')
            elif role == 'l2':
                self.assertEqual(task_vars.get('l1_variable'), 'l1-alt')
                self.assertEqual(task_vars.get('l2_variable'), 'l2-alt')
                self.assertEqual(task_vars.get('l3_variable'), None)
                self.assertEqual(task_vars.get('test_variable'), 'l2-alt')
            elif role == 'l3':
                self.assertEqual(task_vars.get('l1_variable'), 'l1-alt')
                self.assertEqual(task_vars.get('l2_variable'), 'l2-alt')
                self.assertEqual(task_vars.get('l3_variable'), 'l3-alt')
                self.assertEqual(task_vars.get('test_variable'), 'l3-alt')
            else:
                self.fail()
        self.assertFalse(expected_roles)