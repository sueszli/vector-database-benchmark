from __future__ import annotations
import os
import unittest
from unittest.mock import MagicMock, patch
from ansible.inventory.manager import InventoryManager
from ansible.playbook.play import Play
from units.mock.loader import DictDataLoader
from units.mock.path import mock_unfrackpath_noop
from ansible.vars.manager import VariableManager

class TestVariableManager(unittest.TestCase):

    def test_basic_manager(self):
        if False:
            i = 10
            return i + 15
        fake_loader = DictDataLoader({})
        mock_inventory = MagicMock()
        v = VariableManager(loader=fake_loader, inventory=mock_inventory)
        variables = v.get_vars(use_cache=False)
        for (varname, value) in (('playbook_dir', os.path.abspath('.')),):
            self.assertEqual(variables[varname], value)

    def test_variable_manager_extra_vars(self):
        if False:
            i = 10
            return i + 15
        fake_loader = DictDataLoader({})
        extra_vars = dict(a=1, b=2, c=3)
        mock_inventory = MagicMock()
        v = VariableManager(loader=fake_loader, inventory=mock_inventory)
        v._extra_vars = extra_vars
        myvars = v.get_vars(use_cache=False)
        for (key, val) in extra_vars.items():
            self.assertEqual(myvars.get(key), val)

    def test_variable_manager_options_vars(self):
        if False:
            return 10
        fake_loader = DictDataLoader({})
        options_vars = dict(a=1, b=2, c=3)
        mock_inventory = MagicMock()
        v = VariableManager(loader=fake_loader, inventory=mock_inventory)
        v._extra_vars = options_vars
        myvars = v.get_vars(use_cache=False)
        for (key, val) in options_vars.items():
            self.assertEqual(myvars.get(key), val)

    def test_variable_manager_play_vars(self):
        if False:
            return 10
        fake_loader = DictDataLoader({})
        mock_play = MagicMock()
        mock_play.get_vars.return_value = dict(foo='bar')
        mock_play.get_roles.return_value = []
        mock_play.get_vars_files.return_value = []
        mock_inventory = MagicMock()
        v = VariableManager(loader=fake_loader, inventory=mock_inventory)
        self.assertEqual(v.get_vars(play=mock_play, use_cache=False).get('foo'), 'bar')

    def test_variable_manager_play_vars_files(self):
        if False:
            print('Hello World!')
        fake_loader = DictDataLoader({__file__: '\n               foo: bar\n            '})
        mock_play = MagicMock()
        mock_play.get_vars.return_value = dict()
        mock_play.get_roles.return_value = []
        mock_play.get_vars_files.return_value = [__file__]
        mock_inventory = MagicMock()
        v = VariableManager(inventory=mock_inventory, loader=fake_loader)
        self.assertEqual(v.get_vars(play=mock_play, use_cache=False).get('foo'), 'bar')

    def test_variable_manager_task_vars(self):
        if False:
            i = 10
            return i + 15
        return
        fake_loader = DictDataLoader({})
        mock_task = MagicMock()
        mock_task._role = None
        mock_task.loop = None
        mock_task.get_vars.return_value = dict(foo='bar')
        mock_task.get_include_params.return_value = dict()
        mock_all = MagicMock()
        mock_all.get_vars.return_value = {}
        mock_all.get_file_vars.return_value = {}
        mock_host = MagicMock()
        mock_host.get.name.return_value = 'test01'
        mock_host.get_vars.return_value = {}
        mock_host.get_host_vars.return_value = {}
        mock_inventory = MagicMock()
        mock_inventory.hosts.get.return_value = mock_host
        mock_inventory.hosts.get.name.return_value = 'test01'
        mock_inventory.get_host.return_value = mock_host
        mock_inventory.groups.__getitem__.return_value = mock_all
        v = VariableManager(loader=fake_loader, inventory=mock_inventory)
        self.assertEqual(v.get_vars(task=mock_task, use_cache=False).get('foo'), 'bar')

    @patch('ansible.playbook.role.definition.unfrackpath', mock_unfrackpath_noop)
    def test_variable_manager_precedence(self):
        if False:
            return 10
        return
        mock_inventory = MagicMock()
        inventory1_filedata = '\n            [group2:children]\n            group1\n\n            [group1]\n            host1 host_var=host_var_from_inventory_host1\n\n            [group1:vars]\n            group_var = group_var_from_inventory_group1\n\n            [group2:vars]\n            group_var = group_var_from_inventory_group2\n            '
        fake_loader = DictDataLoader({'/etc/ansible/inventory1': inventory1_filedata, '/etc/ansible/roles/defaults_only1/defaults/main.yml': '\n            default_var: "default_var_from_defaults_only1"\n            host_var: "host_var_from_defaults_only1"\n            group_var: "group_var_from_defaults_only1"\n            group_var_all: "group_var_all_from_defaults_only1"\n            extra_var: "extra_var_from_defaults_only1"\n            ', '/etc/ansible/roles/defaults_only1/tasks/main.yml': '\n            - debug: msg="here i am"\n            ', '/etc/ansible/roles/defaults_only2/defaults/main.yml': '\n            default_var: "default_var_from_defaults_only2"\n            host_var: "host_var_from_defaults_only2"\n            group_var: "group_var_from_defaults_only2"\n            group_var_all: "group_var_all_from_defaults_only2"\n            extra_var: "extra_var_from_defaults_only2"\n            '})
        inv1 = InventoryManager(loader=fake_loader, sources=['/etc/ansible/inventory1'])
        v = VariableManager(inventory=mock_inventory, loader=fake_loader)
        play1 = Play.load(dict(hosts=['all'], roles=['defaults_only1', 'defaults_only2']), loader=fake_loader, variable_manager=v)
        res = v.get_vars(play=play1)
        self.assertEqual(res['default_var'], 'default_var_from_defaults_only2')
        blocks = play1.compile()
        task = blocks[1].block[0]
        res = v.get_vars(play=play1, task=task)
        self.assertEqual(res['default_var'], 'default_var_from_defaults_only1')
        v.set_inventory(inv1)
        h1 = inv1.get_host('host1')
        res = v.get_vars(play=play1, host=h1)
        self.assertEqual(res['group_var'], 'group_var_from_inventory_group1')
        self.assertEqual(res['host_var'], 'host_var_from_inventory_host1')
        fake_loader.push('/etc/ansible/group_vars/all', '\n        group_var_all: group_var_all_from_group_vars_all\n        ')
        fake_loader.push('/etc/ansible/group_vars/group1', '\n        group_var: group_var_from_group_vars_group1\n        ')
        fake_loader.push('/etc/ansible/group_vars/group3', '\n        # this is a dummy, which should not be used anywhere\n        group_var: group_var_from_group_vars_group3\n        ')
        fake_loader.push('/etc/ansible/host_vars/host1', '\n        host_var: host_var_from_host_vars_host1\n        ')
        fake_loader.push('group_vars/group1', '\n        playbook_group_var: playbook_group_var\n        ')
        fake_loader.push('host_vars/host1', '\n        playbook_host_var: playbook_host_var\n        ')
        res = v.get_vars(play=play1, host=h1)
        v._fact_cache['host1'] = dict(fact_cache_var='fact_cache_var_from_fact_cache')
        res = v.get_vars(play=play1, host=h1)
        self.assertEqual(res['fact_cache_var'], 'fact_cache_var_from_fact_cache')

    @patch('ansible.playbook.role.definition.unfrackpath', mock_unfrackpath_noop)
    def test_variable_manager_role_vars_dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests vars from role dependencies with duplicate dependencies.\n        '
        mock_inventory = MagicMock()
        fake_loader = DictDataLoader({'/etc/ansible/roles/common-role/tasks/main.yml': '\n            - debug: msg="{{role_var}}"\n            ', '/etc/ansible/roles/role1/vars/main.yml': '\n            role_var: "role_var_from_role1"\n            ', '/etc/ansible/roles/role1/meta/main.yml': '\n            dependencies:\n              - { role: common-role }\n            ', '/etc/ansible/roles/role2/vars/main.yml': '\n            role_var: "role_var_from_role2"\n            ', '/etc/ansible/roles/role2/meta/main.yml': '\n            dependencies:\n              - { role: common-role }\n            '})
        v = VariableManager(loader=fake_loader, inventory=mock_inventory)
        play1 = Play.load(dict(hosts=['all'], roles=['role1', 'role2']), loader=fake_loader, variable_manager=v)
        blocks = play1.compile()
        task = blocks[1].block[0]
        res = v.get_vars(play=play1, task=task)
        self.assertEqual(res['role_var'], 'role_var_from_role1')
        task = blocks[2].block[0]
        res = v.get_vars(play=play1, task=task)
        self.assertEqual(res['role_var'], 'role_var_from_role2')