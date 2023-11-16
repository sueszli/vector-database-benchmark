from __future__ import annotations
import pickle
import unittest
from ansible.inventory.group import Group
from ansible.inventory.host import Host

class TestHost(unittest.TestCase):
    ansible_port = 22

    def setUp(self):
        if False:
            while True:
                i = 10
        self.hostA = Host('a')
        self.hostB = Host('b')

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.hostA, self.hostA)
        self.assertNotEqual(self.hostA, self.hostB)
        self.assertNotEqual(self.hostA, Host('a'))

    def test_hashability(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(hash(self.hostA), hash(Host('a')))

    def test_get_vars(self):
        if False:
            i = 10
            return i + 15
        host_vars = self.hostA.get_vars()
        self.assertIsInstance(host_vars, dict)

    def test_repr(self):
        if False:
            return 10
        host_repr = repr(self.hostA)
        self.assertIsInstance(host_repr, str)

    def test_add_group(self):
        if False:
            for i in range(10):
                print('nop')
        group = Group('some_group')
        group_len = len(self.hostA.groups)
        self.hostA.add_group(group)
        self.assertEqual(len(self.hostA.groups), group_len + 1)

    def test_get_groups(self):
        if False:
            i = 10
            return i + 15
        group = Group('some_group')
        self.hostA.add_group(group)
        groups = self.hostA.get_groups()
        self.assertEqual(len(groups), 1)
        for _group in groups:
            self.assertIsInstance(_group, Group)

    def test_equals_none(self):
        if False:
            i = 10
            return i + 15
        other = None
        assert not self.hostA == other
        assert not other == self.hostA
        assert self.hostA != other
        assert other != self.hostA
        self.assertNotEqual(self.hostA, other)

    def test_serialize(self):
        if False:
            for i in range(10):
                print('nop')
        group = Group('some_group')
        self.hostA.add_group(group)
        data = self.hostA.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_then_deserialize(self):
        if False:
            return 10
        group = Group('some_group')
        self.hostA.add_group(group)
        hostA_data = self.hostA.serialize()
        hostA_clone = Host()
        hostA_clone.deserialize(hostA_data)
        self.assertEqual(self.hostA, hostA_clone)

    def test_set_state(self):
        if False:
            for i in range(10):
                print('nop')
        group = Group('some_group')
        self.hostA.add_group(group)
        pickled_hostA = pickle.dumps(self.hostA)
        hostA_clone = pickle.loads(pickled_hostA)
        self.assertEqual(self.hostA, hostA_clone)

class TestHostWithPort(TestHost):
    ansible_port = 8822

    def setUp(self):
        if False:
            return 10
        self.hostA = Host(name='a', port=self.ansible_port)
        self.hostB = Host(name='b', port=self.ansible_port)

    def test_get_vars_ansible_port(self):
        if False:
            return 10
        host_vars = self.hostA.get_vars()
        self.assertEqual(host_vars['ansible_port'], self.ansible_port)