from __future__ import annotations
import copy
import itertools
import unittest
from typing import Any
from mockupdb import CommandBase, MockupDB, going
from operations import operations
from pymongo import MongoClient, ReadPreference
from pymongo.read_preferences import _MONGOS_MODES, make_read_preference, read_pref_mode_from_name

class OpMsgReadPrefBase(unittest.TestCase):
    single_mongod = False
    primary: MockupDB
    secondary: MockupDB

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()

    @classmethod
    def add_test(cls, mode, test_name, test):
        if False:
            while True:
                i = 10
        setattr(cls, test_name, test)

    def setup_client(self, read_preference):
        if False:
            for i in range(10):
                print('nop')
        client = MongoClient(self.primary.uri, read_preference=read_preference)
        self.addCleanup(client.close)
        return client

class TestOpMsgMongos(OpMsgReadPrefBase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        auto_ismaster = {'ismaster': True, 'msg': 'isdbgrid', 'minWireVersion': 2, 'maxWireVersion': 6}
        cls.primary = MockupDB(auto_ismaster=auto_ismaster)
        cls.primary.run()
        cls.secondary = cls.primary

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.primary.stop()
        super().tearDownClass()

class TestOpMsgReplicaSet(OpMsgReadPrefBase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        (cls.primary, cls.secondary) = (MockupDB(), MockupDB())
        for server in (cls.primary, cls.secondary):
            server.run()
        hosts = [server.address_string for server in (cls.primary, cls.secondary)]
        primary_ismaster = {'ismaster': True, 'setName': 'rs', 'hosts': hosts, 'minWireVersion': 2, 'maxWireVersion': 6}
        cls.primary.autoresponds(CommandBase('ismaster'), primary_ismaster)
        secondary_ismaster = copy.copy(primary_ismaster)
        secondary_ismaster['ismaster'] = False
        secondary_ismaster['secondary'] = True
        cls.secondary.autoresponds(CommandBase('ismaster'), secondary_ismaster)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        for server in (cls.primary, cls.secondary):
            server.stop()
        super().tearDownClass()

    @classmethod
    def add_test(cls, mode, test_name, test):
        if False:
            i = 10
            return i + 15
        if mode != 'nearest':
            setattr(cls, test_name, test)

    def setup_client(self, read_preference):
        if False:
            for i in range(10):
                print('nop')
        client = MongoClient(self.primary.uri, replicaSet='rs', read_preference=read_preference)
        client.admin.command('ismaster', read_preference=ReadPreference.SECONDARY)
        self.addCleanup(client.close)
        return client

class TestOpMsgSingle(OpMsgReadPrefBase):
    single_mongod = True

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        auto_ismaster = {'ismaster': True, 'minWireVersion': 2, 'maxWireVersion': 6}
        cls.primary = MockupDB(auto_ismaster=auto_ismaster)
        cls.primary.run()
        cls.secondary = cls.primary

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        cls.primary.stop()
        super().tearDownClass()

def create_op_msg_read_mode_test(mode, operation):
    if False:
        for i in range(10):
            print('nop')

    def test(self):
        if False:
            while True:
                i = 10
        pref = make_read_preference(read_pref_mode_from_name(mode), tag_sets=None)
        client = self.setup_client(read_preference=pref)
        expected_pref: Any
        if operation.op_type == 'always-use-secondary':
            expected_server = self.secondary
            expected_pref = ReadPreference.SECONDARY
        elif operation.op_type == 'must-use-primary':
            expected_server = self.primary
            expected_pref = None
        elif operation.op_type == 'may-use-secondary':
            if mode == 'primary':
                expected_server = self.primary
                expected_pref = None
            elif mode == 'primaryPreferred':
                expected_server = self.primary
                expected_pref = pref
            else:
                expected_server = self.secondary
                expected_pref = pref
        else:
            self.fail('unrecognized op_type %r' % operation.op_type)
        if self.single_mongod:
            expected_pref = None
        with going(operation.function, client):
            request = expected_server.receive()
            request.reply(operation.reply)
        actual_pref = request.doc.get('$readPreference')
        if expected_pref:
            self.assertEqual(expected_pref.document, actual_pref)
        else:
            self.assertIsNone(actual_pref)
        self.assertNotIn('$query', request.doc)
    return test

def generate_op_msg_read_mode_tests():
    if False:
        while True:
            i = 10
    matrix = itertools.product(_MONGOS_MODES, operations)
    for entry in matrix:
        (mode, operation) = entry
        test = create_op_msg_read_mode_test(mode, operation)
        test_name = 'test_{}_with_mode_{}'.format(operation.name.replace(' ', '_'), mode)
        test.__name__ = test_name
        for cls in (TestOpMsgMongos, TestOpMsgReplicaSet, TestOpMsgSingle):
            cls.add_test(mode, test_name, test)
generate_op_msg_read_mode_tests()
if __name__ == '__main__':
    unittest.main()