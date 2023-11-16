"""Test the server_description module."""
from __future__ import annotations
import sys
sys.path[0:0] = ['']
from test import unittest
from bson.int64 import Int64
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.server_description import ServerDescription
from pymongo.server_type import SERVER_TYPE
address = ('localhost', 27017)

def parse_hello_response(doc):
    if False:
        i = 10
        return i + 15
    hello_response = Hello(doc)
    return ServerDescription(address, hello_response)

class TestServerDescription(unittest.TestCase):

    def test_unknown(self):
        if False:
            while True:
                i = 10
        s = ServerDescription(address)
        self.assertEqual(SERVER_TYPE.Unknown, s.server_type)
        self.assertFalse(s.is_writable)
        self.assertFalse(s.is_readable)

    def test_mongos(self):
        if False:
            print('Hello World!')
        s = parse_hello_response({'ok': 1, 'msg': 'isdbgrid'})
        self.assertEqual(SERVER_TYPE.Mongos, s.server_type)
        self.assertEqual('Mongos', s.server_type_name)
        self.assertTrue(s.is_writable)
        self.assertTrue(s.is_readable)

    def test_primary(self):
        if False:
            while True:
                i = 10
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs'})
        self.assertEqual(SERVER_TYPE.RSPrimary, s.server_type)
        self.assertEqual('RSPrimary', s.server_type_name)
        self.assertTrue(s.is_writable)
        self.assertTrue(s.is_readable)

    def test_secondary(self):
        if False:
            return 10
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'setName': 'rs'})
        self.assertEqual(SERVER_TYPE.RSSecondary, s.server_type)
        self.assertEqual('RSSecondary', s.server_type_name)
        self.assertFalse(s.is_writable)
        self.assertTrue(s.is_readable)

    def test_arbiter(self):
        if False:
            for i in range(10):
                print('nop')
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: False, 'arbiterOnly': True, 'setName': 'rs'})
        self.assertEqual(SERVER_TYPE.RSArbiter, s.server_type)
        self.assertEqual('RSArbiter', s.server_type_name)
        self.assertFalse(s.is_writable)
        self.assertFalse(s.is_readable)

    def test_other(self):
        if False:
            return 10
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: False, 'setName': 'rs'})
        self.assertEqual(SERVER_TYPE.RSOther, s.server_type)
        self.assertEqual('RSOther', s.server_type_name)
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'hidden': True, 'setName': 'rs'})
        self.assertEqual(SERVER_TYPE.RSOther, s.server_type)
        self.assertFalse(s.is_writable)
        self.assertFalse(s.is_readable)

    def test_ghost(self):
        if False:
            return 10
        s = parse_hello_response({'ok': 1, 'isreplicaset': True})
        self.assertEqual(SERVER_TYPE.RSGhost, s.server_type)
        self.assertEqual('RSGhost', s.server_type_name)
        self.assertFalse(s.is_writable)
        self.assertFalse(s.is_readable)

    def test_fields(self):
        if False:
            while True:
                i = 10
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: False, 'secondary': True, 'primary': 'a:27017', 'tags': {'a': 'foo', 'b': 'baz'}, 'maxMessageSizeBytes': 1, 'maxBsonObjectSize': 2, 'maxWriteBatchSize': 3, 'minWireVersion': 4, 'maxWireVersion': 5, 'setName': 'rs'})
        self.assertEqual(SERVER_TYPE.RSSecondary, s.server_type)
        self.assertEqual(('a', 27017), s.primary)
        self.assertEqual({'a': 'foo', 'b': 'baz'}, s.tags)
        self.assertEqual(1, s.max_message_size)
        self.assertEqual(2, s.max_bson_size)
        self.assertEqual(3, s.max_write_batch_size)
        self.assertEqual(4, s.min_wire_version)
        self.assertEqual(5, s.max_wire_version)

    def test_default_max_message_size(self):
        if False:
            return 10
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: True, 'maxBsonObjectSize': 2})
        self.assertEqual(4, s.max_message_size)

    def test_standalone(self):
        if False:
            i = 10
            return i + 15
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: True})
        self.assertEqual(SERVER_TYPE.Standalone, s.server_type)
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: False})
        self.assertEqual(SERVER_TYPE.Standalone, s.server_type)
        self.assertTrue(s.is_writable)
        self.assertTrue(s.is_readable)

    def test_ok_false(self):
        if False:
            return 10
        s = parse_hello_response({'ok': 0, HelloCompat.LEGACY_CMD: True})
        self.assertEqual(SERVER_TYPE.Unknown, s.server_type)
        self.assertFalse(s.is_writable)
        self.assertFalse(s.is_readable)

    def test_all_hosts(self):
        if False:
            return 10
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: True, 'hosts': ['a'], 'passives': ['b:27018'], 'arbiters': ['c']})
        self.assertEqual([('a', 27017), ('b', 27018), ('c', 27017)], sorted(s.all_hosts))

    def test_repr(self):
        if False:
            return 10
        s = parse_hello_response({'ok': 1, 'msg': 'isdbgrid'})
        self.assertEqual(repr(s), "<ServerDescription ('localhost', 27017) server_type: Mongos, rtt: None>")

    def test_topology_version(self):
        if False:
            for i in range(10):
                print('nop')
        topology_version = {'processId': ObjectId(), 'counter': Int64('0')}
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs', 'topologyVersion': topology_version})
        self.assertEqual(SERVER_TYPE.RSPrimary, s.server_type)
        self.assertEqual(topology_version, s.topology_version)
        s_unknown = s.to_unknown()
        self.assertEqual(SERVER_TYPE.Unknown, s_unknown.server_type)
        self.assertEqual(topology_version, s_unknown.topology_version)

    def test_topology_version_not_present(self):
        if False:
            while True:
                i = 10
        s = parse_hello_response({'ok': 1, HelloCompat.LEGACY_CMD: True, 'setName': 'rs'})
        self.assertEqual(SERVER_TYPE.RSPrimary, s.server_type)
        self.assertEqual(None, s.topology_version)
if __name__ == '__main__':
    unittest.main()