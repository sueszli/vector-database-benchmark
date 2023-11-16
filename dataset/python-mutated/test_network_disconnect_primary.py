from __future__ import annotations
import unittest
from mockupdb import Future, MockupDB, OpReply, going, wait_until
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.topology_description import TOPOLOGY_TYPE

class TestNetworkDisconnectPrimary(unittest.TestCase):

    def test_network_disconnect_primary(self):
        if False:
            return 10
        (primary, secondary) = (MockupDB(), MockupDB())
        for server in (primary, secondary):
            server.run()
            self.addCleanup(server.stop)
        hosts = [server.address_string for server in (primary, secondary)]
        primary_response = OpReply(ismaster=True, setName='rs', hosts=hosts, minWireVersion=2, maxWireVersion=6)
        primary.autoresponds('ismaster', primary_response)
        secondary.autoresponds('ismaster', ismaster=False, secondary=True, setName='rs', hosts=hosts, minWireVersion=2, maxWireVersion=6)
        client = MongoClient(primary.uri, replicaSet='rs')
        self.addCleanup(client.close)
        wait_until(lambda : client.primary == primary.address, 'discover primary')
        topology = client._topology
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetWithPrimary, topology.description.topology_type)
        with going(client.db.command, 'buildinfo'):
            primary.receives('buildinfo').ok()
        ismaster_future = Future()
        primary.autoresponds('ismaster', lambda r: r.ok(ismaster_future.result()))
        with self.assertRaises(ConnectionFailure):
            with going(client.db.command, 'buildinfo'):
                primary.receives('buildinfo').hangup()
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetNoPrimary, topology.description.topology_type)
        ismaster_future.set_result(primary_response)
        with going(client.db.command, 'buildinfo'):
            wait_until(lambda : client.primary == primary.address, 'rediscover primary')
            primary.receives('buildinfo').ok()
        self.assertEqual(TOPOLOGY_TYPE.ReplicaSetWithPrimary, topology.description.topology_type)
if __name__ == '__main__':
    unittest.main()