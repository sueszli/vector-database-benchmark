"""Test $clusterTime handling."""
from __future__ import annotations
import unittest
from mockupdb import MockupDB, going
from bson import Timestamp
from pymongo import DeleteMany, InsertOne, MongoClient, UpdateOne

class TestClusterTime(unittest.TestCase):

    def cluster_time_conversation(self, callback, replies):
        if False:
            print('Hello World!')
        cluster_time = Timestamp(0, 0)
        server = MockupDB()
        _ = server.autoresponds('ismaster', {'minWireVersion': 0, 'maxWireVersion': 6, '$clusterTime': {'clusterTime': cluster_time}})
        server.run()
        self.addCleanup(server.stop)
        client = MongoClient(server.uri)
        self.addCleanup(client.close)
        with going(callback, client):
            for reply in replies:
                request = server.receives()
                self.assertIn('$clusterTime', request)
                self.assertEqual(request['$clusterTime']['clusterTime'], cluster_time)
                cluster_time = Timestamp(cluster_time.time, cluster_time.inc + 1)
                reply['$clusterTime'] = {'clusterTime': cluster_time}
                request.reply(reply)

    def test_command(self):
        if False:
            while True:
                i = 10

        def callback(client):
            if False:
                return 10
            client.db.command('ping')
            client.db.command('ping')
        self.cluster_time_conversation(callback, [{'ok': 1}] * 2)

    def test_bulk(self):
        if False:
            i = 10
            return i + 15

        def callback(client: MongoClient[dict]) -> None:
            if False:
                print('Hello World!')
            client.db.collection.bulk_write([InsertOne({}), InsertOne({}), UpdateOne({}, {'$inc': {'x': 1}}), DeleteMany({})])
        self.cluster_time_conversation(callback, [{'ok': 1, 'nInserted': 2}, {'ok': 1, 'nModified': 1}, {'ok': 1, 'nDeleted': 2}])
    batches = [{'cursor': {'id': 123, 'firstBatch': [{'a': 1}]}}, {'cursor': {'id': 123, 'nextBatch': [{'a': 2}]}}, {'cursor': {'id': 0, 'nextBatch': [{'a': 3}]}}]

    def test_cursor(self):
        if False:
            while True:
                i = 10

        def callback(client):
            if False:
                while True:
                    i = 10
            list(client.db.collection.find())
        self.cluster_time_conversation(callback, self.batches)

    def test_aggregate(self):
        if False:
            while True:
                i = 10

        def callback(client):
            if False:
                return 10
            list(client.db.collection.aggregate([]))
        self.cluster_time_conversation(callback, self.batches)

    def test_explain(self):
        if False:
            return 10

        def callback(client):
            if False:
                return 10
            client.db.collection.find().explain()
        self.cluster_time_conversation(callback, [{'ok': 1}])

    def test_monitor(self):
        if False:
            while True:
                i = 10
        cluster_time = Timestamp(0, 0)
        reply = {'minWireVersion': 0, 'maxWireVersion': 6, '$clusterTime': {'clusterTime': cluster_time}}
        server = MockupDB()
        server.run()
        self.addCleanup(server.stop)
        client = MongoClient(server.uri, heartbeatFrequencyMS=500)
        self.addCleanup(client.close)
        request = server.receives('ismaster')
        self.assertNotIn('$clusterTime', request)
        request.ok(reply)
        request = server.receives('ismaster')
        self.assertIn('$clusterTime', request)
        self.assertEqual(request['$clusterTime']['clusterTime'], cluster_time)
        cluster_time = Timestamp(cluster_time.time, cluster_time.inc + 1)
        reply['$clusterTime'] = {'clusterTime': cluster_time}
        request.reply(reply)
        request = server.receives('ismaster')
        self.assertEqual(request['$clusterTime']['clusterTime'], cluster_time)
        cluster_time = Timestamp(cluster_time.time, cluster_time.inc + 1)
        error = {'ok': 0, 'code': 211, 'errmsg': 'Cache Reader No keys found for HMAC ...', '$clusterTime': {'clusterTime': cluster_time}}
        request.reply(error)
        request = server.receives('ismaster')
        self.assertNotIn('$clusterTime', request)
        reply.pop('$clusterTime')
        request.reply(reply)
        request = server.receives('ismaster')
        self.assertEqual(request['$clusterTime']['clusterTime'], cluster_time)
        request.reply(reply)
        client.close()
if __name__ == '__main__':
    unittest.main()