"""Test the CSOT unified spec tests."""
from __future__ import annotations
import os
import sys
sys.path[0:0] = ['']
from test import IntegrationTest, client_context, unittest
from test.unified_format import generate_test_classes
import pymongo
from pymongo import _csot
from pymongo.errors import PyMongoError
TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'csot')
globals().update(generate_test_classes(TEST_PATH, module=__name__))

class TestCSOT(IntegrationTest):
    RUN_ON_SERVERLESS = True
    RUN_ON_LOAD_BALANCER = True

    def test_timeout_nested(self):
        if False:
            return 10
        coll = self.db.coll
        self.assertEqual(_csot.get_timeout(), None)
        self.assertEqual(_csot.get_deadline(), float('inf'))
        self.assertEqual(_csot.get_rtt(), 0.0)
        with pymongo.timeout(10):
            coll.find_one()
            self.assertEqual(_csot.get_timeout(), 10)
            deadline_10 = _csot.get_deadline()
            with pymongo.timeout(15):
                coll.find_one()
                self.assertEqual(_csot.get_timeout(), 15)
                self.assertEqual(_csot.get_deadline(), deadline_10)
            self.assertEqual(_csot.get_timeout(), 10)
            self.assertEqual(_csot.get_deadline(), deadline_10)
            coll.find_one()
            with pymongo.timeout(5):
                coll.find_one()
                self.assertEqual(_csot.get_timeout(), 5)
                self.assertLess(_csot.get_deadline(), deadline_10)
            self.assertEqual(_csot.get_timeout(), 10)
            self.assertEqual(_csot.get_deadline(), deadline_10)
            coll.find_one()
        self.assertEqual(_csot.get_timeout(), None)
        self.assertEqual(_csot.get_deadline(), float('inf'))
        self.assertEqual(_csot.get_rtt(), 0.0)

    @client_context.require_change_streams
    def test_change_stream_can_resume_after_timeouts(self):
        if False:
            for i in range(10):
                print('nop')
        coll = self.db.test
        with coll.watch() as stream:
            with pymongo.timeout(0.1):
                with self.assertRaises(PyMongoError) as ctx:
                    stream.next()
                self.assertTrue(ctx.exception.timeout)
                self.assertTrue(stream.alive)
                with self.assertRaises(PyMongoError) as ctx:
                    stream.try_next()
                self.assertTrue(ctx.exception.timeout)
                self.assertTrue(stream.alive)
            if client_context.version < (4, 0):
                stream.try_next()
            coll.insert_one({})
            with pymongo.timeout(10):
                self.assertTrue(stream.next())
            self.assertTrue(stream.alive)
            with pymongo.timeout(0.5):
                with self.assertRaises(PyMongoError) as ctx:
                    stream.next()
                self.assertTrue(ctx.exception.timeout)
            self.assertTrue(stream.alive)
        self.assertFalse(stream.alive)
if __name__ == '__main__':
    unittest.main()