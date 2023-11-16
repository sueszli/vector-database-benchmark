import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
import os
import datetime
import unittest
import datetime
import pymongo
import string
import random
import time
import re
import pprint
import pdb
import bson
from bson import ObjectId
from functools import reduce
from pymongo import ASCENDING
from mongodb_common import drop_all_collections, get_test_connection, ensure_environment_variables_set
import decimal
RECORD_COUNT = {}

def random_string_generator(size=6, chars=string.ascii_uppercase + string.digits):
    if False:
        while True:
            i = 10
    return ''.join((random.choice(chars) for x in range(size)))

def generate_simple_coll_docs(num_docs):
    if False:
        for i in range(10):
            print('nop')
    docs = []
    populated_string_fields = {f'string_field_{i}': random_string_generator() for i in range(1, 64)}
    for int_value in range(num_docs):
        docs.append({'int_field': int_value, **populated_string_fields})
    return docs

class MongoDBOplog(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ensure_environment_variables_set()
        with get_test_connection() as client:
            drop_all_collections(client)
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(50))
            client['simple_db']['simple_coll_2'].insert_many(generate_simple_coll_docs(100))
            for index in self.expected_string_fields():
                client['simple_db']['simple_coll_1'].create_index(index)
            self.index_info = client['simple_db']['simple_coll_1'].index_information()

    def expected_check_streams(self):
        if False:
            i = 10
            return i + 15
        return {'simple_db-simple_coll_1', 'simple_db-simple_coll_2'}

    def expected_pks(self):
        if False:
            print('Hello World!')
        return {'simple_coll_1': {'_id'}, 'simple_coll_2': {'_id'}}

    def expected_row_counts(self):
        if False:
            i = 10
            return i + 15
        return {'simple_coll_1': 50, 'simple_coll_2': 100}

    def expected_sync_streams(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_coll_1', 'simple_coll_2'}

    def name(self):
        if False:
            print('Hello World!')
        return 'tap_tester_mongodb_index'

    def tap_name(self):
        if False:
            i = 10
            return i + 15
        return 'tap-mongodb'

    def get_type(self):
        if False:
            print('Hello World!')
        return 'platform.mongodb'

    def get_credentials(self):
        if False:
            return 10
        return {'password': os.getenv('TAP_MONGODB_PASSWORD')}

    def get_properties(self):
        if False:
            for i in range(10):
                print('nop')
        return {'host': os.getenv('TAP_MONGODB_HOST'), 'port': os.getenv('TAP_MONGODB_PORT'), 'user': os.getenv('TAP_MONGODB_USER'), 'database': os.getenv('TAP_MONGODB_DBNAME')}

    def expected_string_fields(self):
        if False:
            return 10
        return {f'string_field_{i}' for i in range(1, 64)}

    def test_run(self):
        if False:
            i = 10
            return i + 15
        conn_id = connections.ensure_connection(self)
        check_job_name = runner.run_check_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, check_job_name)
        menagerie.verify_check_exit_status(self, exit_status, check_job_name)
        found_catalogs = menagerie.get_catalogs(conn_id)
        self.assertEqual(self.expected_check_streams(), {c['tap_stream_id'] for c in found_catalogs})
        for tap_stream_id in self.expected_check_streams():
            found_stream = [c for c in found_catalogs if c['tap_stream_id'] == tap_stream_id][0]
            self.assertEqual(self.expected_pks()[found_stream['stream_name']], set(found_stream.get('metadata', {}).get('table-key-properties')))
            self.assertEqual(self.expected_row_counts()[found_stream['stream_name']], found_stream.get('metadata', {}).get('row-count'))
        discovered_replication_keys = found_catalogs[0]['metadata']['valid-replication-keys']
        for field in self.expected_string_fields():
            self.assertIn(field, discovered_replication_keys)
        self.assertIn('_id', discovered_replication_keys)
        self.assertEqual(64, len(discovered_replication_keys))