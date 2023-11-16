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
import singer
from functools import reduce
from singer import utils, metadata
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
        return 10
    docs = []
    for int_value in range(num_docs):
        docs.append({'int_field': int_value, 'string_field': random_string_generator()})
    return docs

class MongoDBOplogAgedOut(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ensure_environment_variables_set()
        with get_test_connection() as client:
            drop_all_collections(client)
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(50))

    def expected_check_streams(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_db-simple_coll_1'}

    def expected_pks(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_coll_1': {'_id'}}

    def expected_row_counts(self):
        if False:
            while True:
                i = 10
        return {'simple_coll_1': 50}

    def expected_sync_streams(self):
        if False:
            while True:
                i = 10
        return {'simple_coll_1'}

    def name(self):
        if False:
            return 10
        return 'tap_tester_mongodb_oplog_aged_out'

    def tap_name(self):
        if False:
            while True:
                i = 10
        return 'tap-mongodb'

    def get_type(self):
        if False:
            while True:
                i = 10
        return 'platform.mongodb'

    def get_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        return {'password': os.getenv('TAP_MONGODB_PASSWORD')}

    def get_properties(self):
        if False:
            while True:
                i = 10
        return {'host': os.getenv('TAP_MONGODB_HOST'), 'port': os.getenv('TAP_MONGODB_PORT'), 'user': os.getenv('TAP_MONGODB_USER'), 'database': os.getenv('TAP_MONGODB_DBNAME')}

    def test_run(self):
        if False:
            return 10
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
        for stream_catalog in found_catalogs:
            annotated_schema = menagerie.get_annotated_schema(conn_id, stream_catalog['stream_id'])
            additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'LOG_BASED'}}]
            selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
        original_version = int(time.time() * 1000)
        interrupted_state = {'currently_syncing': 'simple_db-simple_coll_1', 'bookmarks': {'simple_db-simple_coll_1': {'version': original_version, 'initial_full_table_complete': True, 'oplog_ts_time': 1, 'oplog_ts_inc': 0}}}
        menagerie.set_state(conn_id, interrupted_state)
        runner.run_sync_mode(self, conn_id)
        records_by_stream = runner.get_records_from_target_output()
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        for stream_name in self.expected_sync_streams():
            self.assertEqual('activate_version', records_by_stream[stream_name]['messages'][0]['action'])
            self.assertEqual('activate_version', records_by_stream[stream_name]['messages'][51]['action'])
        final_state = menagerie.get_state(conn_id)
        self.assertNotEqual(original_version, final_state.get('bookmarks', {}).get('simple_db-simple_coll_1', {}).get('version'))
        for (stream_id, row_count) in self.expected_row_counts().items():
            self.assertGreaterEqual(record_count_by_stream[stream_id], row_count)
        self.assertIsNotNone(final_state.get('bookmarks', {}).get('simple_db-simple_coll_1', {}).get('oplog_ts_time'))
        self.assertIsNotNone(final_state.get('bookmarks', {}).get('simple_db-simple_coll_1', {}).get('oplog_ts_inc'))
        self.assertTrue(final_state.get('bookmarks', {}).get('simple_db-simple_coll_1', {}).get('initial_full_table_complete'))