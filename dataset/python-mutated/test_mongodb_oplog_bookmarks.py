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
        i = 10
        return i + 15
    docs = []
    for int_value in range(num_docs):
        docs.append({'int_field': int_value, 'string_field': random_string_generator()})
    return docs

class MongoDBOplogBookmarks(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ensure_environment_variables_set()
        with get_test_connection() as client:
            drop_all_collections(client)
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(50))
            client['simple_db']['simple_coll_2'].insert_many(generate_simple_coll_docs(100))

    def expected_check_streams(self):
        if False:
            i = 10
            return i + 15
        return {'simple_db-simple_coll_1', 'simple_db-simple_coll_2'}

    def expected_pks(self):
        if False:
            while True:
                i = 10
        return {'simple_coll_1': {'_id'}, 'simple_coll_2': {'_id'}}

    def expected_row_counts(self):
        if False:
            return 10
        return {'simple_coll_1': 50, 'simple_coll_2': 100}

    def expected_sync_streams(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_coll_1', 'simple_coll_2'}

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'tap_tester_mongodb_oplog_bookmarks'

    def tap_name(self):
        if False:
            while True:
                i = 10
        return 'tap-mongodb'

    def get_type(self):
        if False:
            return 10
        return 'platform.mongodb'

    def get_credentials(self):
        if False:
            i = 10
            return i + 15
        return {'password': os.getenv('TAP_MONGODB_PASSWORD')}

    def get_properties(self):
        if False:
            for i in range(10):
                print('nop')
        return {'host': os.getenv('TAP_MONGODB_HOST'), 'port': os.getenv('TAP_MONGODB_PORT'), 'user': os.getenv('TAP_MONGODB_USER'), 'database': os.getenv('TAP_MONGODB_DBNAME')}

    def test_run(self):
        if False:
            print('Hello World!')
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
        additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'LOG_BASED'}}]
        for stream_catalog in found_catalogs:
            if stream_catalog['tap_stream_id'] == 'simple_db-simple_coll_1':
                annotated_schema = menagerie.get_annotated_schema(conn_id, stream_catalog['stream_id'])
                selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        records_by_stream = runner.get_records_from_target_output()
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        tap_stream_id = 'simple_db-simple_coll_1'
        self.assertGreaterEqual(record_count_by_stream['simple_coll_1'], self.expected_row_counts()['simple_coll_1'])
        state = menagerie.get_state(conn_id)
        first_versions = {}
        self.assertTrue(state['bookmarks'][tap_stream_id]['initial_full_table_complete'])
        first_versions[tap_stream_id] = state['bookmarks'][tap_stream_id]['version']
        self.assertIsNotNone(first_versions[tap_stream_id])
        self.assertIsNotNone(state['bookmarks'][tap_stream_id]['oplog_ts_time'])
        self.assertIsNotNone(state['bookmarks'][tap_stream_id]['oplog_ts_inc'])
        with get_test_connection() as client:
            client['simple_db']['simple_coll_1'].insert_one({'int_field': 101, 'string_field': random_string_generator()})
        sync_job_name = runner.run_sync_mode(self, conn_id)
        changed_ids = set()
        with get_test_connection() as client:
            changed_ids.add(client['simple_db']['simple_coll_2'].find({'int_field': 0})[0]['_id'])
            client['simple_db']['simple_coll_2'].delete_one({'int_field': 0})
            changed_ids.add(client['simple_db']['simple_coll_2'].find({'int_field': 1})[0]['_id'])
            client['simple_db']['simple_coll_2'].delete_one({'int_field': 1})
            changed_ids.add(client['simple_db']['simple_coll_2'].find({'int_field': 98})[0]['_id'])
            client['simple_db']['simple_coll_2'].update_one({'int_field': 98}, {'$set': {'int_field': -1}})
            changed_ids.add(client['simple_db']['simple_coll_2'].find({'int_field': 99})[0]['_id'])
            client['simple_db']['simple_coll_2'].update_one({'int_field': 99}, {'$set': {'int_field': -1}})
            client['simple_db']['simple_coll_2'].insert_one({'int_field': 100, 'string_field': random_string_generator()})
            changed_ids.add(client['simple_db']['simple_coll_2'].find({'int_field': 100})[0]['_id'])
            client['simple_db']['simple_coll_2'].insert_one({'int_field': 101, 'string_field': random_string_generator()})
            changed_ids.add(client['simple_db']['simple_coll_2'].find({'int_field': 101})[0]['_id'])
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        messages_by_stream = runner.get_records_from_target_output()
        records_by_stream = {'simple_coll_1': [x for x in messages_by_stream['simple_coll_1']['messages'] if x.get('action') == 'upsert']}
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        self.assertEqual(1, record_count_by_stream['simple_coll_1'])
        final_state = menagerie.get_state(conn_id)
        with get_test_connection() as client:
            row = client.local.oplog.rs.find_one(sort=[('$natural', pymongo.DESCENDING)])
            latest_oplog_ts = row.get('ts')
        self.assertEqual((latest_oplog_ts.time, latest_oplog_ts.inc), (final_state['bookmarks']['simple_db-simple_coll_1']['oplog_ts_time'], final_state['bookmarks']['simple_db-simple_coll_1']['oplog_ts_inc']))