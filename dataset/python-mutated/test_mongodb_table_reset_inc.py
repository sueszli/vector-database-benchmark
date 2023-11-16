import os
import uuid
import decimal
import string
import bson
from datetime import datetime, timedelta
from unittest import TestCase
import pymongo
from tap_tester import connections, menagerie, runner
from mongodb_common import drop_all_collections, get_test_connection, ensure_environment_variables_set
RECORD_COUNT = {}
VALID_REPLICATION_TYPES = {'datetime', 'Int64', 'float', 'int', 'str', 'Timestamp', 'UUID'}

def z_string_generator(size=6):
    if False:
        i = 10
        return i + 15
    return 'z' * size

def generate_simple_coll_docs(num_docs):
    if False:
        return 10
    docs = []
    start_datetime = datetime(2018, 1, 1, 19, 29, 14, 578000)
    for int_value in range(num_docs):
        start_datetime = start_datetime + timedelta(days=5)
        docs.append({'int_field': int_value, 'string_field': z_string_generator(int_value), 'date_field': start_datetime, 'double_field': int_value + 1.00001, 'timestamp_field': bson.timestamp.Timestamp(int_value + 1565897157, 1), 'uuid_field': uuid.UUID('3e139ff5-d622-45c6-bf9e-1dfec7282{:03d}'.format(int_value)), '64_bit_int_field': 34359738368 + int_value})
    return docs

class MongoDBTableResetInc(TestCase):

    def key_names(self):
        if False:
            print('Hello World!')
        return ['int_field', 'string_field', 'date_field', 'timestamp_field', 'uuid_field', '64_bit_int_field', 'double_field']

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ensure_environment_variables_set()
        with get_test_connection() as client:
            drop_all_collections(client)
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(50))
            client['simple_db']['simple_coll_2'].insert_many(generate_simple_coll_docs(100))
            client['simple_db']['simple_coll_1'].create_index([('date_field', pymongo.ASCENDING)])
            client['simple_db']['simple_coll_2'].create_index([('date_field', pymongo.ASCENDING)])
            for key_name in self.key_names():
                client['simple_db']['simple_coll_{}'.format(key_name)].insert_many(generate_simple_coll_docs(50))
                client['simple_db']['simple_coll_{}'.format(key_name)].create_index([(key_name, pymongo.ASCENDING)])

    def expected_check_streams(self):
        if False:
            while True:
                i = 10
        return {'simple_db-simple_coll_1', 'simple_db-simple_coll_2', *['simple_db-simple_coll_{}'.format(k) for k in self.key_names()]}

    def expected_pks(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_coll_1': {'_id'}, 'simple_coll_2': {'_id'}, **{'simple_coll_{}'.format(k): {'_id'} for k in self.key_names()}}

    def expected_valid_replication_keys(self):
        if False:
            return 10
        return {'simple_coll_1': {'_id', 'date_field'}, 'simple_coll_2': {'_id', 'date_field'}, **{'simple_coll_{}'.format(k): {'_id', k} for k in self.key_names()}}

    def expected_row_counts(self):
        if False:
            while True:
                i = 10
        return {'simple_coll_1': 50, 'simple_coll_2': 100, **{'simple_coll_{}'.format(k): 50 for k in self.key_names()}}

    def expected_sync_streams(self):
        if False:
            return 10
        return {'simple_coll_1', 'simple_coll_2', *['simple_coll_{}'.format(k) for k in self.key_names()]}

    def name(self):
        if False:
            print('Hello World!')
        return 'tap_tester_mongodb_table_reset_inc'

    def tap_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'tap-mongodb'

    def get_type(self):
        if False:
            i = 10
            return i + 15
        return 'platform.mongodb'

    def get_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        return {'password': os.getenv('TAP_MONGODB_PASSWORD')}

    def get_properties(self):
        if False:
            for i in range(10):
                print('nop')
        return {'host': os.getenv('TAP_MONGODB_HOST'), 'port': os.getenv('TAP_MONGODB_PORT'), 'user': os.getenv('TAP_MONGODB_USER'), 'database': os.getenv('TAP_MONGODB_DBNAME')}

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        conn_id = connections.ensure_connection(self)
        check_job_name = runner.run_check_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, check_job_name)
        menagerie.verify_check_exit_status(self, exit_status, check_job_name)
        catalog = menagerie.get_catalog(conn_id)
        found_catalogs = menagerie.get_catalogs(conn_id)
        found_streams = {entry['tap_stream_id'] for entry in catalog['streams']}
        self.assertSetEqual(self.expected_check_streams(), found_streams)
        for tap_stream_id in self.expected_check_streams():
            with self.subTest(stream=tap_stream_id):
                stream = tap_stream_id.split('-')[1]
                expected_primary_key = self.expected_pks()[stream]
                expected_row_count = self.expected_row_counts()[stream]
                expected_replication_keys = self.expected_valid_replication_keys()[stream]
                found_stream = [entry for entry in catalog['streams'] if entry['tap_stream_id'] == tap_stream_id][0]
                stream_metadata = [entry['metadata'] for entry in found_stream['metadata'] if entry['breadcrumb'] == []][0]
                primary_key = set(stream_metadata.get('table-key-properties'))
                row_count = stream_metadata.get('row-count')
                replication_key = set(stream_metadata.get('valid-replication-keys'))
                self.assertSetEqual(expected_primary_key, primary_key)
                self.assertEqual(expected_row_count, row_count)
                self.assertSetEqual(replication_key, expected_replication_keys)
        for stream_catalog in found_catalogs:
            annotated_schema = menagerie.get_annotated_schema(conn_id, stream_catalog['stream_id'])
            rep_key = 'date_field'
            for key in self.key_names():
                if key in stream_catalog['stream_name']:
                    rep_key = key
            additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'INCREMENTAL', 'replication-key': rep_key}}]
            selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        messages_by_stream = runner.get_records_from_target_output()
        expected_schema = {'type': 'object'}
        for tap_stream_id in self.expected_sync_streams():
            with self.subTest(stream=tap_stream_id):
                persisted_schema = messages_by_stream[tap_stream_id]['schema']
                self.assertDictEqual(expected_schema, persisted_schema)
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        for tap_stream_id in self.expected_sync_streams():
            with self.subTest(stream=tap_stream_id):
                expected_row_count = self.expected_row_counts()[tap_stream_id]
                row_count = record_count_by_stream[tap_stream_id]
                self.assertEqual(expected_row_count, row_count)
        state = menagerie.get_state(conn_id)
        reset_stream = 'simple_db-simple_coll_int_field'
        state['bookmarks'].pop(reset_stream)
        menagerie.set_state(conn_id, state)
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        state = menagerie.get_state(conn_id)
        expected_state_keys = {'last_replication_method', 'replication_key_name', 'replication_key_type', 'replication_key_value', 'version'}
        for tap_stream_id in self.expected_check_streams():
            with self.subTest(stream=tap_stream_id):
                bookmark = state['bookmarks'][tap_stream_id]
                stream = tap_stream_id.split('-')[1]
                expected_replication_keys = self.expected_valid_replication_keys()[stream]
                replication_key = bookmark['replication_key_name']
                replication_key_type = bookmark['replication_key_type']
                self.assertSetEqual(expected_state_keys, set(bookmark.keys()))
                for key in expected_state_keys:
                    self.assertIsNotNone(bookmark[key])
                self.assertEqual('INCREMENTAL', bookmark['last_replication_method'])
                self.assertIn(replication_key, expected_replication_keys)
                self.assertIn(replication_key_type, VALID_REPLICATION_TYPES)
                self.assertIsNone(state['currently_syncing'])
        messages_by_stream = runner.get_records_from_target_output()
        records_by_stream = {}
        for stream_name in self.expected_sync_streams():
            records_by_stream[stream_name] = [x for x in messages_by_stream[stream_name]['messages'] if x.get('action') == 'upsert']
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        for (k, v) in record_count_by_stream.items():
            if k != 'simple_coll_int_field':
                self.assertEqual(1, v)
            if k == 'simple_coll_int_field':
                self.assertEqual(50, v)