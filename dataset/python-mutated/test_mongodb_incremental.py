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
        while True:
            i = 10
    return 'z' * size

def generate_simple_coll_docs(num_docs):
    if False:
        for i in range(10):
            print('nop')
    docs = []
    start_datetime = datetime(2018, 1, 1, 19, 29, 14, 578000)
    for int_value in range(num_docs):
        start_datetime = start_datetime + timedelta(days=5)
        docs.append({'int_field': int_value, 'string_field': z_string_generator(int_value), 'date_field': start_datetime, 'double_field': int_value + 1.00001, 'timestamp_field': bson.timestamp.Timestamp(int_value + 1565897157, 1), 'uuid_field': uuid.UUID('3e139ff5-d622-45c6-bf9e-1dfec7282{:03d}'.format(int_value)), '64_bit_int_field': 34359738368 + int_value})
    return docs

class MongoDBIncremental(TestCase):

    def key_names(self):
        if False:
            return 10
        return ['int_field', 'string_field', 'date_field', 'timestamp_field', 'uuid_field', '64_bit_int_field', 'double_field']

    def setUp(self):
        if False:
            print('Hello World!')
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
            return 10
        return {'simple_db-simple_coll_1', 'simple_db-simple_coll_2', *['simple_db-simple_coll_{}'.format(k) for k in self.key_names()]}

    def expected_pks(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_coll_1': {'_id'}, 'simple_coll_2': {'_id'}, **{'simple_coll_{}'.format(k): {'_id'} for k in self.key_names()}}

    def expected_valid_replication_keys(self):
        if False:
            print('Hello World!')
        return {'simple_coll_1': {'_id', 'date_field'}, 'simple_coll_2': {'_id', 'date_field'}, **{'simple_coll_{}'.format(k): {'_id', k} for k in self.key_names()}}

    def expected_row_counts(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_coll_1': 50, 'simple_coll_2': 100, **{'simple_coll_{}'.format(k): 50 for k in self.key_names()}}

    def expected_last_sync_row_counts(self):
        if False:
            i = 10
            return i + 15
        return {'simple_coll_1': 53, 'simple_coll_2': 102, **{'simple_coll_{}'.format(k): 1 for k in self.key_names()}}

    def expected_incremental_int_fields(self):
        if False:
            while True:
                i = 10
        return {'simple_coll_1': {49, 50, 51, 0}, 'simple_coll_2': {99, 100, 101, 0}, **{'simple_coll_{}'.format(k): {49, 50, 51, 0} for k in self.key_names() if k not in 'simple_coll_int_field'}, 'simple_coll_int_field': {49, 50, 51, 52}}

    def expected_sync_streams(self):
        if False:
            return 10
        return {'simple_coll_1', 'simple_coll_2', *['simple_coll_{}'.format(k) for k in self.key_names()]}

    def name(self):
        if False:
            return 10
        return 'tap_tester_mongodb_incremental'

    def tap_name(self):
        if False:
            return 10
        return 'tap-mongodb'

    def get_type(self):
        if False:
            return 10
        return 'platform.mongodb'

    def get_credentials(self):
        if False:
            while True:
                i = 10
        return {'password': os.getenv('TAP_MONGODB_PASSWORD')}

    def get_properties(self):
        if False:
            i = 10
            return i + 15
        return {'host': os.getenv('TAP_MONGODB_HOST'), 'port': os.getenv('TAP_MONGODB_PORT'), 'user': os.getenv('TAP_MONGODB_USER'), 'database': os.getenv('TAP_MONGODB_DBNAME')}

    def test_run(self):
        if False:
            while True:
                i = 10
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
        with get_test_connection() as client:
            update_doc_coll_1 = client['simple_db']['simple_coll_1'].find_one()
            client['simple_db']['simple_coll_1'].find_one_and_update({'_id': update_doc_coll_1['_id']}, {'$set': {'date_field': datetime(2020, 1, 1, 19, 29, 14, 578000)}})
            update_doc_coll_2 = client['simple_db']['simple_coll_2'].find_one()
            client['simple_db']['simple_coll_2'].find_one_and_update({'_id': update_doc_coll_2['_id']}, {'$set': {'date_field': datetime(2020, 1, 1, 19, 29, 14, 578000)}})
            for key_name in self.key_names():
                if key_name == 'int_field':
                    doc_to_update = client['simple_db']['simple_coll_{}'.format(key_name)].find_one(sort=[('{}'.format(key_name), -1)])
                    value = doc_to_update['{}'.format(key_name)]
                    int_based_coll = client['simple_db']['simple_coll_{}'.format(key_name)].find_one()
                    client['simple_db']['simple_coll_{}'.format(key_name)].find_one_and_update({'_id': int_based_coll['_id']}, {'$set': {'{}'.format(key_name): value + 3}})
                elif key_name == 'double_field':
                    doc_to_update = client['simple_db']['simple_coll_{}'.format(key_name)].find_one(sort=[('{}'.format(key_name), -1)])
                    value = doc_to_update['{}'.format(key_name)]
                    double_based_coll = client['simple_db']['simple_coll_{}'.format(key_name)].find_one()
                    client['simple_db']['simple_coll_{}'.format(key_name)].find_one_and_update({'_id': double_based_coll['_id']}, {'$set': {'{}'.format(key_name): value + 3}})
                elif key_name == '64_bit_int_field':
                    doc_to_update = client['simple_db']['simple_coll_{}'.format(key_name)].find_one(sort=[('{}'.format(key_name), -1)])
                    value = doc_to_update['{}'.format(key_name)]
                    bit64_int_based_coll = client['simple_db']['simple_coll_{}'.format(key_name)].find_one()
                    client['simple_db']['simple_coll_{}'.format(key_name)].find_one_and_update({'_id': bit64_int_based_coll['_id']}, {'$set': {'{}'.format(key_name): value + 3}})
                elif key_name == 'date_field':
                    date_based_coll = client['simple_db']['simple_coll_{}'.format(key_name)].find_one()
                    client['simple_db']['simple_coll_{}'.format(key_name)].find_one_and_update({'_id': date_based_coll['_id']}, {'$set': {'{}'.format(key_name): datetime(2021, 1, 1, 15, 30, 14, 222000)}})
                elif key_name == 'timestamp_field':
                    timestamp_based_coll = client['simple_db']['simple_coll_{}'.format(key_name)].find_one()
                    client['simple_db']['simple_coll_{}'.format(key_name)].find_one_and_update({'_id': timestamp_based_coll['_id']}, {'$set': {'{}'.format(key_name): bson.timestamp.Timestamp(1565897157 + 99, 1)}})
            client['simple_db']['simple_coll_1'].insert_one({'int_field': 50, 'string_field': z_string_generator(), 'date_field': datetime(2018, 9, 13, 19, 29, 14, 578000), 'double_field': 51.001, 'timestamp_field': bson.timestamp.Timestamp(1565897157 + 50, 1), 'uuid_field': uuid.UUID('3e139ff5-d622-45c6-bf9e-1dfec7282050'), '64_bit_int_field': 34359738368 + 50})
            client['simple_db']['simple_coll_1'].insert_one({'int_field': 51, 'string_field': z_string_generator(), 'date_field': datetime(2018, 9, 18, 19, 29, 14, 578000), 'double_field': 52.001, 'timestamp_field': bson.timestamp.Timestamp(1565897157 + 51, 1), 'uuid_field': uuid.UUID('3e139ff5-d622-45c6-bf9e-1dfec7282051'), '64_bit_int_field': 34359738368 + 51})
            client['simple_db']['simple_coll_2'].insert_one({'int_field': 100, 'string_field': z_string_generator(), 'date_field': datetime(2019, 5, 21, 19, 29, 14, 578000), 'double_field': 101.001, 'timestamp_field': bson.timestamp.Timestamp(1565897157 + 100, 1), 'uuid_field': uuid.UUID('3e139ff5-d622-45c6-bf9e-1dfec7282100'), '64_bit_int_field': 34359738368 + 100})
            client['simple_db']['simple_coll_2'].insert_one({'int_field': 101, 'string_field': z_string_generator(), 'date_field': datetime(2019, 5, 26, 19, 29, 14, 578000), 'double_field': 102.001, 'timestamp_field': bson.timestamp.Timestamp(1565897157 + 101, 1), 'uuid_field': uuid.UUID('3e139ff5-d622-45c6-bf9e-1dfec7282101'), '64_bit_int_field': 34359738368 + 101})
            for key_name in self.key_names():
                client['simple_db']['simple_coll_{}'.format(key_name)].insert_one({'int_field': 50, 'string_field': z_string_generator(50), 'date_field': datetime(2018, 9, 13, 19, 29, 15, 578000), 'double_field': 51.001, 'timestamp_field': bson.timestamp.Timestamp(1565897157 + 50, 1), 'uuid_field': uuid.UUID('3e139ff5-d622-45c6-bf9e-1dfec7282050'), '64_bit_int_field': 34359738368 + 50})
                client['simple_db']['simple_coll_{}'.format(key_name)].insert_one({'int_field': 51, 'string_field': z_string_generator(51), 'date_field': datetime(2018, 9, 18, 19, 29, 16, 578000), 'double_field': 52.001, 'timestamp_field': bson.timestamp.Timestamp(1565897157 + 51, 1), 'uuid_field': uuid.UUID('3e139ff5-d622-45c6-bf9e-1dfec7282051'), '64_bit_int_field': 34359738368 + 51})
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        messages_by_stream = runner.get_records_from_target_output()
        records_by_stream = {}
        for stream_name in self.expected_sync_streams():
            records_by_stream[stream_name] = [x for x in messages_by_stream[stream_name]['messages'] if x.get('action') == 'upsert']
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        for (k, v) in record_count_by_stream.items():
            if k not in ('simple_coll_uuid_field', 'simple_coll_string_field'):
                self.assertEqual(4, v)
        for stream_name in self.expected_sync_streams():
            if stream_name not in ('simple_coll_uuid_field', 'simple_coll_string_field'):
                actual = set([x['data']['int_field'] for x in records_by_stream[stream_name]])
                self.assertEqual(self.expected_incremental_int_fields()[stream_name], actual)
        no_rep_doc_coll_1 = client['simple_db']['simple_coll_1'].find_one({'int_field': 20})
        client['simple_db']['simple_coll_1'].find_one_and_update({'_id': no_rep_doc_coll_1['_id']}, {'$set': {'string_field': 'No_replication'}})
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        messages_by_stream = runner.get_records_from_target_output()
        second_state = menagerie.get_state(conn_id)
        records_by_stream = {}
        for stream_name in self.expected_sync_streams():
            records_by_stream[stream_name] = [x for x in messages_by_stream[stream_name]['messages'] if x.get('action') == 'upsert']
        doc_from_simple_coll_1 = records_by_stream['simple_coll_1']
        self.assertNotEqual(doc_from_simple_coll_1[0]['data']['_id'], no_rep_doc_coll_1['_id'])
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        for (k, v) in record_count_by_stream.items():
            if k not in ('simple_coll_uuid_field', 'simple_coll_string_field'):
                self.assertEqual(1, v)
        for stream_catalog in found_catalogs:
            annotated_schema = menagerie.get_annotated_schema(conn_id, stream_catalog['stream_id'])
            additional_md = []
            if stream_catalog['tap_stream_id'] == 'simple_db-simple_coll_1':
                additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'LOG_BASED'}}]
            elif stream_catalog['tap_stream_id'] == 'simple_db-simple_coll_2':
                additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'INCREMENTAL', 'replication-key': 'timestamp_field'}}]
            else:
                additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'INCREMENTAL', 'replication-key': stream_catalog['stream_name'].replace('simple_coll_', '')}}]
            selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        self.assertDictEqual(record_count_by_stream, self.expected_last_sync_row_counts())