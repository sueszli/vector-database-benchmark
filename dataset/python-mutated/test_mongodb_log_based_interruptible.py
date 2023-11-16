import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
import os
import unittest
import pymongo
import string
import random
import time
from mongodb_common import drop_all_collections, get_test_connection, ensure_environment_variables_set
import copy
RECORD_COUNT = {}

def random_string_generator(size=6, chars=string.ascii_uppercase + string.digits):
    if False:
        for i in range(10):
            print('nop')
    return ''.join((random.choice(chars) for x in range(size)))

def generate_simple_coll_docs(num_docs):
    if False:
        print('Hello World!')
    docs = []
    for int_value in range(num_docs):
        docs.append({'int_field': int_value, 'string_field': random_string_generator()})
    return docs
table_interrupted = 'simple_coll_2'

class MongoDBLogBasedInterruptible(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ensure_environment_variables_set()
        with get_test_connection() as client:
            drop_all_collections(client)
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(10))
            for i in range(20):
                client['simple_db']['simple_coll_2'].insert_many(generate_simple_coll_docs(1))

    def expected_check_streams(self):
        if False:
            while True:
                i = 10
        return {'simple_db-simple_coll_1', 'simple_db-simple_coll_2'}

    def expected_pks(self):
        if False:
            i = 10
            return i + 15
        return {'simple_coll_1': {'_id'}, 'simple_coll_2': {'_id'}}

    def expected_row_count_1(self):
        if False:
            print('Hello World!')
        return {'simple_coll_1': 10, 'simple_coll_2': 20}

    def expected_row_count_2(self):
        if False:
            print('Hello World!')
        return {'simple_coll_1': 3}

    def expected_row_count_3(self):
        if False:
            return 10
        return {'simple_coll_1': 0, 'simple_coll_2': 0}

    def expected_sync_streams(self):
        if False:
            i = 10
            return i + 15
        return {'simple_coll_1', 'simple_coll_2'}

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'tap_tester_mongodb_log_based_interruptible'

    def tap_name(self):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        return {'host': os.getenv('TAP_MONGODB_HOST'), 'port': os.getenv('TAP_MONGODB_PORT'), 'user': os.getenv('TAP_MONGODB_USER'), 'database': os.getenv('TAP_MONGODB_DBNAME')}

    @unittest.skip('Test is unstable')
    def test_run(self):
        if False:
            while True:
                i = 10
        conn_id = connections.ensure_connection(self)
        check_job_name = runner.run_check_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, check_job_name)
        menagerie.verify_check_exit_status(self, exit_status, check_job_name)
        found_catalogs = menagerie.get_catalogs(conn_id)
        self.assertEqual(self.expected_check_streams(), {c['tap_stream_id'] for c in found_catalogs})
        for stream_catalog in found_catalogs:
            annotated_schema = menagerie.get_annotated_schema(conn_id, stream_catalog['stream_id'])
            additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'LOG_BASED'}}]
            selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
        runner.run_sync_mode(self, conn_id)
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        records_by_stream = runner.get_records_from_target_output()
        for tap_stream_id in self.expected_sync_streams():
            self.assertGreaterEqual(record_count_by_stream[tap_stream_id], self.expected_row_count_1()[tap_stream_id])
        initial_state = menagerie.get_state(conn_id)
        bookmarks = initial_state['bookmarks']
        self.assertIsNone(initial_state['currently_syncing'])
        for table_name in self.expected_sync_streams():
            table_bookmark = bookmarks['simple_db-' + table_name]
            bookmark_keys = set(table_bookmark.keys())
            self.assertIn('version', bookmark_keys)
            self.assertIn('last_replication_method', bookmark_keys)
            self.assertIn('initial_full_table_complete', bookmark_keys)
            self.assertIn('oplog_ts_time', bookmark_keys)
            self.assertIn('oplog_ts_inc', bookmark_keys)
            self.assertNotIn('replication_key', bookmark_keys)
            self.assertEqual('LOG_BASED', table_bookmark['last_replication_method'])
            self.assertTrue(table_bookmark['initial_full_table_complete'])
            self.assertIsInstance(table_bookmark['version'], int)
        interrupted_state = copy.deepcopy(initial_state)
        versions = {}
        with get_test_connection() as client:
            docs = list(client.local.oplog.rs.find(sort=[('$natural', pymongo.DESCENDING)]).limit(20))
            ts_to_update = docs[3]['ts']
            updated_ts = str(ts_to_update)
            result = updated_ts[updated_ts.find('(') + 1:updated_ts.find(')')]
            final_result = result.split(',')
            final_result = list(map(int, final_result))
            version = int(time.time() * 1000)
            interrupted_state['bookmarks']['simple_db-' + table_interrupted].update({'oplog_ts_time': final_result[0]})
            interrupted_state['bookmarks']['simple_db-' + table_interrupted].update({'oplog_ts_inc': final_result[1]})
            interrupted_state['currently_syncing'] = 'simple_db-' + table_interrupted
            versions[tap_stream_id] = version
            doc_to_update_1 = client['simple_db']['simple_coll_1'].find_one()
            client['simple_db']['simple_coll_1'].find_one_and_update({'_id': doc_to_update_1['_id']}, {'$set': {'int_field': 999}})
            doc_to_delete_1 = client['simple_db']['simple_coll_1'].find_one({'int_field': 2})
            client['simple_db']['simple_coll_1'].delete_one({'_id': doc_to_delete_1['_id']})
            last_inserted_coll_1 = client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(1))
            last_inserted_id_coll_1 = str(last_inserted_coll_1.inserted_ids[0])
            last_inserted_coll_3 = client['simple_db']['simple_coll_3'].insert_many(generate_simple_coll_docs(1))
            last_inserted_id_coll_3 = str(last_inserted_coll_3.inserted_ids[0])
        menagerie.set_state(conn_id, interrupted_state)
        expected_sync_streams = self.expected_sync_streams()
        expected_row_count_2 = self.expected_row_count_2()
        expected_sync_streams.add('simple_coll_3')
        expected_pks = self.expected_pks()
        expected_pks['simple_coll_3'] = {'_id'}
        expected_row_count_2['simple_coll_2'] = 4
        expected_row_count_2['simple_coll_3'] = 1
        check_job_name_2 = runner.run_check_mode(self, conn_id)
        exit_status_2 = menagerie.get_exit_status(conn_id, check_job_name_2)
        menagerie.verify_check_exit_status(self, exit_status_2, check_job_name_2)
        found_catalogs_2 = menagerie.get_catalogs(conn_id)
        for stream_catalog in found_catalogs_2:
            annotated_schema = menagerie.get_annotated_schema(conn_id, stream_catalog['stream_id'])
            additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'LOG_BASED'}}]
            selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
        second_sync = runner.run_sync_mode(self, conn_id)
        second_sync_exit_status = menagerie.get_exit_status(conn_id, second_sync)
        menagerie.verify_sync_exit_status(self, second_sync_exit_status, second_sync)
        records_by_stream_2 = runner.get_records_from_target_output()
        record_count_by_stream_2 = runner.examine_target_output_file(self, conn_id, expected_sync_streams, expected_pks)
        self.assertGreater(record_count_by_stream[table_interrupted], record_count_by_stream_2[table_interrupted])
        second_state = menagerie.get_state(conn_id)
        for tap_stream_id in initial_state['bookmarks'].keys():
            self.assertSetEqual(set(initial_state['bookmarks'][tap_stream_id].keys()), set(second_state['bookmarks'][tap_stream_id].keys()))
            self.assertEqual(second_state['bookmarks'][tap_stream_id]['version'], initial_state['bookmarks'][tap_stream_id]['version'])
            self.assertEqual(initial_state['bookmarks'][tap_stream_id]['last_replication_method'], second_state['bookmarks'][tap_stream_id]['last_replication_method'])
            self.assertTrue(second_state['bookmarks'][tap_stream_id]['initial_full_table_complete'])
            self.assertGreater(second_state['bookmarks'][tap_stream_id]['oplog_ts_time'], initial_state['bookmarks'][tap_stream_id]['oplog_ts_time'])
        self.assertIsNone(second_state['currently_syncing'])
        third_sync = runner.run_sync_mode(self, conn_id)
        third_sync_exit_status = menagerie.get_exit_status(conn_id, third_sync)
        menagerie.verify_sync_exit_status(self, third_sync_exit_status, third_sync)
        records_by_stream_3 = runner.get_records_from_target_output()
        record_count_by_stream_3 = runner.examine_target_output_file(self, conn_id, expected_sync_streams, expected_pks)
        expected_row_count_3 = self.expected_row_count_3()
        expected_row_count_3['simple_coll_3'] = 1
        for tap_stream_id in expected_sync_streams:
            self.assertEqual(record_count_by_stream_3[tap_stream_id], expected_row_count_3[tap_stream_id])
        self.assertEqual(len(records_by_stream_3['simple_coll_3']['messages']), 2)
        self.assertEqual(records_by_stream_3['simple_coll_3']['messages'][0]['action'], 'activate_version')
        self.assertEqual(records_by_stream_3['simple_coll_3']['messages'][1]['action'], 'upsert')
        self.assertEqual(records_by_stream_3['simple_coll_3']['messages'][1]['data']['_id'], last_inserted_id_coll_3)
        third_state = menagerie.get_state(conn_id)
        for tap_stream_id in third_state['bookmarks'].keys():
            self.assertSetEqual(set(third_state['bookmarks'][tap_stream_id].keys()), set(second_state['bookmarks'][tap_stream_id].keys()))
            self.assertEqual(second_state['bookmarks'][tap_stream_id]['version'], third_state['bookmarks'][tap_stream_id]['version'])
            self.assertEqual(third_state['bookmarks'][tap_stream_id]['last_replication_method'], second_state['bookmarks'][tap_stream_id]['last_replication_method'])
        self.assertIsNone(second_state['currently_syncing'])