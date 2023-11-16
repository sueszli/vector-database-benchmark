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
        return 10
    return ''.join((random.choice(chars) for x in range(size)))

def generate_simple_coll_docs(num_docs):
    if False:
        for i in range(10):
            print('nop')
    docs = []
    for int_value in range(num_docs):
        docs.append({'int_field': int_value, 'string_field': random_string_generator()})
    return docs

class MongoDBFullTable(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        ensure_environment_variables_set()
        with get_test_connection() as client:
            drop_all_collections(client)
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(50))
            client['simple_db'].command(bson.son.SON([('create', 'simple_view_1'), ('viewOn', 'simple_coll_1'), ('pipeline', [])]))
            client['simple_db']['simple_coll_2'].insert_many(generate_simple_coll_docs(100))
            client['admin']['admin_coll_1'].insert_many(generate_simple_coll_docs(50))
            client['simple_db'].create_collection('simple_coll_3')
            client['simple_db']['simple_coll_4'].insert_one({'hebrew_ישרא': 'hebrew_ישרא'})
            client['simple_db']['simple_coll_4'].insert_one({'hebrew_ישרא': 2})
            client['simple_db']['simple_coll_4'].insert_one({'another_hebrew_ישראל': 'another_hebrew_ישרא'})
            nested_doc = {'field0': {}}
            current_doc = nested_doc
            for i in range(1, 101):
                current_doc['field{}'.format(i - 1)]['field{}'.format(i)] = {}
                current_doc = current_doc['field{}'.format(i - 1)]
            current_doc['field100'] = 'some_value'
            client['simple_db']['simple_coll_4'].insert_one(nested_doc)
            max_col_doc = {}
            for x in range(1600):
                max_col_doc['col_{}'.format(x)] = x
            client['simple_db']['simple_coll_4'].insert_one(max_col_doc)

    def tap_stream_id_to_stream(self):
        if False:
            while True:
                i = 10
        return {'simple_db-simple_coll_1': 'simple_db_simple_coll_1', 'simple_db-simple_coll_2': 'simple_db_simple_coll_2', 'simple_db-simple_coll_3': 'simple_db_simple_coll_3', 'simple_db-simple_coll_4': 'simple_db_simple_coll_4', 'admin-admin_coll_1': 'admin_admin_coll_1'}

    def expected_check_streams(self):
        if False:
            return 10
        return {'simple_db-simple_coll_1', 'simple_db-simple_coll_2', 'simple_db-simple_coll_3', 'simple_db-simple_coll_4', 'admin-admin_coll_1'}

    def expected_pks(self):
        if False:
            while True:
                i = 10
        return {'simple_db_simple_coll_1': {'_id'}, 'simple_db_simple_coll_2': {'_id'}, 'simple_db_simple_coll_3': {'_id'}, 'simple_db_simple_coll_4': {'_id'}, 'admin_admin_coll_1': {'_id'}}

    def expected_row_counts(self):
        if False:
            return 10
        return {'simple_db_simple_coll_1': 50, 'simple_db_simple_coll_2': 100, 'simple_db_simple_coll_3': 0, 'simple_db_simple_coll_4': 5, 'admin_admin_coll_1': 50}

    def expected_sync_streams(self):
        if False:
            while True:
                i = 10
        return {'simple_db_simple_coll_1', 'simple_db_simple_coll_2', 'simple_db_simple_coll_3', 'simple_db_simple_coll_4', 'admin_admin_coll_1'}

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'tap_tester_mongodb_full_table'

    def tap_name(self):
        if False:
            while True:
                i = 10
        return 'tap-mongodb'

    def get_type(self):
        if False:
            for i in range(10):
                print('nop')
        return 'platform.mongodb'

    def get_credentials(self):
        if False:
            return 10
        return {'password': os.getenv('TAP_MONGODB_PASSWORD')}

    def get_properties(self):
        if False:
            for i in range(10):
                print('nop')
        return {'host': os.getenv('TAP_MONGODB_HOST'), 'port': os.getenv('TAP_MONGODB_PORT'), 'user': os.getenv('TAP_MONGODB_USER'), 'database': os.getenv('TAP_MONGODB_DBNAME'), 'include_schemas_in_destination_stream_name': 'true'}

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
            additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'FULL_TABLE'}}]
            selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        records_by_stream = runner.get_records_from_target_output()
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        self.assertEqual(self.expected_row_counts(), record_count_by_stream)
        for stream_name in self.expected_sync_streams():
            self.assertEqual('activate_version', records_by_stream[stream_name]['messages'][0]['action'])
            self.assertEqual('activate_version', records_by_stream[stream_name]['messages'][-1]['action'])
        state = menagerie.get_state(conn_id)
        first_versions = {}
        for tap_stream_id in self.expected_check_streams():
            self.assertTrue(state['bookmarks'][tap_stream_id]['initial_full_table_complete'])
            first_versions[tap_stream_id] = state['bookmarks'][tap_stream_id]['version']
            self.assertIsNotNone(first_versions[tap_stream_id])
        with get_test_connection() as client:
            doc_to_update = client['simple_db']['simple_coll_1'].find_one()
            client['simple_db']['simple_coll_1'].find_one_and_update({'_id': doc_to_update['_id']}, {'$set': {'int_field': 999}})
            doc_to_update = client['simple_db']['simple_coll_2'].find_one()
            client['simple_db']['simple_coll_2'].find_one_and_update({'_id': doc_to_update['_id']}, {'$set': {'int_field': 888}})
            doc_to_update = client['admin']['admin_coll_1'].find_one()
            client['admin']['admin_coll_1'].find_one_and_update({'_id': doc_to_update['_id']}, {'$set': {'int_field': 777}})
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(2))
            client['simple_db']['simple_coll_2'].insert_many(generate_simple_coll_docs(2))
            client['admin']['admin_coll_1'].insert_many(generate_simple_coll_docs(2))
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        records_by_stream = runner.get_records_from_target_output()
        record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams(), self.expected_pks())
        state = menagerie.get_state(conn_id)
        self.assertIsNone(state['currently_syncing'])
        self.assertNotIn('oplog', state)
        new_expected_row_counts = {k: v + 2 for (k, v) in self.expected_row_counts().items() if k not in ['simple_db_simple_coll_3', 'simple_db_simple_coll_4']}
        new_expected_row_counts['simple_db_simple_coll_3'] = 0
        new_expected_row_counts['simple_db_simple_coll_4'] = 5
        self.assertEqual(new_expected_row_counts, record_count_by_stream)
        for stream_name in self.expected_sync_streams():
            if len(records_by_stream[stream_name]['messages']) > 1:
                self.assertNotEqual('activate_version', records_by_stream[stream_name]['messages'][0]['action'], stream_name + 'failed')
                self.assertEqual('upsert', records_by_stream[stream_name]['messages'][0]['action'], stream_name + 'failed')
            self.assertEqual('activate_version', records_by_stream[stream_name]['messages'][-1]['action'], stream_name + 'failed')
        second_versions = {}
        for tap_stream_id in self.expected_check_streams():
            found_stream = [c for c in found_catalogs if c['tap_stream_id'] == tap_stream_id][0]
            self.assertTrue(state['bookmarks'][tap_stream_id]['initial_full_table_complete'])
            second_versions[tap_stream_id] = state['bookmarks'][tap_stream_id]['version']
            self.assertIsNotNone(second_versions[tap_stream_id])
            self.assertNotEqual(first_versions[tap_stream_id], second_versions[tap_stream_id])
            self.assertGreater(second_versions[tap_stream_id], first_versions[tap_stream_id])
            self.assertEqual(records_by_stream[self.tap_stream_id_to_stream()[tap_stream_id]]['table_version'], second_versions[tap_stream_id])