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
import json
from bson import ObjectId
import singer
from functools import reduce
from singer import utils, metadata
from mongodb_common import drop_all_collections, get_test_connection, ensure_environment_variables_set
import decimal
RECORD_COUNT = {}

def random_string_generator(size=6, chars=string.ascii_uppercase + string.digits):
    if False:
        for i in range(10):
            print('nop')
    return ''.join((random.choice(chars) for x in range(size)))

def generate_simple_coll_docs(num_docs):
    if False:
        return 10
    docs = []
    for int_value in range(num_docs):
        docs.append({'int_field': int_value, 'string_field': random_string_generator()})
    return docs

class MongoDBProjection(unittest.TestCase):

    def setUpDatabase(self):
        if False:
            while True:
                i = 10
        ensure_environment_variables_set()
        with get_test_connection() as client:
            drop_all_collections(client)
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(50))
            client['simple_db']['simple_coll_2'].insert_many(generate_simple_coll_docs(100))

    def setUp(self):
        if False:
            print('Hello World!')
        pass

    def expected_check_streams(self):
        if False:
            print('Hello World!')
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
            print('Hello World!')
        return {'simple_coll_1', 'simple_coll_2'}

    def projection_expected_keys_list(self):
        if False:
            print('Hello World!')
        return [{'projection': {'int_field': 1}, 'expected_keys': [{'_id', 'int_field'}, {'_id', '_sdc_deleted_at'}]}, {'projection': {'int_field': 1, '_id': 1}, 'expected_keys': [{'_id', 'int_field'}, {'_id', '_sdc_deleted_at'}]}, {'projection': {'int_field': 0}, 'expected_keys': [{'_id', 'string_field'}, {'_id', '_sdc_deleted_at'}]}, {'projection': {'_id': 1}, 'expected_keys': [{'_id'}, {'_id', '_sdc_deleted_at'}]}, {'projection': {}, 'expected_keys': [{'_id'}, {'_id', '_sdc_deleted_at'}]}, {'projection': None, 'expected_keys': [{'_id', 'string_field', 'int_field'}, {'_id', '_sdc_deleted_at'}]}, {'projection': '', 'expected_keys': [{'_id', 'string_field', 'int_field'}, {'_id', '_sdc_deleted_at'}]}]

    def name(self):
        if False:
            print('Hello World!')
        return 'tap_tester_mongodb_projection'

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

    def modify_database(self):
        if False:
            return 10
        with get_test_connection() as client:
            client['simple_db']['simple_coll_1'].delete_one({'int_field': 0})
            client['simple_db']['simple_coll_1'].delete_one({'int_field': 1})
            client['simple_db']['simple_coll_2'].delete_one({'int_field': 0})
            client['simple_db']['simple_coll_2'].delete_one({'int_field': 1})
            client['simple_db']['simple_coll_1'].update_one({'int_field': 48}, {'$set': {'int_field': -1}})
            client['simple_db']['simple_coll_1'].update_one({'int_field': 49}, {'$set': {'int_field': -1}})
            client['simple_db']['simple_coll_2'].update_one({'int_field': 98}, {'$set': {'int_field': -1}})
            client['simple_db']['simple_coll_2'].update_one({'int_field': 99}, {'$set': {'int_field': -1}})
            client['simple_db']['simple_coll_1'].insert_one({'int_field': 50, 'string_field': random_string_generator()})
            client['simple_db']['simple_coll_1'].insert_one({'int_field': 51, 'string_field': random_string_generator()})
            client['simple_db']['simple_coll_2'].insert_one({'int_field': 100, 'string_field': random_string_generator()})
            client['simple_db']['simple_coll_2'].insert_one({'int_field': 101, 'string_field': random_string_generator()})

    def run_single_projection(self, projection_mapping):
        if False:
            print('Hello World!')
        self.setUpDatabase()
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
            if projection_mapping['projection'] is not None:
                additional_md[0]['metadata']['tap_mongodb.projection'] = json.dumps(projection_mapping['projection'])
            selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        messages_by_stream = runner.get_records_from_target_output()
        for stream_name in self.expected_sync_streams():
            stream_records = [x for x in messages_by_stream[stream_name]['messages'] if x.get('action') == 'upsert']
            for record in stream_records:
                self.assertIn(record['data'].keys(), projection_mapping['expected_keys'])
        self.modify_database()
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        menagerie.verify_sync_exit_status(self, exit_status, sync_job_name)
        messages_by_stream = runner.get_records_from_target_output()
        for stream_name in self.expected_sync_streams():
            stream_records = [x for x in messages_by_stream[stream_name]['messages'] if x.get('action') == 'upsert']
            for record in stream_records:
                self.assertIn(record['data'].keys(), projection_mapping['expected_keys'])

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        for projection_mapping in self.projection_expected_keys_list():
            self.run_single_projection(projection_mapping)