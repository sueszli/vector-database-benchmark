import os
import uuid
import decimal
import string
import bson
from datetime import datetime, timedelta
import unittest
from unittest import TestCase
import pymongo
import random
from tap_tester import connections, menagerie, runner
from mongodb_common import drop_all_collections, get_test_connection, ensure_environment_variables_set
RECORD_COUNT = {}

def random_string_generator(size=6, chars=string.ascii_uppercase + string.digits):
    if False:
        i = 10
        return i + 15
    return ''.join((random.choice(chars) for x in range(size)))

def generate_simple_coll_docs(num_docs):
    if False:
        i = 10
        return i + 15
    docs = []
    for int_value in range(num_docs):
        docs.append({'int_field': int_value, 'string_field': random_string_generator()})
    return docs

class MongoDBOpenTransactions(unittest.TestCase):

    def expected_check_streams_sync_1(self):
        if False:
            return 10
        return {'simple_db-simple_coll_1', 'simple_db-simple_coll_2', 'simple_db-simple_coll_3'}

    def expected_check_streams_sync_2(self):
        if False:
            i = 10
            return i + 15
        return {'simple_db-simple_coll_1', 'simple_db-simple_coll_2', 'simple_db-simple_coll_3'}

    def expected_pks_1(self):
        if False:
            i = 10
            return i + 15
        return {'simple_db_simple_coll_1': {'_id'}, 'simple_db_simple_coll_2': {'_id'}, 'simple_db_simple_coll_3': {'_id'}}

    def expected_pks_2(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_db_simple_coll_1': {'_id'}, 'simple_db_simple_coll_2': {'_id'}, 'simple_db_simple_coll_3': {'_id'}}

    def expected_row_counts_sync_1(self):
        if False:
            print('Hello World!')
        return {'simple_db_simple_coll_1': 10, 'simple_db_simple_coll_2': 20, 'simple_db_simple_coll_3': 0}

    def expected_row_counts_sync_2(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_db_simple_coll_1': 2, 'simple_db_simple_coll_2': 2, 'simple_db_simple_coll_3': 5}

    def expected_row_counts_sync_3(self):
        if False:
            i = 10
            return i + 15
        return {'simple_db_simple_coll_1': 0, 'simple_db_simple_coll_2': 0, 'simple_db_simple_coll_3': 0}

    def expected_sync_streams_1(self):
        if False:
            return 10
        return {'simple_db_simple_coll_1', 'simple_db_simple_coll_2', 'simple_db_simple_coll_3'}

    def expected_sync_streams_2(self):
        if False:
            while True:
                i = 10
        return {'simple_db_simple_coll_1', 'simple_db_simple_coll_2', 'simple_db_simple_coll_3'}

    def expected_pk_values_2(self):
        if False:
            return 10
        return {'simple_db_simple_coll_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'simple_db_simple_coll_2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'simple_db_simple_coll_3': []}

    def expected_pk_values_3(self):
        if False:
            for i in range(10):
                print('nop')
        return {'simple_db_simple_coll_1': [9, 11], 'simple_db_simple_coll_2': [19, 21], 'simple_db_simple_coll_3': [0, 1, 2, 3, 4]}

    def name(self):
        if False:
            print('Hello World!')
        return 'tap_tester_mongodb_open_transaction'

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
            while True:
                i = 10
        return {'password': os.getenv('TAP_MONGODB_PASSWORD')}

    def get_properties(self):
        if False:
            while True:
                i = 10
        return {'host': os.getenv('TAP_MONGODB_HOST'), 'port': os.getenv('TAP_MONGODB_PORT'), 'user': os.getenv('TAP_MONGODB_USER'), 'database': os.getenv('TAP_MONGODB_DBNAME'), 'include_schemas_in_destination_stream_name': 'true'}

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        ensure_environment_variables_set()
        with get_test_connection() as client:
            drop_all_collections(client)
            session1 = client.start_session()
            session1.start_transaction()
            client['simple_db']['simple_coll_1'].insert_many(generate_simple_coll_docs(10))
            client['simple_db']['simple_coll_2'].insert_many(generate_simple_coll_docs(20))
            session1.commit_transaction()
            '\n                create empty collection\n                update documents in simple_coll_1 & simple_coll_2 and tie to session 2\n                insert documents in simple_coll_3 and tie to session 2\n                execute the sync with uncommitted changes\n                validate that the uncommitted changes are not replicated by the sync\n            '
            session2 = client.start_session()
            session2.start_transaction()
            client['simple_db'].create_collection('simple_coll_3')
            client['simple_db']['simple_coll_1'].update_one({'int_field': 5}, {'$set': {'int_field': 11}}, session=session2)
            client['simple_db']['simple_coll_2'].update_one({'int_field': 10}, {'$set': {'int_field': 21}}, session=session2)
            client['simple_db']['simple_coll_3'].insert_many(generate_simple_coll_docs(5), session=session2)
            conn_id = connections.ensure_connection(self)
            check_job_name = runner.run_check_mode(self, conn_id)
            exit_status = menagerie.get_exit_status(conn_id, check_job_name)
            menagerie.verify_check_exit_status(self, exit_status, check_job_name)
            found_catalogs = menagerie.get_catalogs(conn_id)
            self.assertEqual(self.expected_check_streams_sync_1(), {c['tap_stream_id'] for c in found_catalogs})
            for stream_catalog in found_catalogs:
                annotated_schema = menagerie.get_annotated_schema(conn_id, stream_catalog['stream_id'])
                additional_md = [{'breadcrumb': [], 'metadata': {'replication-method': 'INCREMENTAL', 'replication_key': 'int_field'}}]
                selected_metadata = connections.select_catalog_and_fields_via_metadata(conn_id, stream_catalog, annotated_schema, additional_md)
            sync_1 = runner.run_sync_mode(self, conn_id)
            exit_status = menagerie.get_exit_status(conn_id, sync_1)
            menagerie.verify_sync_exit_status(self, exit_status, sync_1)
            records_by_stream = runner.get_records_from_target_output()
            record_count_by_stream = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams_1(), self.expected_pks_1())
            self.assertEqual(self.expected_row_counts_sync_1(), record_count_by_stream)
            records_2 = {}
            pk_dict_2 = {}
            for stream in self.expected_sync_streams_1():
                records_2[stream] = [x for x in records_by_stream[stream]['messages'] if x.get('action') == 'upsert']
                pk_2 = []
                for record in range(len(records_2[stream])):
                    pk_2.append(records_2[stream][record]['data']['int_field'])
                pk_dict_2[stream] = pk_2
            self.assertEqual(self.expected_pk_values_2(), pk_dict_2)
            session2.commit_transaction()
            '\n               Execute another sync\n               Validate that the documents committed as part of session 2 should now be replicated in sync_2\n            '
            session3 = client.start_session()
            session3.start_transaction()
            sync_2 = runner.run_sync_mode(self, conn_id)
            exit_status_2 = menagerie.get_exit_status(conn_id, sync_2)
            menagerie.verify_sync_exit_status(self, exit_status_2, sync_2)
            records_by_stream_2 = runner.get_records_from_target_output()
            record_count_by_stream_2 = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams_2(), self.expected_pks_2())
            self.assertEqual(self.expected_row_counts_sync_2(), record_count_by_stream_2)
            records_3 = {}
            pk_dict_3 = {}
            for stream in self.expected_sync_streams_1():
                records_3[stream] = [x for x in records_by_stream_2[stream]['messages'] if x.get('action') == 'upsert']
                pk_3 = []
                for record in range(len(records_3[stream])):
                    pk_3.append(records_3[stream][record]['data']['int_field'])
                pk_dict_3[stream] = pk_3
            self.assertEqual(self.expected_pk_values_3(), pk_dict_3)
            state_2 = menagerie.get_state(conn_id)
            for stream in self.expected_check_streams_sync_1():
                rep_key_value = state_2['bookmarks'][stream]['replication_key_value']
                if stream == 'simple_db-simple_coll_1':
                    collection = 'simple_coll_1'
                elif stream == 'simple_db-simple_coll_2':
                    collection = 'simple_coll_2'
                elif stream == 'simple_db-simple_coll_3':
                    collection = 'simple_coll_3'
                client['simple_db'][collection].delete_one({'int_field': int(rep_key_value)}, session=session3)
            session3.commit_transaction()
            '\n               Execute the sync, after the commit on session 3\n               Session 3 commits includes deleting the bookmarked value in each of the collection\n               Validate the state does not change after deleting the bookmarked value\n               Validate that the sync does not replicate any documents\n            '
            state_3 = menagerie.get_state(conn_id)
            sync_3 = runner.run_sync_mode(self, conn_id)
            exit_status_3 = menagerie.get_exit_status(conn_id, sync_3)
            menagerie.verify_sync_exit_status(self, exit_status_3, sync_3)
            records_by_stream_3 = runner.get_records_from_target_output()
            record_count_by_stream_3 = runner.examine_target_output_file(self, conn_id, self.expected_sync_streams_2(), self.expected_pks_2())
            self.assertEqual(self.expected_row_counts_sync_3(), record_count_by_stream_3)
            self.assertEqual(state_2, state_3)