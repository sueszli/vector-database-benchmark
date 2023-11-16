import os
import sys
import uuid
import backoff
from googleapiclient.errors import HttpError
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from create_dataset import create_dataset
from delete_dataset import delete_dataset
import hl7v2_messages
import hl7v2_stores
location = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
dataset_id = f'test_dataset_{uuid.uuid4()}'
hl7v2_store_id = f'test_hl7v2_store-{uuid.uuid4()}'
hl7v2_message_file = 'resources/hl7-sample-ingest.json'
label_key = 'PROCESSED'
label_value = 'TRUE'

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        return 10

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            for i in range(10):
                print('nop')
        try:
            create_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 409:
                print(f'Got exception {err.resp.status} while creating dataset')
            else:
                raise
    create()
    yield

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            print('Hello World!')
        try:
            delete_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print(f'Got exception {err.resp.status} while deleting dataset')
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def test_hl7v2_store():
    if False:
        while True:
            i = 10

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            i = 10
            return i + 15
        try:
            hl7v2_stores.create_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
        except HttpError as err:
            if err.resp.status == 409:
                print('Got exception {} while creating HL7v2 store'.format(err.resp.status))
            else:
                raise
    create()
    yield

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            for i in range(10):
                print('nop')
        try:
            hl7v2_stores.delete_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting HL7v2 store'.format(err.resp.status))
            else:
                raise
    clean_up()

def test_CRUD_hl7v2_message(test_dataset, test_hl7v2_store, capsys):
    if False:
        for i in range(10):
            print('nop')
    hl7v2_messages.create_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_file)

    @backoff.on_exception(backoff.expo, AssertionError, max_time=60)
    def run_eventually_consistent_test():
        if False:
            i = 10
            return i + 15
        hl7v2_messages_list = hl7v2_messages.list_hl7v2_messages(project_id, location, dataset_id, hl7v2_store_id)
        assert len(hl7v2_messages_list) > 0
        hl7v2_message_name = hl7v2_messages_list[0].get('name')
        elms = hl7v2_message_name.split('/', 9)
        assert len(elms) >= 10
        hl7v2_message_id = elms[9]
        return hl7v2_message_id
    hl7v2_message_id = run_eventually_consistent_test()
    hl7v2_messages.get_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id)
    hl7v2_messages.delete_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id)
    (out, _) = capsys.readouterr()
    assert 'Created HL7v2 message' in out
    assert 'Name' in out
    assert 'Deleted HL7v2 message' in out

def test_ingest_hl7v2_message(test_dataset, test_hl7v2_store, capsys):
    if False:
        return 10
    hl7v2_messages.ingest_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_file)

    @backoff.on_exception(backoff.expo, AssertionError, max_time=60)
    def run_eventually_consistent_test():
        if False:
            for i in range(10):
                print('nop')
        hl7v2_messages_list = hl7v2_messages.list_hl7v2_messages(project_id, location, dataset_id, hl7v2_store_id)
        assert len(hl7v2_messages_list) > 0
        hl7v2_message_name = hl7v2_messages_list[0].get('name')
        elms = hl7v2_message_name.split('/', 9)
        assert len(elms) >= 10
        hl7v2_message_id = elms[9]
        return hl7v2_message_id
    hl7v2_message_id = run_eventually_consistent_test()
    hl7v2_messages.get_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id)
    hl7v2_messages.delete_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id)
    (out, _) = capsys.readouterr()
    assert 'Ingested HL7v2 message' in out
    assert 'Name' in out
    assert 'Deleted HL7v2 message' in out

def test_patch_hl7v2_message(test_dataset, test_hl7v2_store, capsys):
    if False:
        print('Hello World!')
    hl7v2_messages.create_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_file)

    @backoff.on_exception(backoff.expo, (AssertionError, HttpError), max_time=60)
    def run_eventually_consistent_test():
        if False:
            return 10
        hl7v2_messages_list = hl7v2_messages.list_hl7v2_messages(project_id, location, dataset_id, hl7v2_store_id)
        assert len(hl7v2_messages_list) > 0
        hl7v2_message_name = hl7v2_messages_list[0].get('name')
        elms = hl7v2_message_name.split('/', 9)
        assert len(elms) >= 10
        hl7v2_message_id = elms[9]
        return hl7v2_message_id
    hl7v2_message_id = run_eventually_consistent_test()
    hl7v2_messages.patch_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id, label_key, label_value)
    hl7v2_messages.delete_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id)
    (out, _) = capsys.readouterr()
    assert 'Patched HL7v2 message' in out