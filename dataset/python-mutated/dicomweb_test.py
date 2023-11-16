import os
import sys
import uuid
import backoff
from googleapiclient.errors import HttpError
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from create_dataset import create_dataset
from delete_dataset import delete_dataset
import dicom_stores
import dicomweb
location = 'us-central1'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
dataset_id = f'test_dataset-{uuid.uuid4()}'
dicom_store_id = f'test_dicom_store_{uuid.uuid4()}'
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')
dcm_file_name = 'dicom_00000001_000.dcm'
dcm_file = os.path.join(RESOURCES, dcm_file_name)
study_uid = '1.3.6.1.4.1.11129.5.5.111396399361969898205364400549799252857604'
series_uid = '1.3.6.1.4.1.11129.5.5.195628213694300498946760767481291263511724'
instance_uid = '{}.{}'.format('1.3.6.1.4.1.11129.5.5', '153751009835107614666834563294684339746480')

@pytest.fixture(scope='module')
def test_dataset():
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
        try:
            delete_dataset(project_id, location, dataset_id)
        except HttpError as err:
            if err.resp.status == 403:
                print(f'Got exception {err.resp.status} while deleting dataset')
            else:
                raise
    clean_up()

@pytest.fixture(scope='module')
def test_dicom_store():
    if False:
        i = 10
        return i + 15

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create():
        if False:
            return 10
        try:
            dicom_stores.create_dicom_store(project_id, location, dataset_id, dicom_store_id)
        except HttpError as err:
            if err.resp.status == 409:
                print('Got exception {} while creating DICOM store'.format(err.resp.status))
            else:
                raise
    create()
    yield

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def clean_up():
        if False:
            i = 10
            return i + 15
        try:
            dicom_stores.delete_dicom_store(project_id, location, dataset_id, dicom_store_id)
        except HttpError as err:
            if err.resp.status == 404 or err.resp.status == 403:
                print('Got exception {} while deleting DICOM store'.format(err.resp.status))
            else:
                raise
    clean_up()

def test_dicomweb_store_instance(test_dataset, test_dicom_store, capsys):
    if False:
        i = 10
        return i + 15
    dicomweb.dicomweb_store_instance(project_id, location, dataset_id, dicom_store_id, dcm_file)
    (out, _) = capsys.readouterr()
    assert 'Stored DICOM instance' in out

def test_dicomweb_search_instance_studies(test_dataset, test_dicom_store, capsys):
    if False:
        i = 10
        return i + 15
    dicomweb.dicomweb_store_instance(project_id, location, dataset_id, dicom_store_id, dcm_file)
    dicomweb.dicomweb_search_instance(project_id, location, dataset_id, dicom_store_id)
    dicomweb.dicomweb_search_studies(project_id, location, dataset_id, dicom_store_id)
    (out, _) = capsys.readouterr()
    assert 'Instances:' in out
    assert 'Studies found: response is <Response [204]>' in out

def test_dicomweb_retrieve_study(test_dataset, test_dicom_store, capsys):
    if False:
        for i in range(10):
            print('nop')
    try:
        dicomweb.dicomweb_store_instance(project_id, location, dataset_id, dicom_store_id, dcm_file)
        dicomweb.dicomweb_retrieve_study(project_id, location, dataset_id, dicom_store_id, study_uid)
        assert os.path.isfile('study.multipart')
        (out, _) = capsys.readouterr()
        assert 'Retrieved study' in out
    finally:
        os.remove('study.multipart')

def test_dicomweb_retrieve_instance(test_dataset, test_dicom_store, capsys):
    if False:
        for i in range(10):
            print('nop')
    try:
        dicomweb.dicomweb_store_instance(project_id, location, dataset_id, dicom_store_id, dcm_file)
        dicomweb.dicomweb_retrieve_instance(project_id, location, dataset_id, dicom_store_id, study_uid, series_uid, instance_uid)
        assert os.path.isfile('instance.dcm')
        (out, _) = capsys.readouterr()
        assert 'Retrieved DICOM instance' in out
    finally:
        os.remove('instance.dcm')

def test_dicomweb_retrieve_rendered(test_dataset, test_dicom_store, capsys):
    if False:
        while True:
            i = 10
    try:
        dicomweb.dicomweb_store_instance(project_id, location, dataset_id, dicom_store_id, dcm_file)
        dicomweb.dicomweb_retrieve_rendered(project_id, location, dataset_id, dicom_store_id, study_uid, series_uid, instance_uid)
        assert os.path.isfile('rendered_image.png')
        (out, _) = capsys.readouterr()
        assert 'Retrieved rendered image' in out
    finally:
        os.remove('rendered_image.png')

def test_dicomweb_delete_study(test_dataset, test_dicom_store, capsys):
    if False:
        return 10
    dicomweb.dicomweb_store_instance(project_id, location, dataset_id, dicom_store_id, dcm_file)
    dicomweb.dicomweb_delete_study(project_id, location, dataset_id, dicom_store_id, study_uid)
    (out, _) = capsys.readouterr()
    assert 'Deleted study.' in out