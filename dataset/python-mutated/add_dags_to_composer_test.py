import os
import pathlib
from shutil import copytree
import tempfile
import uuid
from google.cloud import storage
import pytest
import add_dags_to_composer
DAGS_DIR = pathlib.Path(__file__).parent.parent / 'dags/'

@pytest.fixture(scope='function')
def dags_directory() -> str:
    if False:
        i = 10
        return i + 15
    'Copies contents of dags/ folder to a temporary directory'
    temp_dir = tempfile.mkdtemp()
    copytree(DAGS_DIR, f'{temp_dir}/', dirs_exist_ok=True)
    yield temp_dir

@pytest.fixture(scope='function')
def empty_directory() -> str:
    if False:
        i = 10
        return i + 15
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

@pytest.fixture(scope='module')
def test_bucket() -> str:
    if False:
        print('Hello World!')
    'Yields a bucket that is deleted after the test completes.'
    storage_client = storage.Client()
    bucket_name = f'temp-composer-cicd-test-{str(uuid.uuid4())}'
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        bucket = storage_client.create_bucket(bucket_name)
    yield bucket_name
    bucket = storage_client.bucket(bucket_name)
    bucket.delete(force=True)

def test_create_dags_list_invalid_directory() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(FileNotFoundError):
        (temp_dir, dags) = add_dags_to_composer._create_dags_list('this-directory-does-not-exist/')

def test_create_dags_list_empty_directory(empty_directory: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    (temp_dir, dags) = add_dags_to_composer._create_dags_list(empty_directory)
    assert len(dags) == 0
    assert len(os.listdir(temp_dir)) == 0

def test_create_dags_list(dags_directory: str) -> None:
    if False:
        print('Hello World!')
    (temp_dir, dags) = add_dags_to_composer._create_dags_list(dags_directory)
    assert len(dags) == 2
    assert f'{temp_dir}/__init__.py' not in dags
    assert f'{temp_dir}/example_dag.py' in dags
    assert f'{temp_dir}/example2_dag.py' in dags
    assert f'{temp_dir}/example_dag_test.py' not in dags
    assert f'{temp_dir}/example2_dag_test.py' not in dags

def test_upload_dags_to_composer_no_files(capsys: pytest.CaptureFixture, empty_directory: str, test_bucket: str) -> None:
    if False:
        i = 10
        return i + 15
    add_dags_to_composer.upload_dags_to_composer(empty_directory, test_bucket)
    (out, _) = capsys.readouterr()
    assert 'No DAGs to upload.' in out

def test_upload_dags_to_composer_no_name_override(test_bucket: str) -> None:
    if False:
        return 10
    with pytest.raises(FileNotFoundError):
        add_dags_to_composer.upload_dags_to_composer(DAGS_DIR, test_bucket)

def test_upload_dags_to_composer(test_bucket: str, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    add_dags_to_composer.upload_dags_to_composer(DAGS_DIR, test_bucket, '../dags/')
    (out, _) = capsys.readouterr()
    assert 'uploaded' in out