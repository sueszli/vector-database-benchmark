import time
import typing
import uuid
from flaky import flaky
import google.auth
import google.cloud.storage as storage
import pytest
from ..usage_report.usage_reports import disable_usage_export
from ..usage_report.usage_reports import get_usage_export_bucket
from ..usage_report.usage_reports import set_usage_export_bucket
PROJECT = google.auth.default()[1]
BUCKET_NAME = 'test' + uuid.uuid4().hex[:10]
TEST_PREFIX = 'some-prefix'

@pytest.fixture
def temp_bucket():
    if False:
        for i in range(10):
            print('nop')
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(BUCKET_NAME)
    yield bucket
    bucket.delete(force=True)

@flaky(max_runs=3)
def test_set_usage_export_bucket_default(capsys: typing.Any, temp_bucket: storage.Bucket) -> None:
    if False:
        for i in range(10):
            print('nop')
    set_usage_export_bucket(project_id=PROJECT, bucket_name=temp_bucket.name)
    time.sleep(5)
    uel = get_usage_export_bucket(project_id=PROJECT)
    assert uel.bucket_name == temp_bucket.name
    assert uel.report_name_prefix == 'usage_gce'
    (out, _) = capsys.readouterr()
    assert 'default prefix of `usage_gce`.' in out
    disable_usage_export(project_id=PROJECT)
    time.sleep(5)
    uel = get_usage_export_bucket(project_id=PROJECT)
    assert uel.bucket_name == ''
    assert uel.report_name_prefix == ''
    set_usage_export_bucket(project_id=PROJECT, bucket_name=temp_bucket.name, report_name_prefix=TEST_PREFIX)
    time.sleep(5)
    uel = get_usage_export_bucket(project_id=PROJECT)
    assert uel.bucket_name == temp_bucket.name
    assert uel.report_name_prefix == TEST_PREFIX
    (out, _) = capsys.readouterr()
    assert 'usage_gce' not in out
    disable_usage_export(project_id=PROJECT)
    time.sleep(5)
    uel = get_usage_export_bucket(project_id=PROJECT)
    assert uel.bucket_name == ''
    assert uel.report_name_prefix == ''