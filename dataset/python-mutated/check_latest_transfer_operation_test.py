import backoff
from google.api_core.exceptions import RetryError
from google.cloud import storage_transfer
from google.cloud.storage import Bucket
from google.protobuf.duration_pb2 import Duration
from googleapiclient.errors import HttpError
import pytest
import check_latest_transfer_operation
import check_latest_transfer_operation_apiary

@pytest.fixture()
def transfer_job(project_id: str, source_bucket: Bucket, destination_bucket: Bucket):
    if False:
        while True:
            i = 10
    client = storage_transfer.StorageTransferServiceClient()
    transfer_job = {'description': 'Sample job', 'status': 'ENABLED', 'project_id': project_id, 'schedule': {'schedule_start_date': {'day': 1, 'month': 1, 'year': 2000}, 'start_time_of_day': {'hours': 0, 'minutes': 0, 'seconds': 0}}, 'transfer_spec': {'gcs_data_source': {'bucket_name': source_bucket.name}, 'gcs_data_sink': {'bucket_name': destination_bucket.name}, 'object_conditions': {'min_time_elapsed_since_last_modification': Duration(seconds=2592000)}, 'transfer_options': {'delete_objects_from_source_after_transfer': True}}}

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def create_job():
        if False:
            i = 10
            return i + 15
        return client.create_transfer_job({'transfer_job': transfer_job})
    result = create_job()
    yield result.name

    @backoff.on_exception(backoff.expo, HttpError, max_time=60)
    def remove_job():
        if False:
            for i in range(10):
                print('nop')
        client.update_transfer_job({'job_name': result.name, 'project_id': project_id, 'transfer_job': {'status': storage_transfer.TransferJob.Status.DELETED}})
    remove_job()

@backoff.on_exception(backoff.expo, (RetryError,), max_time=60)
def test_latest_transfer_operation(capsys, project_id: str, transfer_job: str):
    if False:
        for i in range(10):
            print('nop')
    check_latest_transfer_operation.check_latest_transfer_operation(project_id, transfer_job)
    (out, _) = capsys.readouterr()
    assert transfer_job in out

@backoff.on_exception(backoff.expo, (HttpError,), max_time=60)
def test_latest_transfer_operation_apiary(capsys, project_id: str, transfer_job: str):
    if False:
        print('Hello World!')
    check_latest_transfer_operation_apiary.check_latest_transfer_operation(project_id, transfer_job)
    (out, _) = capsys.readouterr()
    assert transfer_job in out