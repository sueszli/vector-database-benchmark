import backoff
from google.api_core.exceptions import RetryError
from google.cloud import storage_transfer
from google.cloud.storage import Bucket
from google.protobuf.duration_pb2 import Duration
from googleapiclient.errors import HttpError
import pytest
import get_transfer_job_with_retries
import get_transfer_job_with_retries_apiary

@pytest.fixture()
def transfer_job(project_id: str, source_bucket: Bucket, destination_bucket: Bucket):
    if False:
        i = 10
        return i + 15
    client = storage_transfer.StorageTransferServiceClient()
    transfer_job = {'description': 'Sample job', 'status': 'ENABLED', 'project_id': project_id, 'schedule': {'schedule_start_date': {'day': 1, 'month': 1, 'year': 2000}, 'start_time_of_day': {'hours': 0, 'minutes': 0, 'seconds': 0}}, 'transfer_spec': {'gcs_data_source': {'bucket_name': source_bucket.name}, 'gcs_data_sink': {'bucket_name': destination_bucket.name}, 'object_conditions': {'min_time_elapsed_since_last_modification': Duration(seconds=2592000)}, 'transfer_options': {'delete_objects_from_source_after_transfer': True}}}
    result = client.create_transfer_job({'transfer_job': transfer_job})
    yield result.name
    client.update_transfer_job({'job_name': result.name, 'project_id': project_id, 'transfer_job': {'status': storage_transfer.TransferJob.Status.DELETED}})

@backoff.on_exception(backoff.expo, (RetryError,), max_time=60)
def test_get_transfer_job_with_retries(capsys, project_id: str, transfer_job: str):
    if False:
        return 10
    max_retry_duration = 120
    get_transfer_job_with_retries.get_transfer_job_with_retries(project_id, transfer_job, max_retry_duration)
    (out, _) = capsys.readouterr()
    assert f'max retry duration of {max_retry_duration}s' in out

@backoff.on_exception(backoff.expo, (HttpError,), max_time=60)
def test_get_transfer_job_with_retries_apiary(capsys, project_id: str, transfer_job: str):
    if False:
        return 10
    retries = 3
    get_transfer_job_with_retries_apiary.get_transfer_job_with_retries(project_id, transfer_job, retries)
    (out, _) = capsys.readouterr()
    assert f'using {str(retries)} retries' in out