import backoff
from google.api_core.exceptions import RetryError
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.storage import Bucket
import azure_request

@backoff.on_exception(backoff.expo, (RetryError, ServiceUnavailable), max_time=60)
def test_azure_request(capsys, project_id: str, azure_storage_account: str, azure_sas_token: str, azure_source_container: str, destination_bucket: Bucket, job_description_unique: str):
    if False:
        return 10
    azure_request.create_one_time_azure_transfer(project_id=project_id, description=job_description_unique, azure_storage_account=azure_storage_account, azure_sas_token=azure_sas_token, source_container=azure_source_container, sink_bucket=destination_bucket.name)
    (out, _) = capsys.readouterr()
    assert 'Created transferJob:' in out