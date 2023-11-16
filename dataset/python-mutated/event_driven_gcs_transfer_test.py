import backoff
from google.api_core.exceptions import RetryError
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.storage import Bucket
import event_driven_gcs_transfer

@backoff.on_exception(backoff.expo, (RetryError, ServiceUnavailable), max_time=60)
def test_event_driven_gcs_transfer(capsys, project_id: str, job_description_unique: str, source_bucket: Bucket, destination_bucket: Bucket, pubsub_id: str):
    if False:
        print('Hello World!')
    event_driven_gcs_transfer.create_event_driven_gcs_transfer(project_id=project_id, description=job_description_unique, source_bucket=source_bucket.name, sink_bucket=destination_bucket.name, pubsub_id=pubsub_id)
    (out, _) = capsys.readouterr()
    assert 'Created transferJob:' in out