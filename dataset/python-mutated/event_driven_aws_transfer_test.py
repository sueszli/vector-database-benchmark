import backoff
from google.api_core.exceptions import RetryError
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.storage import Bucket
import event_driven_aws_transfer

@backoff.on_exception(backoff.expo, (RetryError, ServiceUnavailable), max_time=60)
def test_event_driven_aws_transfer(capsys, project_id: str, job_description_unique: str, aws_source_bucket: str, destination_bucket: Bucket, sqs_queue_arn: str, aws_access_key_id: str, aws_secret_access_key: str):
    if False:
        i = 10
        return i + 15
    event_driven_aws_transfer.create_event_driven_aws_transfer(project_id=project_id, description=job_description_unique, source_s3_bucket=aws_source_bucket, sink_gcs_bucket=destination_bucket.name, sqs_queue_arn=sqs_queue_arn, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    (out, _) = capsys.readouterr()
    assert 'Created transferJob:' in out