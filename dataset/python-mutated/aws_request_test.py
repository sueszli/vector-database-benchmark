import datetime
import backoff
from google.api_core.exceptions import RetryError
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.storage import Bucket
from googleapiclient.errors import HttpError
import aws_request
import aws_request_apiary

@backoff.on_exception(backoff.expo, (RetryError, ServiceUnavailable), max_time=60)
def test_aws_request(capsys, project_id: str, aws_source_bucket: str, aws_access_key_id: str, aws_secret_access_key: str, destination_bucket: Bucket, job_description_unique: str):
    if False:
        for i in range(10):
            print('nop')
    aws_request.create_one_time_aws_transfer(project_id=project_id, description=job_description_unique, source_bucket=aws_source_bucket, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, sink_bucket=destination_bucket.name)
    (out, _) = capsys.readouterr()
    assert 'Created transferJob:' in out

@backoff.on_exception(backoff.expo, (HttpError,), max_time=60)
def test_aws_request_apiary(capsys, project_id: str, aws_source_bucket: str, aws_access_key_id: str, aws_secret_access_key: str, destination_bucket: Bucket, job_description_unique: str):
    if False:
        print('Hello World!')
    aws_request_apiary.main(description=job_description_unique, project_id=project_id, start_date=datetime.datetime.utcnow(), start_time=datetime.datetime.utcnow(), source_bucket=aws_source_bucket, access_key_id=aws_access_key_id, secret_access_key=aws_secret_access_key, sink_bucket=destination_bucket.name)
    (out, _) = capsys.readouterr()
    assert 'Returned transferJob:' in out