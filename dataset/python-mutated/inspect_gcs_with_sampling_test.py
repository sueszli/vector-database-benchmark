import os
import random
from unittest import mock
from unittest.mock import MagicMock
import google.cloud.dlp_v2
import google.cloud.pubsub
import inspect_gcs_with_sampling as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def mock_job_and_subscriber(mock_dlp_instance: MagicMock, mock_subscriber_instance: MagicMock, dlp_job_path: str, finding_name: str=None, finding_count: int=None):
    if False:
        i = 10
        return i + 15
    mock_dlp_instance.create_dlp_job.return_value.name = dlp_job_path
    mock_job = mock_dlp_instance.get_dlp_job.return_value
    mock_job.name = dlp_job_path
    mock_job.state = google.cloud.dlp_v2.DlpJob.JobState.DONE
    if finding_name:
        finding = mock_job.inspect_details.result.info_type_stats.info_type
        finding.name = finding_name
        mock_job.inspect_details.result.info_type_stats = [MagicMock(info_type=finding, count=finding_count)]
    else:
        mock_job.inspect_details.result.info_type_stats = None

    class MockMessage:

        def __init__(self, data, attributes=None):
            if False:
                print('Hello World!')
            self.data = data
            self.attributes = attributes or {}

        def ack(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def mock_subscribe(*args, **kwargs):
        if False:
            print('Hello World!')
        callback = kwargs.get('callback')
        message = MockMessage(args, {'DlpJobName': dlp_job_path})
        callback(message)
    mock_subscriber_instance.subscribe = mock_subscribe

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
@mock.patch('google.cloud.pubsub.SubscriberClient')
def test_inspect_gcs_with_sampling(subscriber_client: MagicMock, dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    mock_dlp_instance = dlp_client.return_value
    mock_subscriber_instance = subscriber_client.return_value
    mock_job_and_subscriber(mock_dlp_instance, mock_subscriber_instance, f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job', 'EMAIL_ADDRESS', random.randint(0, 1000))
    inspect_content.inspect_gcs_with_sampling(GCLOUD_PROJECT, 'GCS_BUCKET', 'topic_id', 'subscription_id', ['EMAIL_ADDRESS', 'PHONE_NUMBER'], ['TEXT_FILE'], timeout=1)
    (out, _) = capsys.readouterr()
    assert 'Inspection operation started:' in out
    assert 'Job name:' in out
    mock_dlp_instance.create_dlp_job.assert_called_once()
    mock_dlp_instance.get_dlp_job.assert_called_once()