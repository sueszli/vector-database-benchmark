import os
import random
from unittest import mock
from unittest.mock import MagicMock
import google.cloud.dlp_v2
import google.cloud.pubsub
import inspect_gcs as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
TIMEOUT = 900

def mock_job_and_subscriber(mock_dlp_instance: MagicMock, mock_subscriber_instance: MagicMock, dlp_job_path: str, finding_name: str=None, finding_count: int=None):
    if False:
        print('Hello World!')
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
                return 10
            self.data = data
            self.attributes = attributes or {}

        def ack(self):
            if False:
                return 10
            pass

    def mock_subscribe(*args, **kwargs):
        if False:
            return 10
        callback = kwargs.get('callback')
        message = MockMessage(args, {'DlpJobName': dlp_job_path})
        callback(message)
    mock_subscriber_instance.subscribe = mock_subscribe

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
@mock.patch('google.cloud.pubsub.SubscriberClient')
def test_inspect_gcs_file(subscriber_client: MagicMock, dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    mock_dlp_instance = dlp_client.return_value
    mock_subscriber_instance = subscriber_client.return_value
    mock_job_and_subscriber(mock_dlp_instance, mock_subscriber_instance, f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job', 'EMAIL_ADDRESS', 1)
    inspect_content.inspect_gcs_file(GCLOUD_PROJECT, 'MY_BUCKET', 'test.txt', 'topic_id', 'subscription_id', ['EMAIL_ADDRESS', 'PHONE_NUMBER'], timeout=1)
    (out, _) = capsys.readouterr()
    assert 'Job name:' in out
    assert 'Info type: EMAIL_ADDRESS' in out
    mock_dlp_instance.create_dlp_job.assert_called_once()
    mock_dlp_instance.get_dlp_job.assert_called_once()

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
@mock.patch('google.cloud.pubsub.SubscriberClient')
def test_inspect_gcs_file_with_custom_info_types(subscriber_client: MagicMock, dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    mock_dlp_instance = dlp_client.return_value
    mock_subscriber_instance = subscriber_client.return_value
    mock_job_and_subscriber(mock_dlp_instance, mock_subscriber_instance, f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job', 'EMAIL_ADDRESS', 1)
    dictionaries = ['gary@somedomain.com']
    regexes = ['\\(\\d{3}\\) \\d{3}-\\d{4}']
    inspect_content.inspect_gcs_file(GCLOUD_PROJECT, 'MY_BUCKET', 'test.txt', 'topic_id', 'subscription_id', [], custom_dictionaries=dictionaries, custom_regexes=regexes, timeout=1)
    (out, _) = capsys.readouterr()
    assert 'Info type: EMAIL_ADDRESS' in out
    assert 'Job name:' in out
    mock_dlp_instance.create_dlp_job.assert_called_once()
    mock_dlp_instance.get_dlp_job.assert_called_once()

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
@mock.patch('google.cloud.pubsub.SubscriberClient')
def test_inspect_gcs_file_no_results(subscriber_client: MagicMock, dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    mock_dlp_instance = dlp_client.return_value
    mock_subscriber_instance = subscriber_client.return_value
    mock_job_and_subscriber(mock_dlp_instance, mock_subscriber_instance, f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job')
    inspect_content.inspect_gcs_file(GCLOUD_PROJECT, 'MY_BUCKET', 'harmless.txt', 'topic_id', 'subscription_id', ['EMAIL_ADDRESS', 'PHONE_NUMBER'], timeout=TIMEOUT)
    (out, _) = capsys.readouterr()
    assert 'No findings' in out
    assert 'Job name:' in out
    mock_dlp_instance.create_dlp_job.assert_called_once()
    mock_dlp_instance.get_dlp_job.assert_called_once()

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
@mock.patch('google.cloud.pubsub.SubscriberClient')
def test_inspect_gcs_image_file(subscriber_client: MagicMock, dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    mock_dlp_instance = dlp_client.return_value
    mock_subscriber_instance = subscriber_client.return_value
    mock_job_and_subscriber(mock_dlp_instance, mock_subscriber_instance, f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job', 'EMAIL_ADDRESS', 1)
    inspect_content.inspect_gcs_file(GCLOUD_PROJECT, 'MY_BUCKET', 'test.png', 'topic_id', 'subscription_id', ['EMAIL_ADDRESS', 'PHONE_NUMBER'], timeout=1)
    (out, _) = capsys.readouterr()
    assert 'Info type: EMAIL_ADDRESS' in out
    assert 'Job name:' in out
    mock_dlp_instance.create_dlp_job.assert_called_once()
    mock_dlp_instance.get_dlp_job.assert_called_once()

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
@mock.patch('google.cloud.pubsub.SubscriberClient')
def test_inspect_gcs_multiple_files(subscriber_client: MagicMock, dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    mock_dlp_instance = dlp_client.return_value
    mock_subscriber_instance = subscriber_client.return_value
    mock_job_and_subscriber(mock_dlp_instance, mock_subscriber_instance, f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job', 'EMAIL_ADDRESS', random.randint(0, 1000))
    inspect_content.inspect_gcs_file(GCLOUD_PROJECT, 'MY_BUCKET', '*', 'topic_id', 'subscription_id', ['EMAIL_ADDRESS', 'PHONE_NUMBER'], timeout=1)
    (out, _) = capsys.readouterr()
    assert 'Info type: EMAIL_ADDRESS' in out
    assert 'Job name:' in out
    mock_dlp_instance.create_dlp_job.assert_called_once()
    mock_dlp_instance.get_dlp_job.assert_called_once()