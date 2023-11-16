import os
from unittest import mock
from unittest.mock import MagicMock
import uuid
import google.cloud.dlp_v2
import inspect_gcs_send_to_scc as inspect_content
import pytest
UNIQUE_STRING = str(uuid.uuid4()).split('-')[0]
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
TEST_BUCKET_NAME = GCLOUD_PROJECT + '-dlp-python-client-test' + UNIQUE_STRING
RESOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../resources')

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
def test_inspect_gcs_send_to_scc(dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    mock_dlp_instance = dlp_client.return_value
    mock_dlp_instance.create_dlp_job.return_value.name = f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job'
    mock_job = mock_dlp_instance.get_dlp_job.return_value
    mock_job.name = f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job'
    mock_job.state = google.cloud.dlp_v2.DlpJob.JobState.DONE
    file = open(os.path.join(RESOURCE_DIRECTORY, 'test.txt'), 'r')
    data = file.read()
    number_of_characters = len(data)
    mock_job.inspect_details.result.processed_bytes = number_of_characters
    mock_job.inspect_details.result.info_type_stats.info_type.name = 'EMAIL_ADDRESS'
    finding = mock_job.inspect_details.result.info_type_stats.info_type
    mock_job.inspect_details.result.info_type_stats = [MagicMock(info_type=finding, count=1)]
    inspect_content.inspect_gcs_send_to_scc(GCLOUD_PROJECT, f'{TEST_BUCKET_NAME}//test.txt', ['EMAIL_ADDRESS'], 100)
    (out, _) = capsys.readouterr()
    assert 'Info type: EMAIL_ADDRESS' in out
    mock_dlp_instance.create_dlp_job.assert_called_once()
    mock_dlp_instance.get_dlp_job.assert_called_once()