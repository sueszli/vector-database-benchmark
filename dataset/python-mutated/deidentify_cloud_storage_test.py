import os
from unittest import mock
from unittest.mock import MagicMock
import deidentify_cloud_storage as deid
import google.cloud.dlp_v2
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
TXT_FILE = os.path.join(os.path.dirname(__file__), '../resources/test.txt')

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
def test_deidentify_cloud_storage(dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    mock_dlp_instance = dlp_client.return_value
    test_job = f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job'
    mock_dlp_instance.create_dlp_job.return_value.name = test_job
    mock_job = mock_dlp_instance.get_dlp_job.return_value
    mock_job.name = test_job
    mock_job.state = google.cloud.dlp_v2.DlpJob.JobState.DONE
    file = open(TXT_FILE, 'r')
    data = file.read()
    number_of_characters = len(data)
    mock_job.inspect_details.result.processed_bytes = number_of_characters
    mock_job.inspect_details.result.info_type_stats.info_type.name = 'EMAIL_ADDRESS'
    finding = mock_job.inspect_details.result.info_type_stats.info_type
    mock_job.inspect_details.result.info_type_stats = [MagicMock(info_type=finding, count=1)]
    deid.deidentify_cloud_storage(GCLOUD_PROJECT, 'input_bucket', 'output_bucket', ['EMAIL_ADDRESS', 'PERSON_NAME', 'PHONE_NUMBER'], 'deidentify_template_name', 'structured_deidentify_template_name', 'image_redaction_template_name', 'DATASET_ID', 'TABLE_ID', timeout=1)
    (out, _) = capsys.readouterr()
    assert test_job in out
    assert 'Processed Bytes' in out
    assert 'Info type: EMAIL_ADDRESS' in out
    create_job_args = mock_dlp_instance.create_dlp_job.call_args
    mock_dlp_instance.create_dlp_job.assert_called_once_with(request=create_job_args.kwargs['request'])
    mock_dlp_instance.get_dlp_job.assert_called_once_with(request={'name': test_job})