import os
from unittest import mock
from unittest.mock import MagicMock
import google.cloud.dlp_v2
import inspect_bigquery_send_to_scc as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

@mock.patch('google.cloud.dlp_v2.DlpServiceClient')
def test_inspect_bigquery_send_to_scc(dlp_client: MagicMock, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    mock_dlp_instance = dlp_client.return_value
    mock_dlp_instance.create_dlp_job.return_value.name = f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job'
    mock_job = mock_dlp_instance.get_dlp_job.return_value
    mock_job.name = f'projects/{GCLOUD_PROJECT}/dlpJobs/test_job'
    mock_job.state = google.cloud.dlp_v2.DlpJob.JobState.DONE
    mock_job.inspect_details.result.info_type_stats.info_type.name = 'PERSON_NAME'
    finding = mock_job.inspect_details.result.info_type_stats.info_type
    mock_job.inspect_details.result.info_type_stats = [MagicMock(info_type=finding, count=1)]
    inspect_content.inspect_bigquery_send_to_scc(GCLOUD_PROJECT, ['PERSON_NAME'])
    (out, _) = capsys.readouterr()
    assert 'Info type: PERSON_NAME' in out
    mock_dlp_instance.create_dlp_job.assert_called_once()
    mock_dlp_instance.get_dlp_job.assert_called_once()