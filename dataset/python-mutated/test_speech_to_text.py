from __future__ import annotations
from unittest.mock import MagicMock, Mock, patch
import pytest
from google.api_core.gapic_v1.method import DEFAULT
from google.cloud.speech_v1 import RecognizeResponse
from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.operators.speech_to_text import CloudSpeechToTextRecognizeSpeechOperator
PROJECT_ID = 'project-id'
GCP_CONN_ID = 'gcp-conn-id'
IMPERSONATION_CHAIN = ['ACCOUNT_1', 'ACCOUNT_2', 'ACCOUNT_3']
CONFIG = {'encoding': 'LINEAR16'}
AUDIO = {'uri': 'gs://bucket/object'}

class TestCloudSpeechToTextRecognizeSpeechOperator:

    @patch('airflow.providers.google.cloud.operators.speech_to_text.CloudSpeechToTextHook')
    def test_recognize_speech_green_path(self, mock_hook):
        if False:
            return 10
        mock_hook.return_value.recognize_speech.return_value = RecognizeResponse()
        CloudSpeechToTextRecognizeSpeechOperator(project_id=PROJECT_ID, gcp_conn_id=GCP_CONN_ID, config=CONFIG, audio=AUDIO, task_id='id', impersonation_chain=IMPERSONATION_CHAIN).execute(context=MagicMock())
        mock_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_hook.return_value.recognize_speech.assert_called_once_with(config=CONFIG, audio=AUDIO, retry=DEFAULT, timeout=None)

    @patch('airflow.providers.google.cloud.operators.speech_to_text.CloudSpeechToTextHook')
    def test_missing_config(self, mock_hook):
        if False:
            while True:
                i = 10
        mock_hook.return_value.recognize_speech.return_value = True
        with pytest.raises(AirflowException) as ctx:
            CloudSpeechToTextRecognizeSpeechOperator(project_id=PROJECT_ID, gcp_conn_id=GCP_CONN_ID, audio=AUDIO, task_id='id').execute(context={'task_instance': Mock()})
        err = ctx.value
        assert 'config' in str(err)
        mock_hook.assert_not_called()

    @patch('airflow.providers.google.cloud.operators.speech_to_text.CloudSpeechToTextHook')
    def test_missing_audio(self, mock_hook):
        if False:
            i = 10
            return i + 15
        mock_hook.return_value.recognize_speech.return_value = True
        with pytest.raises(AirflowException) as ctx:
            CloudSpeechToTextRecognizeSpeechOperator(project_id=PROJECT_ID, gcp_conn_id=GCP_CONN_ID, config=CONFIG, task_id='id').execute(context={'task_instance': Mock()})
        err = ctx.value
        assert 'audio' in str(err)
        mock_hook.assert_not_called()