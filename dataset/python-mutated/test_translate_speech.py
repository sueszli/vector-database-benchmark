from __future__ import annotations
from unittest import mock
import pytest
from google.cloud.speech_v1 import RecognizeResponse, SpeechRecognitionAlternative, SpeechRecognitionResult
from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.operators.translate_speech import CloudTranslateSpeechOperator
GCP_CONN_ID = 'google_cloud_default'
IMPERSONATION_CHAIN = ['ACCOUNT_1', 'ACCOUNT_2', 'ACCOUNT_3']

class TestCloudTranslateSpeech:

    @mock.patch('airflow.providers.google.cloud.operators.translate_speech.CloudSpeechToTextHook')
    @mock.patch('airflow.providers.google.cloud.operators.translate_speech.CloudTranslateHook')
    def test_minimal_green_path(self, mock_translate_hook, mock_speech_hook):
        if False:
            while True:
                i = 10
        mock_speech_hook.return_value.recognize_speech.return_value = RecognizeResponse(results=[SpeechRecognitionResult(alternatives=[SpeechRecognitionAlternative(transcript='test speech recognition result')])])
        mock_translate_hook.return_value.translate.return_value = [{'translatedText': 'sprawdzić wynik rozpoznawania mowy', 'detectedSourceLanguage': 'en', 'model': 'base', 'input': 'test speech recognition result'}]
        op = CloudTranslateSpeechOperator(audio={'uri': 'gs://bucket/object'}, config={'encoding': 'LINEAR16'}, target_language='pl', format_='text', source_language=None, model='base', gcp_conn_id=GCP_CONN_ID, task_id='id', impersonation_chain=IMPERSONATION_CHAIN)
        context = mock.MagicMock()
        return_value = op.execute(context=context)
        mock_speech_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_translate_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_speech_hook.return_value.recognize_speech.assert_called_once_with(audio={'uri': 'gs://bucket/object'}, config={'encoding': 'LINEAR16'})
        mock_translate_hook.return_value.translate.assert_called_once_with(values='test speech recognition result', target_language='pl', format_='text', source_language=None, model='base')
        assert [{'translatedText': 'sprawdzić wynik rozpoznawania mowy', 'detectedSourceLanguage': 'en', 'model': 'base', 'input': 'test speech recognition result'}] == return_value

    @mock.patch('airflow.providers.google.cloud.operators.translate_speech.CloudSpeechToTextHook')
    @mock.patch('airflow.providers.google.cloud.operators.translate_speech.CloudTranslateHook')
    def test_bad_recognition_response(self, mock_translate_hook, mock_speech_hook):
        if False:
            print('Hello World!')
        mock_speech_hook.return_value.recognize_speech.return_value = RecognizeResponse(results=[SpeechRecognitionResult()])
        op = CloudTranslateSpeechOperator(audio={'uri': 'gs://bucket/object'}, config={'encoding': 'LINEAR16'}, target_language='pl', format_='text', source_language=None, model='base', gcp_conn_id=GCP_CONN_ID, task_id='id')
        with pytest.raises(AirflowException) as ctx:
            op.execute(context=None)
        err = ctx.value
        assert "it should contain 'alternatives' field" in str(err)
        mock_speech_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=None)
        mock_translate_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=None)
        mock_speech_hook.return_value.recognize_speech.assert_called_once_with(audio={'uri': 'gs://bucket/object'}, config={'encoding': 'LINEAR16'})
        mock_translate_hook.return_value.translate.assert_not_called()