from unittest import mock
import flask
from google.cloud import documentai
import pytest
import document_function
_BIGQUERY_REQUEST_JSON = {'calls': [['https://storage.googleapis.com/bucket/apple', 'application/pdf'], ['https://storage.googleapis.com/bucket/banana', 'application/pdf']]}
_BIGQUERY_RESPONSE_JSON = {'replies': [{'text': 'apple'}, {'text': 'banana'}]}

@pytest.fixture(scope='module')
def app() -> flask.Flask:
    if False:
        while True:
            i = 10
    return flask.Flask(__name__)

@mock.patch('document_function.urllib.request')
@mock.patch('document_function.documentai')
def test_document_function(mock_documentai: object, mock_request: object, app: flask.Flask) -> None:
    if False:
        print('Hello World!')
    mock_request.urlopen = mock.Mock(read=mock.Mock(return_value=b'filedata'))
    process_document_mock = mock.Mock(side_effect=[documentai.ProcessResponse({'document': {'text': 'apple'}}), documentai.ProcessResponse({'document': {'text': 'banana'}})])
    mock_documentai.DocumentProcessorServiceClient = mock.Mock(return_value=mock.Mock(process_document=process_document_mock))
    with app.test_request_context(json=_BIGQUERY_REQUEST_JSON):
        response = document_function.document_ocr(flask.request)
        assert response.status_code == 200
        assert response.get_json() == _BIGQUERY_RESPONSE_JSON

@mock.patch('document_function.urllib.request')
@mock.patch('document_function.documentai')
def test_document_function_error(mock_documentai: object, mock_request: object, app: flask.Flask) -> None:
    if False:
        while True:
            i = 10
    mock_request.urlopen = mock.Mock(read=mock.Mock(return_value=b'filedata'))
    process_document_mock = mock.Mock(side_effect=Exception('API error'))
    mock_documentai.DocumentProcessorServiceClient = mock.Mock(return_value=mock.Mock(process_document=process_document_mock))
    with app.test_request_context(json=_BIGQUERY_REQUEST_JSON):
        response = document_function.document_ocr(flask.request)
        assert response.status_code == 400
        assert 'API error' in str(response.get_data())