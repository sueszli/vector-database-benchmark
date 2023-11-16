import os
import re
from time import sleep
from unittest import mock
from unittest.mock import MagicMock
import uuid
from _pytest.capture import CaptureFixture
import backoff
import google.auth.transport.requests
from google.cloud import language_v1
from google.cloud.api_keys_v2 import Key
import pytest
import authenticate_with_api_key
import create_api_key
import delete_api_key
import lookup_api_key
import restrict_api_key_android
import restrict_api_key_api
import restrict_api_key_http
import restrict_api_key_ios
import restrict_api_key_server
(CREDENTIALS, PROJECT) = google.auth.default()
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

@pytest.fixture(scope='session')
def api_key() -> Key:
    if False:
        while True:
            i = 10
    suffix = uuid.uuid4().hex
    api_key = create_api_key.create_api_key(PROJECT, suffix)
    sleep(30)
    yield api_key
    delete_api_key.delete_api_key(PROJECT, get_key_id(api_key.name))

def get_key_id(api_key_name: str) -> str:
    if False:
        print('Hello World!')
    return api_key_name.rsplit('/')[-1]

def get_mock_sentiment_response() -> MagicMock:
    if False:
        print('Hello World!')
    response = mock.MagicMock(spec=language_v1.AnalyzeSentimentResponse)
    sentiment = mock.MagicMock(spec=language_v1.Sentiment)
    sentiment.score = 0.2
    sentiment.magnitude = 3.6
    response.document_sentiment = sentiment
    return mock.MagicMock(return_value=response)

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def test_authenticate_with_api_key(api_key: Key, capsys: CaptureFixture) -> None:
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.language_v1.LanguageServiceClient.analyze_sentiment', get_mock_sentiment_response()):
        authenticate_with_api_key.authenticate_with_api_key(PROJECT, api_key.key_string)
    (out, _) = capsys.readouterr()
    assert re.search('Successfully authenticated using the API key', out)

def test_lookup_api_key(api_key: Key, capsys: CaptureFixture) -> None:
    if False:
        print('Hello World!')
    lookup_api_key.lookup_api_key(api_key.key_string)
    (out, _) = capsys.readouterr()
    assert re.search(f'Successfully retrieved the API key name: {api_key.name}', out)

def test_restrict_api_key_android(api_key: Key, capsys: CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    restrict_api_key_android.restrict_api_key_android(PROJECT, get_key_id(api_key.name))
    (out, _) = capsys.readouterr()
    assert re.search(f'Successfully updated the API key: {api_key.name}', out)

def test_restrict_api_key_api(api_key: Key, capsys: CaptureFixture) -> None:
    if False:
        print('Hello World!')
    restrict_api_key_api.restrict_api_key_api(PROJECT, get_key_id(api_key.name))
    (out, _) = capsys.readouterr()
    assert re.search(f'Successfully updated the API key: {api_key.name}', out)

def test_restrict_api_key_http(api_key: Key, capsys: CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    restrict_api_key_http.restrict_api_key_http(PROJECT, get_key_id(api_key.name))
    (out, _) = capsys.readouterr()
    assert re.search(f'Successfully updated the API key: {api_key.name}', out)

def test_restrict_api_key_ios(api_key: Key, capsys: CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    restrict_api_key_ios.restrict_api_key_ios(PROJECT, get_key_id(api_key.name))
    (out, _) = capsys.readouterr()
    assert re.search(f'Successfully updated the API key: {api_key.name}', out)

def test_restrict_api_key_server(api_key: Key, capsys: CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    restrict_api_key_server.restrict_api_key_server(PROJECT, get_key_id(api_key.name))
    (out, _) = capsys.readouterr()
    assert re.search(f'Successfully updated the API key: {api_key.name}', out)