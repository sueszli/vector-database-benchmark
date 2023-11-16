import pytest
from unittest.mock import Mock
from superagi.llms.google_palm import GooglePalm
from superagi.llms.hugging_face import HuggingFace
from superagi.llms.llm_model_factory import get_model, build_model_with_api_key
from superagi.llms.openai import OpenAi
from superagi.llms.replicate import Replicate

@pytest.fixture
def mock_openai():
    if False:
        return 10
    return Mock(spec=OpenAi)

@pytest.fixture
def mock_replicate():
    if False:
        while True:
            i = 10
    return Mock(spec=Replicate)

@pytest.fixture
def mock_google_palm():
    if False:
        for i in range(10):
            print('nop')
    return Mock(spec=GooglePalm)

@pytest.fixture
def mock_hugging_face():
    if False:
        print('Hello World!')
    return Mock(spec=HuggingFace)

@pytest.fixture
def mock_replicate():
    if False:
        for i in range(10):
            print('nop')
    return Mock(spec=Replicate)

@pytest.fixture
def mock_google_palm():
    if False:
        return 10
    return Mock(spec=GooglePalm)

@pytest.fixture
def mock_hugging_face():
    if False:
        print('Hello World!')
    return Mock(spec=HuggingFace)

def test_build_model_with_openai(mock_openai, monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr('superagi.llms.llm_model_factory.OpenAi', mock_openai)
    model = build_model_with_api_key('OpenAi', 'fake_key')
    mock_openai.assert_called_once_with(api_key='fake_key')
    assert isinstance(model, Mock)

def test_build_model_with_replicate(mock_replicate, monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr('superagi.llms.llm_model_factory.Replicate', mock_replicate)
    model = build_model_with_api_key('Replicate', 'fake_key')
    mock_replicate.assert_called_once_with(api_key='fake_key')
    assert isinstance(model, Mock)

def test_build_model_with_openai(mock_openai, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr('superagi.llms.llm_model_factory.OpenAi', mock_openai)
    model = build_model_with_api_key('OpenAi', 'fake_key')
    mock_openai.assert_called_once_with(api_key='fake_key')
    assert isinstance(model, Mock)

def test_build_model_with_replicate(mock_replicate, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr('superagi.llms.llm_model_factory.Replicate', mock_replicate)
    model = build_model_with_api_key('Replicate', 'fake_key')
    mock_replicate.assert_called_once_with(api_key='fake_key')
    assert isinstance(model, Mock)

def test_build_model_with_google_palm(mock_google_palm, monkeypatch):
    if False:
        return 10
    monkeypatch.setattr('superagi.llms.llm_model_factory.GooglePalm', mock_google_palm)
    model = build_model_with_api_key('Google Palm', 'fake_key')
    mock_google_palm.assert_called_once_with(api_key='fake_key')
    assert isinstance(model, Mock)

def test_build_model_with_hugging_face(mock_hugging_face, monkeypatch):
    if False:
        return 10
    monkeypatch.setattr('superagi.llms.llm_model_factory.HuggingFace', mock_hugging_face)
    model = build_model_with_api_key('Hugging Face', 'fake_key')
    mock_hugging_face.assert_called_once_with(api_key='fake_key')
    assert isinstance(model, Mock)

def test_build_model_with_unknown_provider(capsys):
    if False:
        while True:
            i = 10
    model = build_model_with_api_key('Unknown', 'fake_key')
    assert model is None
    captured = capsys.readouterr()
    assert 'Unknown provider.' in captured.out