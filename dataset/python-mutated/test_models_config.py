from unittest.mock import MagicMock, patch
import pytest
from superagi.models.models_config import ModelsConfig

@pytest.fixture
def mock_session():
    if False:
        while True:
            i = 10
    return MagicMock()

def test_create_models_config(mock_session):
    if False:
        print('Hello World!')
    provider = 'example_provider'
    api_key = 'example_api_key'
    org_id = 1
    mock_session.query.return_value.filter_by.return_value.first.return_value = None
    model_config = ModelsConfig(provider=provider, api_key=api_key, org_id=org_id)
    mock_session.add(model_config)
    mock_session.add.assert_called_once_with(model_config)

def test_repr_method_models_config(mock_session):
    if False:
        print('Hello World!')
    provider = 'example_provider'
    api_key = 'example_api_key'
    org_id = 1
    mock_session.query.return_value.filter_by.return_value.first.return_value = None
    model_config = ModelsConfig(provider=provider, api_key=api_key, org_id=org_id)
    model_config_repr = repr(model_config)
    assert model_config_repr == f'ModelsConfig(id=None, provider={provider}, org_id={org_id})'

def test_fetch_model_by_id(mock_session):
    if False:
        print('Hello World!')
    organisation_id = 1
    model_provider_id = 1
    mock_model = MagicMock()
    mock_model.provider = 'some_provider'
    mock_session.query.return_value.filter.return_value.first.return_value = mock_model
    model = ModelsConfig.fetch_model_by_id(mock_session, organisation_id, model_provider_id)
    assert model == {'provider': 'some_provider'}

def test_fetch_model_by_id_marketplace(mock_session):
    if False:
        print('Hello World!')
    model_provider_id = 1
    mock_model = MagicMock()
    mock_model.provider = 'some_provider'
    mock_session.query.return_value.filter.return_value.first.return_value = mock_model
    model = ModelsConfig.fetch_model_by_id_marketplace(mock_session, model_provider_id)
    assert model == {'provider': 'some_provider'}