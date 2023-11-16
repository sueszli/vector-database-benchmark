from unittest.mock import create_autospec
from sqlalchemy.orm import Session
from superagi.models.api_key import ApiKey

def test_get_by_org_id():
    if False:
        for i in range(10):
            print('nop')
    session = create_autospec(Session)
    org_id = 1
    mock_api_keys = [ApiKey(id=1, org_id=org_id, key='key1', is_expired=False), ApiKey(id=2, org_id=org_id, key='key2', is_expired=False)]
    session.query.return_value.filter.return_value.all.return_value = mock_api_keys
    api_keys = ApiKey.get_by_org_id(session, org_id)
    assert api_keys == mock_api_keys

def test_get_by_id():
    if False:
        for i in range(10):
            print('nop')
    session = create_autospec(Session)
    api_key_id = 1
    mock_api_key = ApiKey(id=api_key_id, org_id=1, key='key1', is_expired=False)
    session.query.return_value.filter.return_value.first.return_value = mock_api_key
    api_key = ApiKey.get_by_id(session, api_key_id)
    assert api_key == mock_api_key

def test_delete_by_id():
    if False:
        return 10
    session = create_autospec(Session)
    api_key_id = 1
    mock_api_key = ApiKey(id=api_key_id, org_id=1, key='key1', is_expired=False)
    session.query.return_value.filter.return_value.first.return_value = mock_api_key
    ApiKey.delete_by_id(session, api_key_id)
    assert mock_api_key.is_expired == True
    session.commit.assert_called_once()
    session.flush.assert_called_once()

def test_edit_by_id():
    if False:
        return 10
    session = create_autospec(Session)
    api_key_id = 1
    new_name = 'New Name'
    mock_api_key = ApiKey(id=api_key_id, org_id=1, key='key1', is_expired=False)
    session.query.return_value.filter.return_value.first.return_value = mock_api_key
    ApiKey.update_api_key(session, api_key_id, new_name)
    assert mock_api_key.name == new_name
    session.commit.assert_called_once()
    session.flush.assert_called_once()