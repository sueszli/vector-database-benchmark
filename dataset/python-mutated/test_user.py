from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from main import app
from superagi.models.user import User
client = TestClient(app)

@pytest.fixture
def authenticated_user():
    if False:
        i = 10
        return i + 15
    user = User()
    user.id = 1
    user.username = 'testuser'
    user.email = 'super6@agi.com'
    user.first_login_source = None
    user.token = 'mock-jwt-token'
    return user

def test_update_first_login_source(authenticated_user):
    if False:
        for i in range(10):
            print('nop')
    with patch('superagi.helper.auth.db') as mock_auth_db:
        source = 'github'
        mock_auth_db.session.query.return_value.filter.return_value.first.return_value = authenticated_user
        response = client.post(f'users/first_login_source/{source}', headers={'Authorization': f'Bearer {authenticated_user.token}'})
        assert response.status_code == 200
        assert 'first_login_source' in response.json()
        assert response.json()['first_login_source'] == 'github'