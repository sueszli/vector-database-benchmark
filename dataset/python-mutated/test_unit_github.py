from unittest import TestCase, mock
import pytest
from custom_auth.oauth.exceptions import GithubError
from custom_auth.oauth.github import NON_200_ERROR_MESSAGE, GithubUser

class GithubUserTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.test_client_id = 'test-client-id'
        self.test_client_secret = 'test-client-secret'
        self.mock_requests = mock.patch('custom_auth.oauth.github.requests').start()

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        self.mock_requests.stop()

    def test_get_access_token_success(self):
        if False:
            for i in range(10):
                print('nop')
        test_code = 'abc123'
        expected_access_token = 'access-token'
        self.mock_requests.post.return_value = mock.MagicMock(text=f'access_token={expected_access_token}&scope=user&token_type=bearer', status_code=200)
        github_user = GithubUser(test_code, client_id=self.test_client_id, client_secret=self.test_client_secret)
        assert github_user.access_token == expected_access_token
        assert self.mock_requests.post.call_count == 1
        request_calls = self.mock_requests.post.call_args
        assert request_calls[1]['data']['code'] == test_code

    def test_get_access_token_fail_non_200(self):
        if False:
            return 10
        invalid_code = 'invalid'
        status_code = 400
        self.mock_requests.post.return_value = mock.MagicMock(status_code=status_code)
        with pytest.raises(GithubError) as e:
            GithubUser(invalid_code, client_id=self.test_client_id, client_secret=self.test_client_secret)
        assert NON_200_ERROR_MESSAGE.format(status_code) in str(e)

    def test_get_access_token_fail_token_expired(self):
        if False:
            return 10
        invalid_code = 'invalid'
        error_description = 'there+was+an+error'
        self.mock_requests.post.return_value = mock.MagicMock(text=f'error=bad_verification_code&error_description={error_description}', status_code=200)
        with pytest.raises(GithubError) as e:
            GithubUser(invalid_code, client_id=self.test_client_id, client_secret=self.test_client_secret)
        assert error_description.replace('+', ' ') in str(e)

    def test_get_user_name_and_id(self):
        if False:
            print('Hello World!')
        self.mock_requests.post.return_value = mock.MagicMock(status_code=200, text='access_token=123456')
        mock_response = mock.MagicMock(status_code=200)
        self.mock_requests.get.return_value = mock_response
        mock_response.json.return_value = {'name': 'tommy tester', 'id': 123456}
        github_user = GithubUser('test-code', client_id=self.test_client_id, client_secret=self.test_client_secret)
        user_name_and_id = github_user._get_user_name_and_id()
        assert user_name_and_id == {'first_name': 'tommy', 'last_name': 'tester', 'github_user_id': 123456}

    def test_get_primary_email(self):
        if False:
            for i in range(10):
                print('nop')
        self.mock_requests.post.return_value = mock.MagicMock(status_code=200, text='access_token=123456')
        mock_response = mock.MagicMock(status_code=200)
        self.mock_requests.get.return_value = mock_response
        verified_emails = [{'email': f'tommy_tester@example_{i}.com', 'verified': True, 'visibility': None, 'primary': False} for i in range(5)]
        verified_emails[3]['primary'] = True
        mock_response.json.return_value = verified_emails
        github_user = GithubUser('test-code', client_id=self.test_client_id, client_secret=self.test_client_secret)
        primary_email = github_user._get_primary_email()
        assert primary_email == verified_emails[3]['email']