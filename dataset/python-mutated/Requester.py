import contextlib
from datetime import datetime, timedelta, timezone
from unittest import mock
import github
from . import Framework
from .GithubIntegration import APP_ID, PRIVATE_KEY
REPO_NAME = 'PyGithub/PyGithub'

class Requester(Framework.TestCase):
    logger = None

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.logger = mock.MagicMock()
        github.Requester.Requester.injectLogger(self.logger)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        github.Requester.Requester.resetLogger()
        super().tearDown()

    def testRecreation(self):
        if False:
            return 10

        class TestAuth(github.Auth.AppAuth):
            pass
        auth = TestAuth(123, 'key')
        requester = github.Requester.Requester(auth=auth, base_url='https://base.url', timeout=1, user_agent='user agent', per_page=123, verify=False, retry=3, pool_size=5, seconds_between_requests=1.2, seconds_between_writes=3.4)
        kwargs = requester.kwargs
        self.assertEqual(kwargs.keys(), github.Requester.Requester.__init__.__annotations__.keys())
        self.assertEqual(kwargs, dict(auth=auth, base_url='https://base.url', timeout=1, user_agent='user agent', per_page=123, verify=False, retry=3, pool_size=5, seconds_between_requests=1.2, seconds_between_writes=3.4))
        copy = github.Requester.Requester(**kwargs)
        self.assertEqual(copy.kwargs, kwargs)
        gh = github.Github(**kwargs)
        self.assertEqual(gh._Github__requester.kwargs, kwargs)
        gi = github.GithubIntegration(**kwargs)
        self.assertEqual(gi._GithubIntegration__requester.kwargs, kwargs)

    def testWithAuth(self):
        if False:
            i = 10
            return i + 15

        class TestAuth(github.Auth.AppAuth):
            pass
        auth = TestAuth(123, 'key')
        requester = github.Requester.Requester(auth=auth, base_url='https://base.url', timeout=1, user_agent='user agent', per_page=123, verify=False, retry=3, pool_size=5, seconds_between_requests=1.2, seconds_between_writes=3.4)
        auth2 = TestAuth(456, 'key2')
        copy = requester.withAuth(auth2)
        self.assertEqual(copy.kwargs, dict(auth=auth2, base_url='https://base.url', timeout=1, user_agent='user agent', per_page=123, verify=False, retry=3, pool_size=5, seconds_between_requests=1.2, seconds_between_writes=3.4))

    def testCloseGithub(self):
        if False:
            print('Hello World!')
        mocked_connection = mock.MagicMock()
        mocked_custom_connection = mock.MagicMock()
        with github.Github() as gh:
            requester = gh._Github__requester
            requester._Requester__connection = mocked_connection
            requester._Requester__custom_connections.append(mocked_custom_connection)
        mocked_connection.close.assert_called_once_with()
        mocked_custom_connection.close.assert_called_once_with()
        self.assertIsNone(requester._Requester__connection)

    def testCloseGithubIntegration(self):
        if False:
            for i in range(10):
                print('nop')
        mocked_connection = mock.MagicMock()
        mocked_custom_connection = mock.MagicMock()
        auth = github.Auth.AppAuth(APP_ID, PRIVATE_KEY)
        with github.GithubIntegration(auth=auth) as gi:
            requester = gi._GithubIntegration__requester
            requester._Requester__connection = mocked_connection
            requester._Requester__custom_connections.append(mocked_custom_connection)
        mocked_connection.close.assert_called_once_with()
        mocked_custom_connection.close.assert_called_once_with()
        self.assertIsNone(requester._Requester__connection)

    def testLoggingRedirection(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.g.get_repo('EnricoMi/test').name, 'test-renamed')
        self.logger.info.assert_called_once_with('Following Github server redirection from /repos/EnricoMi/test to /repositories/638123443')

    def testBaseUrlSchemeRedirection(self):
        if False:
            print('Hello World!')
        gh = github.Github(base_url='http://api.github.com')
        with self.assertRaises(RuntimeError) as exc:
            gh.get_repo('PyGithub/PyGithub')
        self.assertEqual(exc.exception.args, ('Github server redirected from http protocol to https, please correct your Github server URL via base_url: Github(base_url=...)',))

    def testBaseUrlHostRedirection(self):
        if False:
            return 10
        gh = github.Github(base_url='https://www.github.com')
        with self.assertRaises(RuntimeError) as exc:
            gh.get_repo('PyGithub/PyGithub')
        self.assertEqual(exc.exception.args, ('Github server redirected from host www.github.com to github.com, please correct your Github server URL via base_url: Github(base_url=...)',))

    def testBaseUrlPortRedirection(self):
        if False:
            return 10
        gh = github.Github(base_url='https://api.github.com')
        with self.assertRaises(RuntimeError) as exc:
            gh.get_repo('PyGithub/PyGithub')
        self.assertEqual(exc.exception.args, ('Requested https://api.github.com/repos/PyGithub/PyGithub but server redirected to https://api.github.com:443/repos/PyGithub/PyGithub, you may need to correct your Github server URL via base_url: Github(base_url=...)',))

    def testBaseUrlPrefixRedirection(self):
        if False:
            while True:
                i = 10
        gh = github.Github(base_url='https://api.github.com/api/v3')
        self.assertEqual(gh.get_repo('PyGithub/PyGithub').name, 'PyGithub')
        self.logger.info.assert_called_once_with('Following Github server redirection from /api/v3/repos/PyGithub/PyGithub to /repos/PyGithub/PyGithub')
    PrimaryRateLimitErrors = ["API rate limit exceeded for x.x.x.x. (But here's the good news: Authenticated requests get a higher rate limit. Check out the documentation for more details.)"]
    SecondaryRateLimitErrors = ['You have triggered an abuse detection mechanism. Please wait a few minutes before you try again.', 'You have triggered an abuse detection mechanism and have been temporarily blocked from content creation. Please retry your request again later.You have exceeded a secondary rate limit and have been temporarily blocked from content creation. Please retry your request again later.', 'You have exceeded a secondary rate limit. Please wait a few minutes before you try again.', 'Something else here. Please wait a few minutes before you try again.']
    OtherErrors = ['User does not exist or is not a member of the organization']

    def testIsRateLimitError(self):
        if False:
            i = 10
            return i + 15
        for message in self.PrimaryRateLimitErrors + self.SecondaryRateLimitErrors:
            self.assertTrue(github.Requester.Requester.isRateLimitError(message), message)
        for message in self.OtherErrors:
            self.assertFalse(github.Requester.Requester.isRateLimitError(message), message)

    def testIsPrimaryRateLimitError(self):
        if False:
            while True:
                i = 10
        for message in self.PrimaryRateLimitErrors:
            self.assertTrue(github.Requester.Requester.isPrimaryRateLimitError(message), message)
        for message in self.OtherErrors + self.SecondaryRateLimitErrors:
            self.assertFalse(github.Requester.Requester.isPrimaryRateLimitError(message), message)

    def testIsSecondaryRateLimitError(self):
        if False:
            while True:
                i = 10
        for message in self.SecondaryRateLimitErrors:
            self.assertTrue(github.Requester.Requester.isSecondaryRateLimitError(message), message)
        for message in self.OtherErrors + self.PrimaryRateLimitErrors:
            self.assertFalse(github.Requester.Requester.isSecondaryRateLimitError(message), message)

    def assertException(self, exception, exception_type, status, data, headers, string):
        if False:
            print('Hello World!')
        self.assertIsInstance(exception, exception_type)
        self.assertEqual(exception.status, status)
        if data is None:
            self.assertIsNone(exception.data)
        else:
            self.assertEqual(exception.data, data)
        self.assertEqual(exception.headers, headers)
        self.assertEqual(str(exception), string)

    def testShouldCreateBadCredentialsException(self):
        if False:
            print('Hello World!')
        exc = self.g._Github__requester.createException(401, {'header': 'value'}, {'message': 'Bad credentials'})
        self.assertException(exc, github.BadCredentialsException, 401, {'message': 'Bad credentials'}, {'header': 'value'}, '401 {"message": "Bad credentials"}')

    def testShouldCreateTwoFactorException(self):
        if False:
            for i in range(10):
                print('nop')
        exc = self.g._Github__requester.createException(401, {'x-github-otp': 'required; app'}, {'message': 'Must specify two-factor authentication OTP code.', 'documentation_url': 'https://developer.github.com/v3/auth#working-with-two-factor-authentication'})
        self.assertException(exc, github.TwoFactorException, 401, {'message': 'Must specify two-factor authentication OTP code.', 'documentation_url': 'https://developer.github.com/v3/auth#working-with-two-factor-authentication'}, {'x-github-otp': 'required; app'}, '401 {"message": "Must specify two-factor authentication OTP code.", "documentation_url": "https://developer.github.com/v3/auth#working-with-two-factor-authentication"}')

    def testShouldCreateBadUserAgentException(self):
        if False:
            i = 10
            return i + 15
        exc = self.g._Github__requester.createException(403, {'header': 'value'}, {'message': 'Missing or invalid User Agent string'})
        self.assertException(exc, github.BadUserAgentException, 403, {'message': 'Missing or invalid User Agent string'}, {'header': 'value'}, '403 {"message": "Missing or invalid User Agent string"}')

    def testShouldCreateRateLimitExceededException(self):
        if False:
            return 10
        for message in self.PrimaryRateLimitErrors + self.SecondaryRateLimitErrors:
            with self.subTest(message=message):
                exc = self.g._Github__requester.createException(403, {'header': 'value'}, {'message': message})
                self.assertException(exc, github.RateLimitExceededException, 403, {'message': message}, {'header': 'value'}, f'403 {{"message": "{message}"}}')

    def testShouldCreateUnknownObjectException(self):
        if False:
            print('Hello World!')
        exc = self.g._Github__requester.createException(404, {'header': 'value'}, {'message': 'Not Found'})
        self.assertException(exc, github.UnknownObjectException, 404, {'message': 'Not Found'}, {'header': 'value'}, '404 {"message": "Not Found"}')

    def testShouldCreateGithubException(self):
        if False:
            return 10
        for status in range(400, 600):
            with self.subTest(status=status):
                exc = self.g._Github__requester.createException(status, {'header': 'value'}, {'message': 'Something unknown'})
                self.assertException(exc, github.GithubException, status, {'message': 'Something unknown'}, {'header': 'value'}, f'{status} {{"message": "Something unknown"}}')

    def testShouldCreateExceptionWithoutMessage(self):
        if False:
            i = 10
            return i + 15
        for status in range(400, 600):
            with self.subTest(status=status):
                exc = self.g._Github__requester.createException(status, {}, {})
                self.assertException(exc, github.GithubException, status, {}, {}, f'{status} {{}}')

    def testShouldCreateExceptionWithoutOutput(self):
        if False:
            print('Hello World!')
        for status in range(400, 600):
            with self.subTest(status=status):
                exc = self.g._Github__requester.createException(status, {}, None)
                self.assertException(exc, github.GithubException, status, None, {}, f'{status}')

class RequesterThrottleTestCase(Framework.TestCase):
    per_page = 10
    mock_time = [datetime.now(timezone.utc)]

    def sleep(self, seconds):
        if False:
            while True:
                i = 10
        self.mock_time[0] = self.mock_time[0] + timedelta(seconds=seconds)

    def now(self, tz=None):
        if False:
            return 10
        return self.mock_time[0]

    @contextlib.contextmanager
    def mock_sleep(self):
        if False:
            print('Hello World!')
        with mock.patch('github.Requester.time.sleep', side_effect=self.sleep) as sleep_mock, mock.patch('github.Requester.datetime') as datetime_mock:
            datetime_mock.now = self.now
            yield sleep_mock

class RequesterUnThrottled(RequesterThrottleTestCase):

    def testShouldNotDeferRequests(self):
        if False:
            return 10
        with self.mock_sleep() as sleep_mock:
            repository = self.g.get_repo(REPO_NAME)
            releases = list(repository.get_releases())
            self.assertEqual(len(releases), 30)
        sleep_mock.assert_not_called()

class RequesterThrottled(RequesterThrottleTestCase):
    seconds_between_requests = 1.0
    seconds_between_writes = 3.0

    def testShouldDeferRequests(self):
        if False:
            while True:
                i = 10
        with self.mock_sleep() as sleep_mock:
            repository = self.g.get_repo(REPO_NAME)
            releases = [release for release in repository.get_releases()]
            self.assertEqual(len(releases), 30)
        self.assertEqual(sleep_mock.call_args_list, [mock.call(1), mock.call(1), mock.call(1)])

    def testShouldDeferWrites(self):
        if False:
            while True:
                i = 10
        with self.mock_sleep() as sleep_mock:
            user = self.g.get_user()
            emails = user.get_emails()
            self.assertEqual([item.email for item in emails], ['vincent@vincent-jacques.net', 'github.com@vincent-jacques.net'])
            self.assertTrue(emails[0].primary)
            self.assertTrue(emails[0].verified)
            self.assertEqual(emails[0].visibility, 'private')
            user.add_to_emails('1@foobar.com', '2@foobar.com')
            self.assertEqual([item.email for item in user.get_emails()], ['vincent@vincent-jacques.net', '1@foobar.com', '2@foobar.com', 'github.com@vincent-jacques.net'])
            user.remove_from_emails('1@foobar.com', '2@foobar.com')
            self.assertEqual([item.email for item in user.get_emails()], ['vincent@vincent-jacques.net', 'github.com@vincent-jacques.net'])
        self.assertEqual(sleep_mock.call_args_list, [mock.call(1), mock.call(1), mock.call(2), mock.call(1)])