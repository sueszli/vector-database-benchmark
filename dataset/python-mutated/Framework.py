import contextlib
import io
import json
import os
import traceback
import unittest
import warnings
from typing import Optional
import httpretty
from requests.structures import CaseInsensitiveDict
from urllib3.util import Url
import github
from github import Consts
APP_PRIVATE_KEY = '\n-----BEGIN RSA PRIVATE KEY-----\nMIICXAIBAAKBgQC+5ePolLv6VcWLp2f17g6r6vHl+eoLuodOOfUl8JK+MVmvXbPa\nxDy0SS0pQhwTOMtB0VdSt++elklDCadeokhEoGDQp411o+kiOhzLxfakp/kewf4U\nHJnu4M/A2nHmxXVe2lzYnZvZHX5BM4SJo5PGdr0Ue2JtSXoAtYr6qE9maQIDAQAB\nAoGAFhOJ7sy8jG+837Clcihso+8QuHLVYTPaD+7d7dxLbBlS8NfaQ9Nr3cGUqm/N\nxV9NCjiGa7d/y4w/vrPwGh6UUsA+CvndwDgBd0S3WgIdWvAvHM8wKgNh/GBLLzhT\nBg9BouRUzcT1MjAnkGkWqqCAgN7WrCSUMLt57TNleNWfX90CQQDjvVKTT3pOiavD\n3YcLxwkyeGd0VMvKiS4nV0XXJ97cGXs2GpOGXldstDTnF5AnB6PbukdFLHpsx4sW\nHft3LRWnAkEA1pY15ke08wX6DZVXy7zuQ2izTrWSGySn7B41pn55dlKpttjHeutA\n3BEQKTFvMhBCphr8qST7Wf1SR9FgO0tFbwJAEhHji2yy96hUyKW7IWQZhrem/cP8\np4Va9CQolnnDZRNgg1p4eiDiLu3dhLiJ547joXuWTBbLX/Y1Qvv+B+a74QJBAMCW\nO3WbMZlS6eK6//rIa4ZwN00SxDg8I8FUM45jwBsjgVGrKQz2ilV3sutlhIiH82kk\nm1Iq8LMJGYl/LkDJA10CQBV1C+Xu3ukknr7C4A/4lDCa6Xb27cr1HanY7i89A+Ab\neatdM6f/XVqWp8uPT9RggUV9TjppJobYGT2WrWJMkYw=\n-----END RSA PRIVATE KEY-----\n'

def readLine(file_):
    if False:
        while True:
            i = 10
    line = file_.readline()
    if isinstance(line, bytes):
        line = line.decode('utf-8')
    return line.strip()

class FakeHttpResponse:

    def __init__(self, status, headers, output):
        if False:
            print('Hello World!')
        self.status = status
        self.__headers = headers
        self.__output = output

    def getheaders(self):
        if False:
            return 10
        return self.__headers

    def read(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__output

def fixAuthorizationHeader(headers):
    if False:
        i = 10
        return i + 15
    if 'Authorization' in headers:
        if headers['Authorization'].endswith('ZmFrZV9sb2dpbjpmYWtlX3Bhc3N3b3Jk'):
            pass
        elif headers['Authorization'].startswith('token '):
            headers['Authorization'] = 'token private_token_removed'
        elif headers['Authorization'].startswith('Basic '):
            headers['Authorization'] = 'Basic login_and_password_removed'
        elif headers['Authorization'].startswith('Bearer '):
            headers['Authorization'] = 'Bearer jwt_removed'

class RecordingConnection:

    def __init__(self, file, protocol, host, port, *args, **kwds):
        if False:
            while True:
                i = 10
        assert isinstance(file, io.TextIOBase)
        self.__file = file
        self.__protocol = protocol
        self.__host = host
        self.__port = port
        self.__cnx = self._realConnection(host, port, *args, **kwds)

    def request(self, verb, url, input, headers):
        if False:
            for i in range(10):
                print('nop')
        self.__cnx.request(verb, url, input, headers)
        anonymous_headers = headers.copy()
        fixAuthorizationHeader(anonymous_headers)
        self.__writeLine(self.__protocol)
        self.__writeLine(verb)
        self.__writeLine(self.__host)
        self.__writeLine(self.__port)
        self.__writeLine(url)
        self.__writeLine(anonymous_headers)
        self.__writeLine(str(input).replace('\n', '').replace('\r', ''))

    def getresponse(self):
        if False:
            i = 10
            return i + 15
        res = self.__cnx.getresponse()
        status = res.status
        headers = res.getheaders()
        output = res.read()
        self.__writeLine(status)
        self.__writeLine(list(headers))
        self.__writeLine(output)
        return FakeHttpResponse(status, headers, output)

    def close(self):
        if False:
            i = 10
            return i + 15
        self.__writeLine('')
        return self.__cnx.close()

    def __writeLine(self, line):
        if False:
            for i in range(10):
                print('nop')
        self.__file.write(str(line) + '\n')

class RecordingHttpConnection(RecordingConnection):
    _realConnection = github.Requester.HTTPRequestsConnectionClass

    def __init__(self, file, *args, **kwds):
        if False:
            while True:
                i = 10
        super().__init__(file, 'http', *args, **kwds)

class RecordingHttpsConnection(RecordingConnection):
    _realConnection = github.Requester.HTTPSRequestsConnectionClass

    def __init__(self, file, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(file, 'https', *args, **kwds)

class ReplayingConnection:

    def __init__(self, file, protocol, host, port, *args, **kwds):
        if False:
            while True:
                i = 10
        self.__file = file
        self.__protocol = protocol
        self.__host = host
        self.__port = port
        self.response_headers = CaseInsensitiveDict()
        self.__cnx = self._realConnection(host, port, *args, **kwds)

    def request(self, verb, url, input, headers):
        if False:
            for i in range(10):
                print('nop')
        full_url = Url(scheme=self.__protocol, host=self.__host, port=self.__port, path=url)
        httpretty.register_uri(verb, full_url.url, body=self.__request_callback)
        self.__cnx.request(verb, url, input, headers)

    def __readNextRequest(self, verb, url, input, headers):
        if False:
            for i in range(10):
                print('nop')
        fixAuthorizationHeader(headers)
        assert self.__protocol == readLine(self.__file)
        assert verb == readLine(self.__file)
        assert self.__host == readLine(self.__file)
        assert str(self.__port) == readLine(self.__file)
        assert self.__splitUrl(url) == self.__splitUrl(readLine(self.__file))
        assert headers == eval(readLine(self.__file))
        expectedInput = readLine(self.__file)
        if isinstance(input, str):
            trInput = input.replace('\n', '').replace('\r', '')
            if input.startswith('{'):
                assert expectedInput.startswith('{'), expectedInput
                assert json.loads(trInput) == json.loads(expectedInput)
            else:
                assert trInput == expectedInput
        else:
            pass

    def __splitUrl(self, url):
        if False:
            while True:
                i = 10
        splitedUrl = url.split('?')
        if len(splitedUrl) == 1:
            return splitedUrl
        assert len(splitedUrl) == 2
        (base, qs) = splitedUrl
        return (base, sorted(qs.split('&')))

    def __request_callback(self, request, uri, response_headers):
        if False:
            for i in range(10):
                print('nop')
        self.__readNextRequest(self.__cnx.verb, self.__cnx.url, self.__cnx.input, self.__cnx.headers)
        status = int(readLine(self.__file))
        self.response_headers = CaseInsensitiveDict(eval(readLine(self.__file)))
        output = bytearray(readLine(self.__file), 'utf-8')
        readLine(self.__file)
        adding_headers = CaseInsensitiveDict(self.response_headers)
        adding_headers.pop('content-length', None)
        adding_headers.pop('transfer-encoding', None)
        adding_headers.pop('content-encoding', None)
        response_headers.update(adding_headers)
        return [status, response_headers, output]

    def getresponse(self):
        if False:
            while True:
                i = 10
        response = self.__cnx.getresponse()
        response.headers = self.response_headers
        return response

    def close(self):
        if False:
            i = 10
            return i + 15
        self.__cnx.close()

class ReplayingHttpConnection(ReplayingConnection):
    _realConnection = github.Requester.HTTPRequestsConnectionClass

    def __init__(self, file, *args, **kwds):
        if False:
            while True:
                i = 10
        super().__init__(file, 'http', *args, **kwds)

class ReplayingHttpsConnection(ReplayingConnection):
    _realConnection = github.Requester.HTTPSRequestsConnectionClass

    def __init__(self, file, *args, **kwds):
        if False:
            return 10
        super().__init__(file, 'https', *args, **kwds)

class BasicTestCase(unittest.TestCase):
    recordMode = False
    tokenAuthMode = False
    jwtAuthMode = False
    per_page = Consts.DEFAULT_PER_PAGE
    retry = None
    pool_size = None
    seconds_between_requests: Optional[float] = None
    seconds_between_writes: Optional[float] = None
    replayDataFolder = os.path.join(os.path.dirname(__file__), 'ReplayData')

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.__fileName = ''
        self.__file = None
        if self.recordMode:
            github.Requester.Requester.injectConnectionClasses(lambda ignored, *args, **kwds: RecordingHttpConnection(self.__openFile('w'), *args, **kwds), lambda ignored, *args, **kwds: RecordingHttpsConnection(self.__openFile('w'), *args, **kwds))
            import GithubCredentials
            self.login = github.Auth.Login(GithubCredentials.login, GithubCredentials.password) if GithubCredentials.login and GithubCredentials.password else None
            self.oauth_token = github.Auth.Token(GithubCredentials.oauth_token) if GithubCredentials.oauth_token else None
            self.jwt = github.Auth.AppAuthToken(GithubCredentials.jwt) if GithubCredentials.jwt else None
            self.app_auth = github.Auth.AppAuth(GithubCredentials.app_id, GithubCredentials.app_private_key) if GithubCredentials.app_id and GithubCredentials.app_private_key else None
        else:
            github.Requester.Requester.injectConnectionClasses(lambda ignored, *args, **kwds: ReplayingHttpConnection(self.__openFile('r'), *args, **kwds), lambda ignored, *args, **kwds: ReplayingHttpsConnection(self.__openFile('r'), *args, **kwds))
            self.login = github.Auth.Login('login', 'password')
            self.oauth_token = github.Auth.Token('oauth_token')
            self.jwt = github.Auth.AppAuthToken('jwt')
            self.app_auth = github.Auth.AppAuth(123456, APP_PRIVATE_KEY)
            httpretty.enable(allow_net_connect=False)

    @property
    def thisTestFailed(self) -> bool:
        if False:
            print('Hello World!')
        if hasattr(self._outcome, 'errors'):
            result = self.defaultTestResult()
            self._feedErrorsToResult(result, self._outcome.errors)
            ok = all((test != self for (test, text) in result.errors + result.failures))
            return not ok
        else:
            return self._outcome.result._excinfo is not None and self._outcome.result._excinfo

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        httpretty.disable()
        httpretty.reset()
        self.__closeReplayFileIfNeeded(silent=self.thisTestFailed)
        github.Requester.Requester.resetConnectionClasses()

    def assertWarning(self, warning, expected):
        if False:
            for i in range(10):
                print('nop')
        self.assertWarnings(warning, expected)

    def assertWarnings(self, warning, *expecteds):
        if False:
            i = 10
            return i + 15
        actual = [(type(message), type(message.message), message.message.args) for message in warning.warnings]
        expected = [(warnings.WarningMessage, DeprecationWarning, (expected,)) for expected in expecteds]
        self.assertSequenceEqual(actual, expected)

    @contextlib.contextmanager
    def ignoreWarning(self, category=Warning, module=''):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=category, module=module)
            yield

    def __openFile(self, mode):
        if False:
            return 10
        for (_, _, functionName, _) in traceback.extract_stack():
            if functionName.startswith('test') or functionName == 'setUp' or functionName == 'tearDown':
                if functionName != 'test':
                    fileName = os.path.join(self.replayDataFolder, f'{self.__class__.__name__}.{functionName}.txt')
        if fileName != self.__fileName:
            self.__closeReplayFileIfNeeded()
            self.__fileName = fileName
            self.__file = open(self.__fileName, mode, encoding='utf-8')
        return self.__file

    def __closeReplayFileIfNeeded(self, silent=False):
        if False:
            return 10
        if self.__file is not None:
            if not self.recordMode and (not silent):
                self.assertEqual(readLine(self.__file), '', self.__fileName)
            self.__file.close()

    def assertListKeyEqual(self, elements, key, expectedKeys):
        if False:
            while True:
                i = 10
        realKeys = [key(element) for element in elements]
        self.assertEqual(realKeys, expectedKeys)

    def assertListKeyBegin(self, elements, key, expectedKeys):
        if False:
            for i in range(10):
                print('nop')
        realKeys = [key(element) for element in elements[:len(expectedKeys)]]
        self.assertEqual(realKeys, expectedKeys)

class TestCase(BasicTestCase):

    def doCheckFrame(self, obj, frame):
        if False:
            print('Hello World!')
        if obj._headers == {} and frame is None:
            return
        if obj._headers is None and frame == {}:
            return
        self.assertEqual(obj._headers, frame[2])

    def getFrameChecker(self):
        if False:
            while True:
                i = 10
        return lambda requester, obj, frame: self.doCheckFrame(obj, frame)

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        github.GithubObject.GithubObject.setCheckAfterInitFlag(True)
        github.Requester.Requester.setDebugFlag(True)
        github.Requester.Requester.setOnCheckMe(self.getFrameChecker())
        self.g = self.get_github(self.retry, self.pool_size)

    def get_github(self, retry, pool_size):
        if False:
            return 10
        if self.tokenAuthMode:
            return github.Github(auth=self.oauth_token, per_page=self.per_page, retry=retry, pool_size=pool_size, seconds_between_requests=self.seconds_between_requests, seconds_between_writes=self.seconds_between_writes)
        elif self.jwtAuthMode:
            return github.Github(auth=self.jwt, per_page=self.per_page, retry=retry, pool_size=pool_size, seconds_between_requests=self.seconds_between_requests, seconds_between_writes=self.seconds_between_writes)
        else:
            return github.Github(auth=self.login, per_page=self.per_page, retry=retry, pool_size=pool_size, seconds_between_requests=self.seconds_between_requests, seconds_between_writes=self.seconds_between_writes)

def activateRecordMode():
    if False:
        for i in range(10):
            print('nop')
    BasicTestCase.recordMode = True

def activateTokenAuthMode():
    if False:
        for i in range(10):
            print('nop')
    BasicTestCase.tokenAuthMode = True

def activateJWTAuthMode():
    if False:
        i = 10
        return i + 15
    BasicTestCase.jwtAuthMode = True

def enableRetry(retry):
    if False:
        return 10
    BasicTestCase.retry = retry

def setPoolSize(pool_size):
    if False:
        for i in range(10):
            print('nop')
    BasicTestCase.pool_size = pool_size