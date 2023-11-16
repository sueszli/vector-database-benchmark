import functools
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pytest
import pip._internal.network.auth
from pip._internal.network.auth import MultiDomainBasicAuth
from tests.lib.requests_mocks import MockConnection, MockRequest, MockResponse

@pytest.fixture(scope='function', autouse=True)
def reset_keyring() -> Iterable[None]:
    if False:
        return 10
    yield None
    pip._internal.network.auth.KEYRING_DISABLED = False
    pip._internal.network.auth.get_keyring_provider.cache_clear()

@pytest.mark.parametrize(['input_url', 'url', 'username', 'password'], [('http://user%40email.com:password@example.com/path', 'http://example.com/path', 'user@email.com', 'password'), ('http://username:password@example.com/path', 'http://example.com/path', 'username', 'password'), ('http://token@example.com/path', 'http://example.com/path', 'token', ''), ('http://example.com/path', 'http://example.com/path', None, None)])
def test_get_credentials_parses_correctly(input_url: str, url: str, username: Optional[str], password: Optional[str]) -> None:
    if False:
        i = 10
        return i + 15
    auth = MultiDomainBasicAuth()
    get = auth._get_url_and_credentials
    assert get(input_url) == (url, username, password)
    assert username is None and password is None or auth.passwords['example.com'] == (username, password)

def test_get_credentials_not_to_uses_cached_credentials() -> None:
    if False:
        while True:
            i = 10
    auth = MultiDomainBasicAuth()
    auth.passwords['example.com'] = ('user', 'pass')
    got = auth._get_url_and_credentials('http://foo:bar@example.com/path')
    expected = ('http://example.com/path', 'foo', 'bar')
    assert got == expected

def test_get_credentials_not_to_uses_cached_credentials_only_username() -> None:
    if False:
        i = 10
        return i + 15
    auth = MultiDomainBasicAuth()
    auth.passwords['example.com'] = ('user', 'pass')
    got = auth._get_url_and_credentials('http://foo@example.com/path')
    expected = ('http://example.com/path', 'foo', '')
    assert got == expected

def test_get_credentials_uses_cached_credentials() -> None:
    if False:
        return 10
    auth = MultiDomainBasicAuth()
    auth.passwords['example.com'] = ('user', 'pass')
    got = auth._get_url_and_credentials('http://example.com/path')
    expected = ('http://example.com/path', 'user', 'pass')
    assert got == expected

def test_get_credentials_uses_cached_credentials_only_username() -> None:
    if False:
        while True:
            i = 10
    auth = MultiDomainBasicAuth()
    auth.passwords['example.com'] = ('user', 'pass')
    got = auth._get_url_and_credentials('http://user@example.com/path')
    expected = ('http://example.com/path', 'user', 'pass')
    assert got == expected

def test_get_index_url_credentials() -> None:
    if False:
        print('Hello World!')
    auth = MultiDomainBasicAuth(index_urls=['http://example.com/', 'http://foo:bar@example.com/path'])
    get = functools.partial(auth._get_new_credentials, allow_netrc=False, allow_keyring=False)
    assert get('http://example.com/path/path2') == ('foo', 'bar')
    assert get('http://example.com/path3/path2') == (None, None)

def test_prioritize_longest_path_prefix_match_organization() -> None:
    if False:
        i = 10
        return i + 15
    auth = MultiDomainBasicAuth(index_urls=['http://foo:bar@example.com/org-name-alpha/repo-alias/simple', 'http://bar:foo@example.com/org-name-beta/repo-alias/simple'])
    get = functools.partial(auth._get_new_credentials, allow_netrc=False, allow_keyring=False)
    assert get('http://example.com/org-name-alpha/repo-guid/dowbload/') == ('foo', 'bar')
    assert get('http://example.com/org-name-beta/repo-guid/dowbload/') == ('bar', 'foo')

def test_prioritize_longest_path_prefix_match_project() -> None:
    if False:
        return 10
    auth = MultiDomainBasicAuth(index_urls=['http://foo:bar@example.com/org-alpha/project-name-alpha/repo-alias/simple', 'http://bar:foo@example.com/org-alpha/project-name-beta/repo-alias/simple'])
    get = functools.partial(auth._get_new_credentials, allow_netrc=False, allow_keyring=False)
    assert get('http://example.com/org-alpha/project-name-alpha/repo-guid/dowbload/') == ('foo', 'bar')
    assert get('http://example.com/org-alpha/project-name-beta/repo-guid/dowbload/') == ('bar', 'foo')

class KeyringModuleV1:
    """Represents the supported API of keyring before get_credential
    was added.
    """

    def __init__(self) -> None:
        if False:
            return 10
        self.saved_passwords: List[Tuple[str, str, str]] = []

    def get_password(self, system: str, username: str) -> Optional[str]:
        if False:
            while True:
                i = 10
        if system == 'example.com' and username:
            return username + '!netloc'
        if system == 'http://example.com/path2/' and username:
            return username + '!url'
        return None

    def set_password(self, system: str, username: str, password: str) -> None:
        if False:
            print('Hello World!')
        self.saved_passwords.append((system, username, password))

@pytest.mark.parametrize('url, expect', (('http://example.com/path1', (None, None)), ('http://user@example.com/path3', ('user', 'user!netloc')), ('http://user2@example.com/path3', ('user2', 'user2!netloc')), ('http://example.com/path2/path3', (None, None)), ('http://foo@example.com/path2/path3', ('foo', 'foo!url'))))
def test_keyring_get_password(monkeypatch: pytest.MonkeyPatch, url: str, expect: Tuple[Optional[str], Optional[str]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    keyring = KeyringModuleV1()
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    auth = MultiDomainBasicAuth(index_urls=['http://example.com/path2', 'http://example.com/path3'], keyring_provider='import')
    actual = auth._get_new_credentials(url, allow_netrc=False, allow_keyring=True)
    assert actual == expect

def test_keyring_get_password_after_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        print('Hello World!')
    keyring = KeyringModuleV1()
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    auth = MultiDomainBasicAuth(keyring_provider='import')

    def ask_input(prompt: str) -> str:
        if False:
            return 10
        assert prompt == 'User for example.com: '
        return 'user'
    monkeypatch.setattr('pip._internal.network.auth.ask_input', ask_input)
    actual = auth._prompt_for_password('example.com')
    assert actual == ('user', 'user!netloc', False)

def test_keyring_get_password_after_prompt_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        return 10
    keyring = KeyringModuleV1()
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    auth = MultiDomainBasicAuth(keyring_provider='import')

    def ask_input(prompt: str) -> str:
        if False:
            return 10
        assert prompt == 'User for unknown.com: '
        return 'user'

    def ask_password(prompt: str) -> str:
        if False:
            while True:
                i = 10
        assert prompt == 'Password: '
        return 'fake_password'
    monkeypatch.setattr('pip._internal.network.auth.ask_input', ask_input)
    monkeypatch.setattr('pip._internal.network.auth.ask_password', ask_password)
    actual = auth._prompt_for_password('unknown.com')
    assert actual == ('user', 'fake_password', True)

def test_keyring_get_password_username_in_index(monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        print('Hello World!')
    keyring = KeyringModuleV1()
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    auth = MultiDomainBasicAuth(index_urls=['http://user@example.com/path2', 'http://example.com/path4'], keyring_provider='import')
    get = functools.partial(auth._get_new_credentials, allow_netrc=False, allow_keyring=True)
    assert get('http://example.com/path2/path3') == ('user', 'user!url')
    assert get('http://example.com/path4/path1') == (None, None)

@pytest.mark.parametrize('response_status, creds, expect_save', ((403, ('user', 'pass', True), False), (200, ('user', 'pass', True), True), (200, ('user', 'pass', False), False)))
def test_keyring_set_password(monkeypatch: pytest.MonkeyPatch, response_status: int, creds: Tuple[str, str, bool], expect_save: bool) -> None:
    if False:
        return 10
    keyring = KeyringModuleV1()
    monkeypatch.setitem(sys.modules, 'keyring', keyring)
    auth = MultiDomainBasicAuth(prompting=True, keyring_provider='import')
    monkeypatch.setattr(auth, '_get_url_and_credentials', lambda u: (u, None, None))
    monkeypatch.setattr(auth, '_prompt_for_password', lambda *a: creds)
    if creds[2]:

        def should_save_password_to_keyring(*a: Any) -> bool:
            if False:
                while True:
                    i = 10
            return True
    else:

        def should_save_password_to_keyring(*a: Any) -> bool:
            if False:
                while True:
                    i = 10
            assert False, '_should_save_password_to_keyring should not be called'
    monkeypatch.setattr(auth, '_should_save_password_to_keyring', should_save_password_to_keyring)
    req = MockRequest('https://example.com')
    resp = MockResponse(b'')
    resp.url = req.url
    connection = MockConnection()

    def _send(sent_req: MockRequest, **kwargs: Any) -> MockResponse:
        if False:
            return 10
        assert sent_req is req
        assert 'Authorization' in sent_req.headers
        r = MockResponse(b'')
        r.status_code = response_status
        return r
    connection._send = _send
    resp.request = req
    resp.status_code = 401
    resp.connection = connection
    auth.handle_401(resp)
    if expect_save:
        assert keyring.saved_passwords == [('example.com', creds[0], creds[1])]
    else:
        assert keyring.saved_passwords == []

class KeyringModuleV2:
    """Represents the current supported API of keyring"""

    class Credential:

        def __init__(self, username: str, password: str) -> None:
            if False:
                return 10
            self.username = username
            self.password = password

    def get_password(self, system: str, username: str) -> None:
        if False:
            return 10
        assert False, 'get_password should not ever be called'

    def get_credential(self, system: str, username: str) -> Optional[Credential]:
        if False:
            i = 10
            return i + 15
        if system == 'http://example.com/path2/':
            return self.Credential('username', 'url')
        if system == 'example.com':
            return self.Credential('username', 'netloc')
        return None

@pytest.mark.parametrize('url, expect', (('http://example.com/path1', ('username', 'netloc')), ('http://example.com/path2/path3', ('username', 'url')), ('http://user2@example.com/path2/path3', ('username', 'url'))))
def test_keyring_get_credential(monkeypatch: pytest.MonkeyPatch, url: str, expect: Tuple[str, str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setitem(sys.modules, 'keyring', KeyringModuleV2())
    auth = MultiDomainBasicAuth(index_urls=['http://example.com/path1', 'http://example.com/path2'], keyring_provider='import')
    assert auth._get_new_credentials(url, allow_netrc=False, allow_keyring=True) == expect

class KeyringModuleBroken:
    """Represents the current supported API of keyring, but broken"""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._call_count = 0

    def get_credential(self, system: str, username: str) -> None:
        if False:
            print('Hello World!')
        self._call_count += 1
        raise Exception('This keyring is broken!')

def test_broken_keyring_disables_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        print('Hello World!')
    keyring_broken = KeyringModuleBroken()
    monkeypatch.setitem(sys.modules, 'keyring', keyring_broken)
    auth = MultiDomainBasicAuth(index_urls=['http://example.com/'], keyring_provider='import')
    assert keyring_broken._call_count == 0
    for i in range(5):
        url = 'http://example.com/path' + str(i)
        assert auth._get_new_credentials(url, allow_netrc=False, allow_keyring=True) == (None, None)
        assert keyring_broken._call_count == 1

class KeyringSubprocessResult(KeyringModuleV1):
    """Represents the subprocess call to keyring"""
    returncode = 0

    def __call__(self, cmd: List[str], *, env: Dict[str, str], stdin: Optional[Any]=None, stdout: Optional[Any]=None, input: Optional[bytes]=None, check: Optional[bool]=None) -> Any:
        if False:
            i = 10
            return i + 15
        if cmd[1] == 'get':
            assert stdin == -3
            assert stdout == subprocess.PIPE
            assert env['PYTHONIOENCODING'] == 'utf-8'
            assert check is None
            password = self.get_password(*cmd[2:])
            if password is None:
                self.returncode = 1
            else:
                self.returncode = 0
                self.stdout = (password + os.linesep).encode('utf-8')
        if cmd[1] == 'set':
            assert stdin is None
            assert stdout is None
            assert env['PYTHONIOENCODING'] == 'utf-8'
            assert input is not None
            assert check
            self.set_password(cmd[2], cmd[3], input.decode('utf-8').strip(os.linesep))
        return self

    def check_returncode(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.returncode:
            raise Exception()

@pytest.mark.parametrize('url, expect', (('http://example.com/path1', (None, None)), ('http://user@example.com/path3', ('user', 'user!netloc')), ('http://user2@example.com/path3', ('user2', 'user2!netloc')), ('http://example.com/path2/path3', (None, None)), ('http://foo@example.com/path2/path3', ('foo', 'foo!url'))))
def test_keyring_cli_get_password(monkeypatch: pytest.MonkeyPatch, url: str, expect: Tuple[Optional[str], Optional[str]]) -> None:
    if False:
        while True:
            i = 10
    monkeypatch.setattr(pip._internal.network.auth.shutil, 'which', lambda x: 'keyring')
    monkeypatch.setattr(pip._internal.network.auth.subprocess, 'run', KeyringSubprocessResult())
    auth = MultiDomainBasicAuth(index_urls=['http://example.com/path2', 'http://example.com/path3'], keyring_provider='subprocess')
    actual = auth._get_new_credentials(url, allow_netrc=False, allow_keyring=True)
    assert actual == expect

@pytest.mark.parametrize('response_status, creds, expect_save', ((403, ('user', 'pass', True), False), (200, ('user', 'pass', True), True), (200, ('user', 'pass', False), False)))
def test_keyring_cli_set_password(monkeypatch: pytest.MonkeyPatch, response_status: int, creds: Tuple[str, str, bool], expect_save: bool) -> None:
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(pip._internal.network.auth.shutil, 'which', lambda x: 'keyring')
    keyring = KeyringSubprocessResult()
    monkeypatch.setattr(pip._internal.network.auth.subprocess, 'run', keyring)
    auth = MultiDomainBasicAuth(prompting=True, keyring_provider='subprocess')
    monkeypatch.setattr(auth, '_get_url_and_credentials', lambda u: (u, None, None))
    monkeypatch.setattr(auth, '_prompt_for_password', lambda *a: creds)
    if creds[2]:

        def should_save_password_to_keyring(*a: Any) -> bool:
            if False:
                while True:
                    i = 10
            return True
    else:

        def should_save_password_to_keyring(*a: Any) -> bool:
            if False:
                while True:
                    i = 10
            assert False, '_should_save_password_to_keyring should not be called'
    monkeypatch.setattr(auth, '_should_save_password_to_keyring', should_save_password_to_keyring)
    req = MockRequest('https://example.com')
    resp = MockResponse(b'')
    resp.url = req.url
    connection = MockConnection()

    def _send(sent_req: MockRequest, **kwargs: Any) -> MockResponse:
        if False:
            while True:
                i = 10
        assert sent_req is req
        assert 'Authorization' in sent_req.headers
        r = MockResponse(b'')
        r.status_code = response_status
        return r
    connection._send = _send
    resp.request = req
    resp.status_code = 401
    resp.connection = connection
    auth.handle_401(resp)
    if expect_save:
        assert keyring.saved_passwords == [('example.com', creds[0], creds[1])]
    else:
        assert keyring.saved_passwords == []