import collections
import textwrap
import pytest
import salt.roster.sshconfig as sshconfig
from tests.support.mock import mock_open, patch

@pytest.fixture
def target_abc():
    if False:
        for i in range(10):
            print('nop')
    return collections.OrderedDict([('user', 'user.mcuserface'), ('priv', '~/.ssh/id_rsa_abc'), ('host', 'abc.asdfgfdhgjkl.com')])

@pytest.fixture
def target_abc123():
    if False:
        i = 10
        return i + 15
    return collections.OrderedDict([('user', 'user.mcuserface'), ('priv', '~/.ssh/id_rsa_abc'), ('host', 'abc123.asdfgfdhgjkl.com')])

@pytest.fixture
def target_def():
    if False:
        return 10
    return collections.OrderedDict([('user', 'user.mcuserface'), ('priv', '~/.ssh/id_rsa_def'), ('host', 'def.asdfgfdhgjkl.com')])

@pytest.fixture
def all_(target_abc, target_abc123, target_def):
    if False:
        print('Hello World!')
    return {'abc.asdfgfdhgjkl.com': target_abc, 'abc123.asdfgfdhgjkl.com': target_abc123, 'def.asdfgfdhgjkl.com': target_def}

@pytest.fixture
def abc_glob(target_abc, target_abc123):
    if False:
        while True:
            i = 10
    return {'abc.asdfgfdhgjkl.com': target_abc, 'abc123.asdfgfdhgjkl.com': target_abc123}

@pytest.fixture
def mock_fp():
    if False:
        while True:
            i = 10
    sample_ssh_config = textwrap.dedent('\n    Host *\n        User user.mcuserface\n\n    Host abc*\n        IdentityFile ~/.ssh/id_rsa_abc\n\n    Host def*\n        IdentityFile  ~/.ssh/id_rsa_def\n\n    Host abc.asdfgfdhgjkl.com\n        HostName 123.123.123.123\n\n    Host abc123.asdfgfdhgjkl.com\n        HostName 123.123.123.124\n\n    Host def.asdfgfdhgjkl.com\n        HostName      234.234.234.234\n    ')
    return mock_open(read_data=sample_ssh_config)

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {sshconfig: {}}

def test_all(mock_fp, all_):
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.files.fopen', mock_fp):
        with patch('salt.roster.sshconfig._get_ssh_config_file'):
            targets = sshconfig.targets('*')
    assert targets == all_

def test_abc_glob(mock_fp, abc_glob):
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.files.fopen', mock_fp):
        with patch('salt.roster.sshconfig._get_ssh_config_file'):
            targets = sshconfig.targets('abc*')
    assert targets == abc_glob