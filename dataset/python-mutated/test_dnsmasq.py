"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>
"""
import os
import textwrap
import pytest
import salt.modules.dnsmasq as dnsmasq
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, mock_open, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {dnsmasq: {}}

def test_version():
    if False:
        while True:
            i = 10
    '\n    test to show installed version of dnsmasq.\n    '
    mock = MagicMock(return_value='A B C')
    with patch.dict(dnsmasq.__salt__, {'cmd.run': mock}):
        assert dnsmasq.version() == 'C'

def test_fullversion():
    if False:
        while True:
            i = 10
    '\n    Test to Show installed version of dnsmasq and compile options.\n    '
    mock = MagicMock(return_value='A B C\nD E F G H I')
    with patch.dict(dnsmasq.__salt__, {'cmd.run': mock}):
        assert dnsmasq.fullversion() == {'version': 'C', 'compile options': ['G', 'H', 'I']}

def test_set_config():
    if False:
        while True:
            i = 10
    '\n    test to show installed version of dnsmasq.\n    '
    mock = MagicMock(return_value={'conf-dir': 'A'})
    with patch.object(dnsmasq, 'get_config', mock):
        mock = MagicMock(return_value=['.', '~', 'bak', '#'])
        with patch.object(os, 'listdir', mock):
            assert dnsmasq.set_config() == {}

def test_set_config_filter_pub_kwargs():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the kwargs returned from running the set_config function\n    do not contain the __pub that may have been passed through in **kwargs.\n    '
    with patch('salt.modules.dnsmasq.get_config', MagicMock(return_value={'conf-dir': 'A'})):
        mock_domain = 'local'
        mock_address = '/some-test-address.local/8.8.4.4'
        with patch.dict(dnsmasq.__salt__, {'file.append': MagicMock()}):
            ret = dnsmasq.set_config(follow=False, domain=mock_domain, address=mock_address, __pub_pid=8184, __pub_jid=20161101194639387946, __pub_tgt='salt-call')
        assert ret == {'domain': mock_domain, 'address': mock_address}

def test_get_config():
    if False:
        print('Hello World!')
    '\n    test to dumps all options from the config file.\n    '
    mock = MagicMock(return_value={'conf-dir': 'A'})
    with patch.object(dnsmasq, 'get_config', mock):
        mock = MagicMock(return_value=['.', '~', 'bak', '#'])
        with patch.object(os, 'listdir', mock):
            assert dnsmasq.get_config() == {'conf-dir': 'A'}

def test_parse_dnsmasq_no_file():
    if False:
        return 10
    "\n    Tests that a CommandExecutionError is when a filename that doesn't exist is\n    passed in.\n    "
    pytest.raises(CommandExecutionError, dnsmasq._parse_dnamasq, 'filename')

def test_parse_dnamasq():
    if False:
        i = 10
        return i + 15
    '\n    test for generic function for parsing dnsmasq files including includes.\n    '
    with patch('os.path.isfile', MagicMock(return_value=True)):
        text_file_data = textwrap.dedent('            line here\n            second line\n            A=B\n            #')
        with patch('salt.utils.files.fopen', mock_open(read_data=text_file_data)):
            assert dnsmasq._parse_dnamasq('filename') == {'A': 'B', 'unparsed': ['line here\n', 'second line\n']}