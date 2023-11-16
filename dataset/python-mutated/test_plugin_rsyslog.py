import re
from unittest import mock
import pytest
import apprise
import socket
import logging
logging.disable(logging.CRITICAL)
from apprise.plugins.NotifyRSyslog import NotifyRSyslog

@mock.patch('socket.socket')
@mock.patch('os.getpid')
def test_plugin_rsyslog_by_url(mock_getpid, mock_socket):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyRSyslog() Apprise URLs\n\n    '
    payload = 'test'
    mock_connection = mock.Mock()
    mock_getpid.return_value = 123
    mock_connection.sendto.return_value = 16
    mock_socket.return_value = mock_connection
    assert NotifyRSyslog.parse_url(object) is None
    assert NotifyRSyslog.parse_url(42) is None
    assert NotifyRSyslog.parse_url(None) is None
    obj = apprise.Apprise.instantiate('rsyslog://localhost')
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert obj.notify(body=payload) is True
    mock_connection.sendto.return_value = 18
    obj = apprise.Apprise.instantiate('rsyslog://localhost/?facility=local5')
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost/local5') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert obj.notify(body=payload) is True
    assert apprise.Apprise.instantiate('rsyslog://localhost/?facility=invalid') is None
    mock_connection.sendto.return_value = 17
    obj = apprise.Apprise.instantiate('rsyslog://localhost/?facility=d')
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost/daemon') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert obj.notify(body=payload) is True
    mock_connection.sendto.return_value = 0
    assert obj.notify(body=payload) is False
    mock_connection.sendto.return_value = 17
    obj = apprise.Apprise.instantiate('rsyslog://localhost:518')
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost:518') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert obj.notify(body=payload) is True
    mock_connection.sendto.return_value = 39
    assert obj.notify(body=payload, title='Testing a title entry') is True
    mock_connection.sendto.return_value = 16
    obj = apprise.Apprise.instantiate('rsyslog://localhost:514')
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert obj.notify(body=payload) is True
    obj = apprise.Apprise.instantiate('rsyslog://localhost/kern')
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost/kern') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert obj.notify(body=payload) is True
    obj = apprise.Apprise.instantiate('rsyslog://localhost:514/d')
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost/daemon') is True
    assert re.search('logpid=yes', obj.url()) is not None
    mock_connection.sendto.return_value = 17
    assert obj.notify(body=payload) is True
    obj = apprise.Apprise.instantiate('rsyslog://localhost:9000/d?logpid=no')
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost:9000/daemon') is True
    assert re.search('logpid=no', obj.url()) is not None
    mock_connection.sendto.return_value = len(payload) + 5 + len(str(mock_getpid.return_value))
    assert obj.notify(body=payload) is True
    assert obj.notify(body='a different payload size') is False
    mock_connection.sendto.return_value = None
    mock_connection.sendto.side_effect = socket.gaierror
    assert obj.notify(body=payload) is False
    mock_connection.sendto.side_effect = socket.timeout
    assert obj.notify(body=payload) is False

def test_plugin_rsyslog_edge_cases():
    if False:
        return 10
    '\n    NotifyRSyslog() Edge Cases\n\n    '
    obj = NotifyRSyslog(host='localhost', facility=None)
    assert isinstance(obj, NotifyRSyslog)
    assert obj.url().startswith('rsyslog://localhost/user') is True
    assert re.search('logpid=yes', obj.url()) is not None
    with pytest.raises(TypeError):
        NotifyRSyslog(host='localhost', facility='invalid')
    with pytest.raises(TypeError):
        NotifyRSyslog(host='localhost', facility=object)