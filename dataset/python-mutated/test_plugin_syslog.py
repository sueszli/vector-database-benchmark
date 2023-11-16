import re
import sys
import pytest
from unittest import mock
import apprise
import logging
logging.disable(logging.CRITICAL)
if 'syslog' not in sys.modules:
    pytest.skip('Skipping syslog based tests', allow_module_level=True)
from apprise.plugins.NotifySyslog import NotifySyslog

@mock.patch('syslog.syslog')
@mock.patch('syslog.openlog')
def test_plugin_syslog_by_url(openlog, syslog):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifySyslog() Apprise URLs\n\n    '
    assert NotifySyslog.parse_url(object) is None
    assert NotifySyslog.parse_url(42) is None
    assert NotifySyslog.parse_url(None) is None
    obj = apprise.Apprise.instantiate('syslog://')
    assert obj.url().startswith('syslog://user') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert re.search('logperror=no', obj.url()) is not None
    assert isinstance(apprise.Apprise.instantiate('syslog://:@/'), NotifySyslog)
    obj = apprise.Apprise.instantiate('syslog://?logpid=no&logperror=yes')
    assert isinstance(obj, NotifySyslog)
    assert obj.url().startswith('syslog://user') is True
    assert re.search('logpid=no', obj.url()) is not None
    assert re.search('logperror=yes', obj.url()) is not None
    assert obj.notify('body') is True
    assert obj.notify(title='title', body='body') is True
    assert obj.notify('body', notify_type='invalid') is False
    obj = apprise.Apprise.instantiate('syslog://_/?facility=local5')
    assert isinstance(obj, NotifySyslog)
    assert obj.url().startswith('syslog://local5') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert re.search('logperror=no', obj.url()) is not None
    assert apprise.Apprise.instantiate('syslog://_/?facility=invalid') is None
    obj = apprise.Apprise.instantiate('syslog://_/?facility=d')
    assert isinstance(obj, NotifySyslog)
    assert obj.url().startswith('syslog://daemon') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert re.search('logperror=no', obj.url()) is not None
    obj = apprise.Apprise.instantiate('syslog://kern?logpid=no&logperror=y')
    assert isinstance(obj, NotifySyslog)
    assert obj.url().startswith('syslog://kern') is True
    assert re.search('logpid=no', obj.url()) is not None
    assert re.search('logperror=yes', obj.url()) is not None
    obj = apprise.Apprise.instantiate('syslog://kern?facility=d')
    assert isinstance(obj, NotifySyslog)
    assert obj.url().startswith('syslog://daemon') is True

@mock.patch('syslog.syslog')
@mock.patch('syslog.openlog')
def test_plugin_syslog_edge_cases(openlog, syslog):
    if False:
        i = 10
        return i + 15
    '\n    NotifySyslog() Edge Cases\n\n    '
    obj = NotifySyslog(facility=None)
    assert isinstance(obj, NotifySyslog)
    assert obj.url().startswith('syslog://user') is True
    assert re.search('logpid=yes', obj.url()) is not None
    assert re.search('logperror=no', obj.url()) is not None
    with pytest.raises(TypeError):
        NotifySyslog(facility='invalid')
    with pytest.raises(TypeError):
        NotifySyslog(facility=object)