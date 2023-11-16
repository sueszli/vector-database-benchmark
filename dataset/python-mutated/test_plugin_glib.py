import importlib
import logging
import re
import sys
import types
from unittest.mock import Mock, call, ANY
import pytest
import apprise
from helpers import reload_plugin
logging.disable(logging.CRITICAL)
if 'dbus' not in sys.modules:
    pytest.skip('Skipping dbus-python based tests', allow_module_level=True)
from dbus import DBusException
from apprise.plugins.NotifyDBus import DBusUrgency, NotifyDBus

def setup_glib_environment():
    if False:
        i = 10
        return i + 15
    '\n    Setup a heavily mocked Glib environment.\n    '
    mock_mainloop = Mock()
    gi_name = 'gi'
    if gi_name in sys.modules:
        del sys.modules[gi_name]
        importlib.reload(sys.modules['apprise.plugins.NotifyDBus'])
    gi = types.ModuleType(gi_name)
    gi.repository = types.ModuleType(gi_name + '.repository')
    mock_pixbuf = Mock()
    mock_image = Mock()
    mock_pixbuf.new_from_file.return_value = mock_image
    mock_image.get_width.return_value = 100
    mock_image.get_height.return_value = 100
    mock_image.get_rowstride.return_value = 1
    mock_image.get_has_alpha.return_value = 0
    mock_image.get_bits_per_sample.return_value = 8
    mock_image.get_n_channels.return_value = 1
    mock_image.get_pixels.return_value = ''
    gi.repository.GdkPixbuf = types.ModuleType(gi_name + '.repository.GdkPixbuf')
    gi.repository.GdkPixbuf.Pixbuf = mock_pixbuf
    gi.require_version = Mock(name=gi_name + '.require_version')
    sys.modules[gi_name] = gi
    sys.modules[gi_name + '.repository'] = gi.repository
    mock_mainloop.qt.DBusQtMainLoop.return_value = True
    mock_mainloop.qt.DBusQtMainLoop.side_effect = ImportError
    sys.modules['dbus.mainloop.qt'] = mock_mainloop.qt
    mock_mainloop.qt.DBusQtMainLoop.side_effect = None
    mock_mainloop.glib.NativeMainLoop.return_value = True
    mock_mainloop.glib.NativeMainLoop.side_effect = ImportError()
    sys.modules['dbus.mainloop.glib'] = mock_mainloop.glib
    mock_mainloop.glib.DBusGMainLoop.side_effect = None
    mock_mainloop.glib.NativeMainLoop.side_effect = None
    current_module = sys.modules[__name__]
    reload_plugin('NotifyDBus', replace_in=current_module)

@pytest.fixture
def dbus_environment(mocker):
    if False:
        i = 10
        return i + 15
    '\n    Fixture to provide a mocked Dbus environment to test case functions.\n    '
    interface_mock = mocker.patch('dbus.Interface', spec=True, Notify=Mock())
    mocker.patch('dbus.SessionBus', spec=True, **{'get_object.return_value': interface_mock})

@pytest.fixture
def glib_environment():
    if False:
        i = 10
        return i + 15
    '\n    Fixture to provide a mocked Glib environment to test case functions.\n    '
    setup_glib_environment()

@pytest.fixture
def dbus_glib_environment(dbus_environment, glib_environment):
    if False:
        print('Hello World!')
    '\n    Fixture to provide a mocked Glib/DBus environment to test case functions.\n    '
    pass

def test_plugin_dbus_general_success(mocker, dbus_glib_environment):
    if False:
        i = 10
        return i + 15
    '\n    NotifyDBus() general tests\n\n    Test class loading using different arguments, provided via URL.\n    '
    obj = apprise.Apprise.instantiate('dbus://', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.url().startswith('dbus://_/')
    obj = apprise.Apprise.instantiate('kde://', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.url().startswith('kde://_/')
    obj = apprise.Apprise.instantiate('qt://', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.url().startswith('qt://_/')
    obj = apprise.Apprise.instantiate('glib://', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.url().startswith('glib://_/')
    obj.duration = 0
    assert NotifyDBus(x_axis=0, y_axis=0, **{'schema': 'dbus'}).notify(title='', body='body', notify_type=apprise.NotifyType.INFO) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    assert obj.notify(title='', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?image=True', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.url().startswith('dbus://_/')
    assert re.search('image=yes', obj.url())
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?image=False', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.url().startswith('dbus://_/')
    assert re.search('image=no', obj.url())
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?priority=invalid', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?priority=high', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?priority=2', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?urgency=invalid', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?urgency=high', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?urgency=2', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?urgency=', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('dbus://_/?x=5&y=5', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True

def test_plugin_dbus_general_failure(dbus_glib_environment):
    if False:
        i = 10
        return i + 15
    '\n    Verify a few failure conditions.\n    '
    with pytest.raises(TypeError):
        NotifyDBus(**{'schema': 'invalid'})
    with pytest.raises(TypeError):
        apprise.Apprise.instantiate('dbus://_/?x=invalid&y=invalid', suppress_exceptions=False)

def test_plugin_dbus_parse_configuration(dbus_glib_environment):
    if False:
        print('Hello World!')
    content = '\n    urls:\n      - dbus://:\n          - priority: 0\n            tag: dbus_int low\n          - priority: "0"\n            tag: dbus_str_int low\n          - priority: low\n            tag: dbus_str low\n          - urgency: 0\n            tag: dbus_int low\n          - urgency: "0"\n            tag: dbus_str_int low\n          - urgency: low\n            tag: dbus_str low\n\n          # These will take on normal (default) urgency\n          - priority: invalid\n            tag: dbus_invalid\n          - urgency: invalid\n            tag: dbus_invalid\n\n      - dbus://:\n          - priority: 2\n            tag: dbus_int high\n          - priority: "2"\n            tag: dbus_str_int high\n          - priority: high\n            tag: dbus_str high\n          - urgency: 2\n            tag: dbus_int high\n          - urgency: "2"\n            tag: dbus_str_int high\n          - urgency: high\n            tag: dbus_str high\n    '
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 14
    assert len(aobj) == 14
    assert len([x for x in aobj.find(tag='low')]) == 6
    for s in aobj.find(tag='low'):
        assert s.urgency == DBusUrgency.LOW
    assert len([x for x in aobj.find(tag='high')]) == 6
    for s in aobj.find(tag='high'):
        assert s.urgency == DBusUrgency.HIGH
    assert len([x for x in aobj.find(tag='dbus_str')]) == 4
    assert len([x for x in aobj.find(tag='dbus_str_int')]) == 4
    assert len([x for x in aobj.find(tag='dbus_int')]) == 4
    assert len([x for x in aobj.find(tag='dbus_invalid')]) == 2
    for s in aobj.find(tag='dbus_invalid'):
        assert s.urgency == DBusUrgency.NORMAL

def test_plugin_dbus_missing_icon(mocker, dbus_glib_environment):
    if False:
        return 10
    '\n    Test exception when loading icon; the notification will still be sent.\n    '
    gi = importlib.import_module('gi')
    gi.repository.GdkPixbuf.Pixbuf.new_from_file.side_effect = AttributeError('Something failed')
    obj = apprise.Apprise.instantiate('dbus://', suppress_exceptions=False)
    logger: Mock = mocker.spy(obj, 'logger')
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    assert logger.mock_calls == [call.warning('Could not load notification icon (%s).', ANY), call.debug('DBus Exception: Something failed'), call.info('Sent DBus notification.')]

def test_plugin_dbus_disabled_plugin(dbus_glib_environment):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify notification will not be submitted if plugin is disabled.\n    '
    obj = apprise.Apprise.instantiate('dbus://', suppress_exceptions=False)
    obj.enabled = False
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is False

def test_plugin_dbus_set_urgency():
    if False:
        print('Hello World!')
    '\n    Test the setting of an urgency.\n    '
    NotifyDBus(urgency=0)

def test_plugin_dbus_gi_missing(dbus_glib_environment):
    if False:
        print('Hello World!')
    '\n    Verify notification succeeds even if the `gi` package is not available.\n    '
    gi = importlib.import_module('gi')
    gi.require_version.side_effect = ImportError()
    current_module = sys.modules[__name__]
    reload_plugin('NotifyDBus', replace_in=current_module)
    obj = apprise.Apprise.instantiate('glib://', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    obj.duration = 0
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True

def test_plugin_dbus_gi_require_version_error(dbus_glib_environment):
    if False:
        print('Hello World!')
    '\n    Verify notification succeeds even if `gi.require_version()` croaks.\n    '
    gi = importlib.import_module('gi')
    gi.require_version.side_effect = ValueError('Something failed')
    current_module = sys.modules[__name__]
    reload_plugin('NotifyDBus', replace_in=current_module)
    obj = apprise.Apprise.instantiate('glib://', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    obj.duration = 0
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True

def test_plugin_dbus_module_croaks(mocker, dbus_glib_environment):
    if False:
        print('Hello World!')
    '\n    Verify plugin is not available when `dbus` module is missing.\n    '
    mocker.patch.dict(sys.modules, {'dbus': compile('raise ImportError()', 'dbus', 'exec')})
    current_module = sys.modules[__name__]
    reload_plugin('NotifyDBus', replace_in=current_module)
    obj = apprise.Apprise.instantiate('glib://', suppress_exceptions=False)
    assert obj is None

def test_plugin_dbus_session_croaks(mocker, dbus_glib_environment):
    if False:
        while True:
            i = 10
    '\n    Verify notification fails if DBus croaks.\n    '
    mocker.patch('dbus.SessionBus', side_effect=DBusException('test'))
    setup_glib_environment()
    obj = apprise.Apprise.instantiate('dbus://', suppress_exceptions=False)
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is False

def test_plugin_dbus_interface_notify_croaks(mocker):
    if False:
        return 10
    '\n    Fail gracefully if underlying object croaks for whatever reason.\n    '
    mocker.patch('dbus.SessionBus', spec=True)
    mocker.patch('dbus.Interface', spec=True, Notify=Mock(side_effect=AttributeError('Something failed')))
    setup_glib_environment()
    obj = apprise.Apprise.instantiate('dbus://', suppress_exceptions=False)
    assert isinstance(obj, NotifyDBus) is True
    logger: Mock = mocker.spy(obj, 'logger')
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is False
    assert [call.warning('Failed to send DBus notification.'), call.debug('DBus Exception: Something failed')] in logger.mock_calls