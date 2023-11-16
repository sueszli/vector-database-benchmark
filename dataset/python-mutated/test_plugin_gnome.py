import importlib
import logging
import sys
import types
from unittest import mock
from unittest.mock import Mock, call, ANY
import pytest
import apprise
from apprise.plugins.NotifyGnome import GnomeUrgency, NotifyGnome
from helpers import reload_plugin
logging.disable(logging.CRITICAL)

def setup_glib_environment():
    if False:
        return 10
    '\n    Setup a heavily mocked Glib environment.\n    '
    gi_name = 'gi'
    if gi_name in sys.modules:
        del sys.modules[gi_name]
        reload_plugin('NotifyGnome')
    gi = types.ModuleType(gi_name)
    gi.repository = types.ModuleType(gi_name + '.repository')
    gi.module = types.ModuleType(gi_name + '.module')
    mock_pixbuf = mock.Mock()
    mock_notify = mock.Mock()
    gi.repository.GdkPixbuf = types.ModuleType(gi_name + '.repository.GdkPixbuf')
    gi.repository.GdkPixbuf.Pixbuf = mock_pixbuf
    gi.repository.Notify = mock.Mock()
    gi.repository.Notify.init.return_value = True
    gi.repository.Notify.Notification = mock_notify
    gi.require_version = mock.Mock(name=gi_name + '.require_version')
    sys.modules[gi_name] = gi
    sys.modules[gi_name + '.repository'] = gi.repository
    sys.modules[gi_name + '.repository.Notify'] = gi.repository.Notify
    notify_obj = mock.Mock()
    notify_obj.set_urgency.return_value = True
    notify_obj.set_icon_from_pixbuf.return_value = True
    notify_obj.set_image_from_pixbuf.return_value = True
    notify_obj.show.return_value = True
    mock_notify.new.return_value = notify_obj
    mock_pixbuf.new_from_file.return_value = True
    current_module = sys.modules[__name__]
    reload_plugin('NotifyGnome', replace_in=current_module)

@pytest.fixture
def glib_environment():
    if False:
        return 10
    '\n    Fixture to provide a mocked Glib environment to test case functions.\n    '
    setup_glib_environment()

@pytest.fixture
def obj(glib_environment):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fixture to provide a mocked Apprise instance.\n    '
    obj = apprise.Apprise.instantiate('gnome://', suppress_exceptions=False)
    assert obj is not None
    assert isinstance(obj, NotifyGnome) is True
    obj.duration = 0
    assert obj.enabled is True
    return obj

def test_plugin_gnome_general_success(obj):
    if False:
        return 10
    '\n    NotifyGnome() general checks\n    '
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    assert obj.notify(title='', body='body', notify_type=apprise.NotifyType.INFO) is True

def test_plugin_gnome_image_success(glib_environment):
    if False:
        while True:
            i = 10
    '\n    Verify using the `image` query argument works as intended.\n    '
    obj = apprise.Apprise.instantiate('gnome://_/?image=True', suppress_exceptions=False)
    assert isinstance(obj, NotifyGnome) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('gnome://_/?image=False', suppress_exceptions=False)
    assert isinstance(obj, NotifyGnome) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True

def test_plugin_gnome_priority(glib_environment):
    if False:
        while True:
            i = 10
    '\n    Verify correctness of the `priority` query argument.\n    '
    obj = apprise.Apprise.instantiate('gnome://_/?priority=invalid', suppress_exceptions=False)
    assert isinstance(obj, NotifyGnome) is True
    assert obj.urgency == 1
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('gnome://_/?priority=high', suppress_exceptions=False)
    assert isinstance(obj, NotifyGnome) is True
    assert obj.urgency == 2
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('gnome://_/?priority=2', suppress_exceptions=False)
    assert isinstance(obj, NotifyGnome) is True
    assert obj.urgency == 2
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True

def test_plugin_gnome_urgency(glib_environment):
    if False:
        while True:
            i = 10
    '\n    Verify correctness of the `urgency` query argument.\n    '
    obj = apprise.Apprise.instantiate('gnome://_/?urgency=invalid', suppress_exceptions=False)
    assert obj.urgency == 1
    assert isinstance(obj, NotifyGnome) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('gnome://_/?urgency=high', suppress_exceptions=False)
    assert obj.urgency == 2
    assert isinstance(obj, NotifyGnome) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('gnome://_/?urgency=2', suppress_exceptions=False)
    assert isinstance(obj, NotifyGnome) is True
    assert obj.urgency == 2
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True

def test_plugin_gnome_parse_configuration(obj):
    if False:
        i = 10
        return i + 15
    '\n    Verify configuration parsing works correctly.\n    '
    content = '\n    urls:\n      - gnome://:\n          - priority: 0\n            tag: gnome_int low\n          - priority: "0"\n            tag: gnome_str_int low\n          - priority: low\n            tag: gnome_str low\n          - urgency: 0\n            tag: gnome_int low\n          - urgency: "0"\n            tag: gnome_str_int low\n          - urgency: low\n            tag: gnome_str low\n\n          # These will take on normal (default) urgency\n          - priority: invalid\n            tag: gnome_invalid\n          - urgency: invalid\n            tag: gnome_invalid\n\n      - gnome://:\n          - priority: 2\n            tag: gnome_int high\n          - priority: "2"\n            tag: gnome_str_int high\n          - priority: high\n            tag: gnome_str high\n          - urgency: 2\n            tag: gnome_int high\n          - urgency: "2"\n            tag: gnome_str_int high\n          - urgency: high\n            tag: gnome_str high\n    '
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 14
    assert len(aobj) == 14
    assert len([x for x in aobj.find(tag='low')]) == 6
    for s in aobj.find(tag='low'):
        assert s.urgency == GnomeUrgency.LOW
    assert len([x for x in aobj.find(tag='high')]) == 6
    for s in aobj.find(tag='high'):
        assert s.urgency == GnomeUrgency.HIGH
    assert len([x for x in aobj.find(tag='gnome_str')]) == 4
    assert len([x for x in aobj.find(tag='gnome_str_int')]) == 4
    assert len([x for x in aobj.find(tag='gnome_int')]) == 4
    assert len([x for x in aobj.find(tag='gnome_invalid')]) == 2
    for s in aobj.find(tag='gnome_invalid'):
        assert s.urgency == GnomeUrgency.NORMAL

def test_plugin_gnome_missing_icon(mocker, obj):
    if False:
        while True:
            i = 10
    '\n    Verify the notification will be submitted, even if loading the icon fails.\n    '
    gi = importlib.import_module('gi')
    gi.repository.GdkPixbuf.Pixbuf.new_from_file.side_effect = AttributeError('Something failed')
    logger: Mock = mocker.spy(obj, 'logger')
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    assert logger.mock_calls == [call.warning('Could not load notification icon (%s).', ANY), call.debug('Gnome Exception: Something failed'), call.info('Sent Gnome notification.')]

def test_plugin_gnome_disabled_plugin(obj):
    if False:
        while True:
            i = 10
    '\n    Verify notification will not be submitted if plugin is disabled.\n    '
    obj.enabled = False
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is False

def test_plugin_gnome_set_urgency():
    if False:
        print('Hello World!')
    '\n    Test the setting of an urgency, through `priority` keyword argument.\n    '
    NotifyGnome(priority=0)

def test_plugin_gnome_gi_croaks():
    if False:
        i = 10
        return i + 15
    '\n    Verify notification fails when `gi.require_version()` croaks.\n    '
    try:
        gi = importlib.import_module('gi')
    except ModuleNotFoundError:
        raise pytest.skip('`gi` package not installed')
    gi.require_version.side_effect = ValueError('Something failed')
    current_module = sys.modules[__name__]
    reload_plugin('NotifyGnome', replace_in=current_module)
    obj = apprise.Apprise.instantiate('gnome://', suppress_exceptions=False)
    assert obj is None

def test_plugin_gnome_notify_croaks(mocker, obj):
    if False:
        i = 10
        return i + 15
    '\n    Fail gracefully if underlying object croaks for whatever reason.\n    '
    mocker.patch('gi.repository.Notify.Notification.new', side_effect=AttributeError('Something failed'))
    logger: Mock = mocker.spy(obj, 'logger')
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is False
    assert logger.mock_calls == [call.warning('Failed to send Gnome notification.'), call.debug('Gnome Exception: Something failed')]