import sys
from unittest import mock
import pytest
import apprise
from apprise import NotifyBase
from apprise.plugins.NotifyGrowl import GrowlPriority, NotifyGrowl
try:
    from gntp import errors
    TEST_GROWL_EXCEPTIONS = (errors.NetworkError(0, 'gntp.ParseError() not handled'), errors.AuthError(0, 'gntp.AuthError() not handled'), errors.ParseError(0, 'gntp.ParseError() not handled'), errors.UnsupportedError(0, 'gntp.UnsupportedError() not handled'))
except ImportError:
    pass
import logging
logging.disable(logging.CRITICAL)

@pytest.mark.skipif('gntp' in sys.modules, reason='Requires that gntp NOT be installed')
def test_plugin_growl_gntp_import_error():
    if False:
        i = 10
        return i + 15
    '\n    NotifyGrowl() Import Error\n\n    '
    obj = apprise.Apprise.instantiate('growl://growl.server')
    assert obj is None

@pytest.mark.skipif('gntp' not in sys.modules, reason='Requires gntp')
@mock.patch('gntp.notifier.GrowlNotifier')
def test_plugin_growl_exception_handling(mock_gntp):
    if False:
        print('Hello World!')
    '\n    NotifyGrowl() Exception Handling\n    '
    TEST_GROWL_EXCEPTIONS = (errors.NetworkError(0, 'gntp.ParseError() not handled'), errors.AuthError(0, 'gntp.AuthError() not handled'), errors.ParseError(0, 'gntp.ParseError() not handled'), errors.UnsupportedError(0, 'gntp.UnsupportedError() not handled'))
    mock_notifier = mock.Mock()
    mock_gntp.return_value = mock_notifier
    mock_notifier.notify.return_value = True
    for exception in TEST_GROWL_EXCEPTIONS:
        mock_notifier.register.side_effect = exception
        obj = apprise.Apprise.instantiate('growl://growl.server.hostname', suppress_exceptions=False)
        assert obj is not None
        assert obj.notify(title='test', body='body', notify_type=apprise.NotifyType.INFO) is False
    mock_notifier.register.side_effect = None
    for exception in TEST_GROWL_EXCEPTIONS:
        mock_notifier.notify.side_effect = exception
        obj = apprise.Apprise.instantiate('growl://growl.server.hostname', suppress_exceptions=False)
        assert obj is not None
        assert obj.notify(title='test', body='body', notify_type=apprise.NotifyType.INFO) is False

@pytest.mark.skipif('gntp' not in sys.modules, reason='Requires gntp')
@mock.patch('gntp.notifier.GrowlNotifier')
def test_plugin_growl_general(mock_gntp):
    if False:
        i = 10
        return i + 15
    '\n    NotifyGrowl() General Checks\n\n    '
    urls = (('growl://', {'instance': None}), ('growl://:@/', {'instance': None}), ('growl://pass@growl.server', {'instance': NotifyGrowl}), ('growl://ignored:pass@growl.server', {'instance': NotifyGrowl}), ('growl://growl.server', {'instance': NotifyGrowl, 'include_image': False}), ('growl://growl.server?version=1', {'instance': NotifyGrowl}), ('growl://growl.server?sticky=yes', {'instance': NotifyGrowl}), ('growl://growl.server?sticky=no', {'instance': NotifyGrowl}), ('growl://growl.server?version=1', {'instance': NotifyGrowl, 'growl_response': None}), ('growl://growl.server?version=2', {'include_image': False, 'instance': NotifyGrowl}), ('growl://growl.server?version=2', {'include_image': False, 'instance': NotifyGrowl, 'growl_response': None}), ('growl://pass@growl.server?priority=low', {'instance': NotifyGrowl}), ('growl://pass@growl.server?priority=moderate', {'instance': NotifyGrowl}), ('growl://pass@growl.server?priority=normal', {'instance': NotifyGrowl}), ('growl://pass@growl.server?priority=high', {'instance': NotifyGrowl}), ('growl://pass@growl.server?priority=emergency', {'instance': NotifyGrowl}), ('growl://pass@growl.server?priority=invalid', {'instance': NotifyGrowl}), ('growl://pass@growl.server?priority=', {'instance': NotifyGrowl}), ('growl://growl.server?version=', {'instance': NotifyGrowl}), ('growl://growl.server?version=crap', {'instance': NotifyGrowl}), ('growl://growl.changeport:2000', {'instance': NotifyGrowl}), ('growl://growl.garbageport:garbage', {'instance': NotifyGrowl}), ('growl://growl.colon:', {'instance': NotifyGrowl}))
    for (url, meta) in urls:
        instance = meta.get('instance', None)
        exception = meta.get('exception', None)
        self = meta.get('self', None)
        response = meta.get('response', True)
        growl_response = meta.get('growl_response', True if response else False)
        mock_notifier = mock.Mock()
        mock_gntp.return_value = mock_notifier
        mock_notifier.notify.side_effect = None
        mock_notifier.notify.return_value = growl_response
        try:
            obj = apprise.Apprise.instantiate(url, suppress_exceptions=False)
            assert exception is None
            if obj is None:
                continue
            if instance is None:
                assert False
            assert isinstance(obj, instance) is True
            if isinstance(obj, NotifyBase):
                assert isinstance(obj.url(), str) is True
                assert isinstance(obj.url(privacy=True), str) is True
                obj_cmp = apprise.Apprise.instantiate(obj.url())
                if not isinstance(obj_cmp, NotifyBase):
                    print('TEST FAIL: {} regenerated as {}'.format(url, obj.url()))
                    assert False
            if self:
                for (key, val) in self.items():
                    assert hasattr(key, obj)
                    assert getattr(key, obj) == val
            try:
                assert obj.notify(title='test', body='body', notify_type=apprise.NotifyType.INFO) == response
            except Exception as e:
                assert isinstance(e, response)
        except AssertionError:
            print('%s AssertionError' % url)
            raise
        except Exception as e:
            print('%s / %s' % (url, str(e)))
            assert exception is not None
            assert isinstance(e, exception)

@pytest.mark.skipif('gntp' not in sys.modules, reason='Requires gntp')
@mock.patch('gntp.notifier.GrowlNotifier')
def test_plugin_growl_config_files(mock_gntp):
    if False:
        i = 10
        return i + 15
    '\n    NotifyGrowl() Config File Cases\n    '
    content = '\n    urls:\n      - growl://pass@growl.server:\n          - priority: -2\n            tag: growl_int low\n          - priority: "-2"\n            tag: growl_str_int low\n          - priority: low\n            tag: growl_str low\n\n          # This will take on moderate (default) priority\n          - priority: invalid\n            tag: growl_invalid\n\n      - growl://pass@growl.server:\n          - priority: 2\n            tag: growl_int emerg\n          - priority: "2"\n            tag: growl_str_int emerg\n          - priority: emergency\n            tag: growl_str emerg\n    '
    mock_notifier = mock.Mock()
    mock_gntp.return_value = mock_notifier
    mock_notifier.notify.return_value = True
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 7
    assert len(aobj) == 7
    assert len([x for x in aobj.find(tag='low')]) == 3
    for s in aobj.find(tag='low'):
        assert s.priority == GrowlPriority.LOW
    assert len([x for x in aobj.find(tag='emerg')]) == 3
    for s in aobj.find(tag='emerg'):
        assert s.priority == GrowlPriority.EMERGENCY
    assert len([x for x in aobj.find(tag='growl_str')]) == 2
    assert len([x for x in aobj.find(tag='growl_str_int')]) == 2
    assert len([x for x in aobj.find(tag='growl_int')]) == 2
    assert len([x for x in aobj.find(tag='growl_invalid')]) == 1
    assert next(aobj.find(tag='growl_invalid')).priority == GrowlPriority.NORMAL