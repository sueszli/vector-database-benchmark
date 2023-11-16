import sys
from importlib import import_module
from celery.app.defaults import _OLD_DEFAULTS, _OLD_SETTING_KEYS, _TO_NEW_KEY, _TO_OLD_KEY, DEFAULTS, NAMESPACES, SETTING_KEYS

class test_defaults:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self._prev = sys.modules.pop('celery.app.defaults', None)

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        if self._prev:
            sys.modules['celery.app.defaults'] = self._prev

    def test_option_repr(self):
        if False:
            print('Hello World!')
        assert repr(NAMESPACES['broker']['url'])

    def test_any(self):
        if False:
            return 10
        val = object()
        assert self.defaults.Option.typemap['any'](val) is val

    def test_compat_indices(self):
        if False:
            return 10
        assert not any((key.isupper() for key in DEFAULTS))
        assert not any((key.islower() for key in _OLD_DEFAULTS))
        assert not any((key.isupper() for key in _TO_OLD_KEY))
        assert not any((key.islower() for key in _TO_NEW_KEY))
        assert not any((key.isupper() for key in SETTING_KEYS))
        assert not any((key.islower() for key in _OLD_SETTING_KEYS))
        assert not any((value.isupper() for value in _TO_NEW_KEY.values()))
        assert not any((value.islower() for value in _TO_OLD_KEY.values()))
        for key in _TO_NEW_KEY:
            assert key in _OLD_SETTING_KEYS
        for key in _TO_OLD_KEY:
            assert key in SETTING_KEYS

    def test_find(self):
        if False:
            i = 10
            return i + 15
        find = self.defaults.find
        assert find('default_queue')[2].default == 'celery'
        assert find('task_default_exchange')[2] is None

    @property
    def defaults(self):
        if False:
            i = 10
            return i + 15
        return import_module('celery.app.defaults')