from __future__ import unicode_literals, division, absolute_import, print_function
import os
from threading import Lock
from copy import deepcopy
from time import sleep
from functools import wraps
from powerline.renderer import Renderer
from powerline.lib.config import ConfigLoader
from powerline import Powerline, get_default_theme
from tests.modules.lib import Args, replace_attr
UT = get_default_theme(is_unicode=True)
AT = get_default_theme(is_unicode=False)

class TestHelpers(object):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        self.config = config
        self.access_log = []
        self.access_lock = Lock()

    def loader_condition(self, path):
        if False:
            i = 10
            return i + 15
        return path in self.config and path

    def find_config_files(self, cfg_path, config_loader, loader_callback):
        if False:
            print('Hello World!')
        if cfg_path.endswith('.json'):
            cfg_path = cfg_path[:-5]
        if cfg_path.startswith('/'):
            cfg_path = cfg_path.lstrip('/')
        with self.access_lock:
            self.access_log.append('check:' + cfg_path)
        if cfg_path in self.config:
            yield cfg_path
        else:
            if config_loader:
                config_loader.register_missing(self.loader_condition, loader_callback, cfg_path)
            raise IOError(('fcf:' if cfg_path.endswith('raise') else '') + cfg_path)

    def load_json_config(self, config_file_path, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if config_file_path.endswith('.json'):
            config_file_path = config_file_path[:-5]
        if config_file_path.startswith('/'):
            config_file_path = config_file_path.lstrip('/')
        with self.access_lock:
            self.access_log.append('load:' + config_file_path)
        try:
            return deepcopy(self.config[config_file_path])
        except KeyError:
            raise IOError(config_file_path)

    def pop_events(self):
        if False:
            for i in range(10):
                print('nop')
        with self.access_lock:
            r = self.access_log[:]
            self.access_log = []
        return r

def log_call(func):
    if False:
        for i in range(10):
            print('nop')

    @wraps(func)
    def ret(self, *args, **kwargs):
        if False:
            return 10
        self._calls.append((func.__name__, args, kwargs))
        return func(self, *args, **kwargs)
    return ret

class TestWatcher(object):
    events = set()
    lock = Lock()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._calls = []

    @log_call
    def watch(self, file):
        if False:
            print('Hello World!')
        pass

    @log_call
    def __call__(self, file):
        if False:
            print('Hello World!')
        with self.lock:
            if file in self.events:
                self.events.remove(file)
                return True
        return False

    def _reset(self, files):
        if False:
            print('Hello World!')
        with self.lock:
            self.events.clear()
            self.events.update(files)

    @log_call
    def unsubscribe(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class Logger(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.messages = []
        self.lock = Lock()

    def _add_msg(self, attr, msg):
        if False:
            print('Hello World!')
        with self.lock:
            self.messages.append(attr + ':' + msg)

    def _pop_msgs(self):
        if False:
            while True:
                i = 10
        with self.lock:
            r = self.messages
            self.messages = []
        return r

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        return lambda *args, **kwargs: self._add_msg(attr, *args, **kwargs)

class SimpleRenderer(Renderer):

    def hlstyle(self, fg=None, bg=None, attrs=None):
        if False:
            return 10
        return '<{fg} {bg} {attrs}>'.format(fg=fg and fg[0], bg=bg and bg[0], attrs=attrs)

class EvenSimplerRenderer(Renderer):

    def hlstyle(self, fg=None, bg=None, attrs=None):
        if False:
            for i in range(10):
                print('nop')
        return '{{{fg}{bg}{attrs}}}'.format(fg=fg and fg[0] or '-', bg=bg and bg[0] or '-', attrs=attrs if attrs else '')

class TestPowerline(Powerline):
    _created = False

    def __init__(self, _helpers, **kwargs):
        if False:
            i = 10
            return i + 15
        super(TestPowerline, self).__init__(**kwargs)
        self._helpers = _helpers
        self.find_config_files = _helpers.find_config_files

    @staticmethod
    def get_local_themes(local_themes):
        if False:
            print('Hello World!')
        return local_themes

    @staticmethod
    def get_config_paths():
        if False:
            while True:
                i = 10
        return ['']

    def _will_create_renderer(self):
        if False:
            return 10
        return self.cr_kwargs

    def _pop_events(self):
        if False:
            return 10
        return self._helpers.pop_events()
renderer = EvenSimplerRenderer

class TestConfigLoader(ConfigLoader):

    def __init__(self, _helpers, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        watcher = TestWatcher()
        super(TestConfigLoader, self).__init__(load=_helpers.load_json_config, watcher=watcher, watcher_type='test', **kwargs)

def get_powerline(config, **kwargs):
    if False:
        while True:
            i = 10
    helpers = TestHelpers(config)
    return get_powerline_raw(helpers, TestPowerline, _helpers=helpers, ext='test', renderer_module='tests.modules.lib.config_mock', logger=Logger(), **kwargs)

def select_renderer(simpler_renderer=False):
    if False:
        print('Hello World!')
    global renderer
    renderer = EvenSimplerRenderer if simpler_renderer else SimpleRenderer

def get_powerline_raw(helpers, PowerlineClass, replace_gcp=False, **kwargs):
    if False:
        print('Hello World!')
    if not isinstance(helpers, TestHelpers):
        helpers = TestHelpers(helpers)
    select_renderer(kwargs.pop('simpler_renderer', False))
    if replace_gcp:

        class PowerlineClass(PowerlineClass):

            @staticmethod
            def get_config_paths():
                if False:
                    for i in range(10):
                        print('nop')
                return ['/']
    pl = PowerlineClass(config_loader=TestConfigLoader(_helpers=helpers, run_once=kwargs.get('run_once')), **kwargs)
    pl._watcher = pl.config_loader.watcher
    return pl

def swap_attributes(config, powerline_module):
    if False:
        print('Hello World!')
    return replace_attr(powerline_module, 'os', Args(path=Args(isfile=lambda path: path.lstrip('/').replace('.json', '') in config, join=os.path.join, expanduser=lambda path: path, realpath=lambda path: path, dirname=os.path.dirname), environ={}))

def add_watcher_events(p, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    if isinstance(p._watcher, TestWatcher):
        p._watcher._reset(args)
    while not p._will_create_renderer():
        sleep(kwargs.get('interval', 0.1))
        if not kwargs.get('wait', True):
            return