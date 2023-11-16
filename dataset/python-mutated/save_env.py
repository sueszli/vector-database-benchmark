import builtins
import locale
import os
import sys
import threading
from test import support
from test.support import os_helper
from test.libregrtest.utils import print_warning

class SkipTestEnvironment(Exception):
    pass

class saved_test_environment:
    """Save bits of the test environment and restore them at block exit.

        with saved_test_environment(testname, verbose, quiet):
            #stuff

    Unless quiet is True, a warning is printed to stderr if any of
    the saved items was changed by the test. The support.environment_altered
    attribute is set to True if a change is detected.

    If verbose is more than 1, the before and after state of changed
    items is also printed.
    """

    def __init__(self, testname, verbose=0, quiet=False, *, pgo=False):
        if False:
            for i in range(10):
                print('nop')
        self.testname = testname
        self.verbose = verbose
        self.quiet = quiet
        self.pgo = pgo
    resources = ('sys.argv', 'cwd', 'sys.stdin', 'sys.stdout', 'sys.stderr', 'os.environ', 'sys.path', 'sys.path_hooks', '__import__', 'warnings.filters', 'asyncore.socket_map', 'logging._handlers', 'logging._handlerList', 'sys.gettrace', 'sys.warnoptions', 'multiprocessing.process._dangling', 'threading._dangling', 'sysconfig._CONFIG_VARS', 'sysconfig._INSTALL_SCHEMES', 'files', 'locale', 'warnings.showwarning', 'shutil_archive_formats', 'shutil_unpack_formats', 'asyncio.events._event_loop_policy', 'urllib.requests._url_tempfiles', 'urllib.requests._opener')

    def get_module(self, name):
        if False:
            return 10
        return sys.modules[name]

    def try_get_module(self, name):
        if False:
            print('Hello World!')
        try:
            return self.get_module(name)
        except KeyError:
            raise SkipTestEnvironment

    def get_urllib_requests__url_tempfiles(self):
        if False:
            i = 10
            return i + 15
        urllib_request = self.try_get_module('urllib.request')
        return list(urllib_request._url_tempfiles)

    def restore_urllib_requests__url_tempfiles(self, tempfiles):
        if False:
            return 10
        for filename in tempfiles:
            os_helper.unlink(filename)

    def get_urllib_requests__opener(self):
        if False:
            while True:
                i = 10
        urllib_request = self.try_get_module('urllib.request')
        return urllib_request._opener

    def restore_urllib_requests__opener(self, opener):
        if False:
            while True:
                i = 10
        urllib_request = self.get_module('urllib.request')
        urllib_request._opener = opener

    def get_asyncio_events__event_loop_policy(self):
        if False:
            i = 10
            return i + 15
        self.try_get_module('asyncio')
        return support.maybe_get_event_loop_policy()

    def restore_asyncio_events__event_loop_policy(self, policy):
        if False:
            for i in range(10):
                print('nop')
        asyncio = self.get_module('asyncio')
        asyncio.set_event_loop_policy(policy)

    def get_sys_argv(self):
        if False:
            for i in range(10):
                print('nop')
        return (id(sys.argv), sys.argv, sys.argv[:])

    def restore_sys_argv(self, saved_argv):
        if False:
            i = 10
            return i + 15
        sys.argv = saved_argv[1]
        sys.argv[:] = saved_argv[2]

    def get_cwd(self):
        if False:
            while True:
                i = 10
        return os.getcwd()

    def restore_cwd(self, saved_cwd):
        if False:
            i = 10
            return i + 15
        os.chdir(saved_cwd)

    def get_sys_stdout(self):
        if False:
            while True:
                i = 10
        return sys.stdout

    def restore_sys_stdout(self, saved_stdout):
        if False:
            i = 10
            return i + 15
        sys.stdout = saved_stdout

    def get_sys_stderr(self):
        if False:
            while True:
                i = 10
        return sys.stderr

    def restore_sys_stderr(self, saved_stderr):
        if False:
            print('Hello World!')
        sys.stderr = saved_stderr

    def get_sys_stdin(self):
        if False:
            return 10
        return sys.stdin

    def restore_sys_stdin(self, saved_stdin):
        if False:
            while True:
                i = 10
        sys.stdin = saved_stdin

    def get_os_environ(self):
        if False:
            while True:
                i = 10
        return (id(os.environ), os.environ, dict(os.environ))

    def restore_os_environ(self, saved_environ):
        if False:
            for i in range(10):
                print('nop')
        os.environ = saved_environ[1]
        os.environ.clear()
        os.environ.update(saved_environ[2])

    def get_sys_path(self):
        if False:
            while True:
                i = 10
        return (id(sys.path), sys.path, sys.path[:])

    def restore_sys_path(self, saved_path):
        if False:
            return 10
        sys.path = saved_path[1]
        sys.path[:] = saved_path[2]

    def get_sys_path_hooks(self):
        if False:
            i = 10
            return i + 15
        return (id(sys.path_hooks), sys.path_hooks, sys.path_hooks[:])

    def restore_sys_path_hooks(self, saved_hooks):
        if False:
            print('Hello World!')
        sys.path_hooks = saved_hooks[1]
        sys.path_hooks[:] = saved_hooks[2]

    def get_sys_gettrace(self):
        if False:
            for i in range(10):
                print('nop')
        return sys.gettrace()

    def restore_sys_gettrace(self, trace_fxn):
        if False:
            print('Hello World!')
        sys.settrace(trace_fxn)

    def get___import__(self):
        if False:
            while True:
                i = 10
        return builtins.__import__

    def restore___import__(self, import_):
        if False:
            return 10
        builtins.__import__ = import_

    def get_warnings_filters(self):
        if False:
            return 10
        warnings = self.try_get_module('warnings')
        return (id(warnings.filters), warnings.filters, warnings.filters[:])

    def restore_warnings_filters(self, saved_filters):
        if False:
            return 10
        warnings = self.get_module('warnings')
        warnings.filters = saved_filters[1]
        warnings.filters[:] = saved_filters[2]

    def get_asyncore_socket_map(self):
        if False:
            i = 10
            return i + 15
        asyncore = sys.modules.get('asyncore')
        return asyncore and asyncore.socket_map.copy() or {}

    def restore_asyncore_socket_map(self, saved_map):
        if False:
            return 10
        asyncore = sys.modules.get('asyncore')
        if asyncore is not None:
            asyncore.close_all(ignore_all=True)
            asyncore.socket_map.update(saved_map)

    def get_shutil_archive_formats(self):
        if False:
            print('Hello World!')
        shutil = self.try_get_module('shutil')
        return (shutil._ARCHIVE_FORMATS, shutil._ARCHIVE_FORMATS.copy())

    def restore_shutil_archive_formats(self, saved):
        if False:
            for i in range(10):
                print('nop')
        shutil = self.get_module('shutil')
        shutil._ARCHIVE_FORMATS = saved[0]
        shutil._ARCHIVE_FORMATS.clear()
        shutil._ARCHIVE_FORMATS.update(saved[1])

    def get_shutil_unpack_formats(self):
        if False:
            return 10
        shutil = self.try_get_module('shutil')
        return (shutil._UNPACK_FORMATS, shutil._UNPACK_FORMATS.copy())

    def restore_shutil_unpack_formats(self, saved):
        if False:
            for i in range(10):
                print('nop')
        shutil = self.get_module('shutil')
        shutil._UNPACK_FORMATS = saved[0]
        shutil._UNPACK_FORMATS.clear()
        shutil._UNPACK_FORMATS.update(saved[1])

    def get_logging__handlers(self):
        if False:
            return 10
        logging = self.try_get_module('logging')
        return (id(logging._handlers), logging._handlers, logging._handlers.copy())

    def restore_logging__handlers(self, saved_handlers):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_logging__handlerList(self):
        if False:
            i = 10
            return i + 15
        logging = self.try_get_module('logging')
        return (id(logging._handlerList), logging._handlerList, logging._handlerList[:])

    def restore_logging__handlerList(self, saved_handlerList):
        if False:
            i = 10
            return i + 15
        pass

    def get_sys_warnoptions(self):
        if False:
            print('Hello World!')
        return (id(sys.warnoptions), sys.warnoptions, sys.warnoptions[:])

    def restore_sys_warnoptions(self, saved_options):
        if False:
            i = 10
            return i + 15
        sys.warnoptions = saved_options[1]
        sys.warnoptions[:] = saved_options[2]

    def get_threading__dangling(self):
        if False:
            print('Hello World!')
        return threading._dangling.copy()

    def restore_threading__dangling(self, saved):
        if False:
            i = 10
            return i + 15
        threading._dangling.clear()
        threading._dangling.update(saved)

    def get_multiprocessing_process__dangling(self):
        if False:
            i = 10
            return i + 15
        multiprocessing_process = self.try_get_module('multiprocessing.process')
        multiprocessing_process._cleanup()
        return multiprocessing_process._dangling.copy()

    def restore_multiprocessing_process__dangling(self, saved):
        if False:
            for i in range(10):
                print('nop')
        multiprocessing_process = self.get_module('multiprocessing.process')
        multiprocessing_process._dangling.clear()
        multiprocessing_process._dangling.update(saved)

    def get_sysconfig__CONFIG_VARS(self):
        if False:
            i = 10
            return i + 15
        sysconfig = self.try_get_module('sysconfig')
        sysconfig.get_config_var('prefix')
        return (id(sysconfig._CONFIG_VARS), sysconfig._CONFIG_VARS, dict(sysconfig._CONFIG_VARS))

    def restore_sysconfig__CONFIG_VARS(self, saved):
        if False:
            return 10
        sysconfig = self.get_module('sysconfig')
        sysconfig._CONFIG_VARS = saved[1]
        sysconfig._CONFIG_VARS.clear()
        sysconfig._CONFIG_VARS.update(saved[2])

    def get_sysconfig__INSTALL_SCHEMES(self):
        if False:
            return 10
        sysconfig = self.try_get_module('sysconfig')
        return (id(sysconfig._INSTALL_SCHEMES), sysconfig._INSTALL_SCHEMES, sysconfig._INSTALL_SCHEMES.copy())

    def restore_sysconfig__INSTALL_SCHEMES(self, saved):
        if False:
            for i in range(10):
                print('nop')
        sysconfig = self.get_module('sysconfig')
        sysconfig._INSTALL_SCHEMES = saved[1]
        sysconfig._INSTALL_SCHEMES.clear()
        sysconfig._INSTALL_SCHEMES.update(saved[2])

    def get_files(self):
        if False:
            while True:
                i = 10
        return sorted((fn + ('/' if os.path.isdir(fn) else '') for fn in os.listdir()))

    def restore_files(self, saved_value):
        if False:
            for i in range(10):
                print('nop')
        fn = os_helper.TESTFN
        if fn not in saved_value and fn + '/' not in saved_value:
            if os.path.isfile(fn):
                os_helper.unlink(fn)
            elif os.path.isdir(fn):
                os_helper.rmtree(fn)
    _lc = [getattr(locale, lc) for lc in dir(locale) if lc.startswith('LC_')]

    def get_locale(self):
        if False:
            i = 10
            return i + 15
        pairings = []
        for lc in self._lc:
            try:
                pairings.append((lc, locale.setlocale(lc, None)))
            except (TypeError, ValueError):
                continue
        return pairings

    def restore_locale(self, saved):
        if False:
            for i in range(10):
                print('nop')
        for (lc, setting) in saved:
            locale.setlocale(lc, setting)

    def get_warnings_showwarning(self):
        if False:
            return 10
        warnings = self.try_get_module('warnings')
        return warnings.showwarning

    def restore_warnings_showwarning(self, fxn):
        if False:
            while True:
                i = 10
        warnings = self.get_module('warnings')
        warnings.showwarning = fxn

    def resource_info(self):
        if False:
            i = 10
            return i + 15
        for name in self.resources:
            method_suffix = name.replace('.', '_')
            get_name = 'get_' + method_suffix
            restore_name = 'restore_' + method_suffix
            yield (name, getattr(self, get_name), getattr(self, restore_name))

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.saved_values = []
        for (name, get, restore) in self.resource_info():
            try:
                original = get()
            except SkipTestEnvironment:
                continue
            self.saved_values.append((name, get, restore, original))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        saved_values = self.saved_values
        self.saved_values = None
        support.gc_collect()
        for (name, get, restore, original) in saved_values:
            current = get()
            if current != original:
                support.environment_altered = True
                restore(original)
                if not self.quiet and (not self.pgo):
                    print_warning(f'{name} was modified by {self.testname}')
                    print(f'  Before: {original}\n  After:  {current} ', file=sys.stderr, flush=True)
        return False