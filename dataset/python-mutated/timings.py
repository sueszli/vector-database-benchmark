"""Global timing settings for all of pywinauto

This module has one object that should be used for all timing adjustments:

 * timings.Timings

There are a couple of predefined settings:

 * ``timings.Timings.fast()``
 * ``timings.Timings.defaults()``
 * ``timings.Timings.slow()``

The Following are the individual timing settings that can be adjusted:

* window_find_timeout (default 5)
* window_find_retry (default .09)

* app_start_timeout (default 10)
* app_start_retry   (default .90)

* app_connect_timeout (default 5.)
* app_connect_retry (default .1)

* cpu_usage_interval (default .5)
* cpu_usage_wait_timeout (default 20)

* exists_timeout  (default .5)
* exists_retry   (default .3)

* after_invoke_wait (default .1)

* after_click_wait  (default .09)
* after_clickinput_wait (default .09)

* after_menu_wait   (default .1)

* after_sendkeys_key_wait   (default .01)

* after_button_click_wait   (default 0)

* before_closeclick_wait    (default .1)
* closeclick_retry  (default .05)
* closeclick_dialog_close_wait  (default 2)
* after_closeclick_wait (default .2)

* after_windowclose_timeout (default 2)
* after_windowclose_retry (default .5)

* after_setfocus_wait   (default .06)
* setfocus_timeout   (default 2)
* setfocus_retry   (default .1)

* after_setcursorpos_wait   (default .01)

* sendmessagetimeout_timeout   (default .01)

* after_tabselect_wait   (default .05)

* after_listviewselect_wait   (default .01)
* after_listviewcheck_wait  default(.001)
* listviewitemcontrol_timeout default(1.5)

* after_treeviewselect_wait  default(.1)

* after_toobarpressbutton_wait  default(.01)

* after_updownchange_wait  default(.1)

* after_movewindow_wait  default(0)
* after_buttoncheck_wait  default(0)
* after_comboboxselect_wait  default(.001)
* after_listboxselect_wait  default(0)
* after_listboxfocuschange_wait  default(0)
* after_editsetedittext_wait  default(0)
* after_editselect_wait  default(.02)

* drag_n_drop_move_mouse_wait  default(.1)
* before_drag_wait  default(.2)
* before_drop_wait  default(.1)
* after_drag_n_drop_wait  default(.1)
* scroll_step_wait  default(.1)

"""
import six
import time
import operator
from functools import wraps
from . import deprecated

class TimeConfig(object):
    """Central storage and manipulation of timing values"""
    __default_timing = {'window_find_timeout': 5.0, 'window_find_retry': 0.09, 'app_start_timeout': 10.0, 'app_start_retry': 0.9, 'app_connect_timeout': 5.0, 'app_connect_retry': 0.1, 'cpu_usage_interval': 0.5, 'cpu_usage_wait_timeout': 20.0, 'exists_timeout': 0.5, 'exists_retry': 0.3, 'after_invoke_wait': 0.1, 'after_click_wait': 0.09, 'after_clickinput_wait': 0.09, 'after_menu_wait': 0.1, 'after_sendkeys_key_wait': 0.01, 'after_button_click_wait': 0, 'before_closeclick_wait': 0.1, 'closeclick_retry': 0.05, 'closeclick_dialog_close_wait': 2.0, 'after_closeclick_wait': 0.2, 'after_windowclose_timeout': 2, 'after_windowclose_retry': 0.5, 'after_setfocus_wait': 0.06, 'setfocus_timeout': 2, 'setfocus_retry': 0.1, 'after_setcursorpos_wait': 0.01, 'sendmessagetimeout_timeout': 0.01, 'after_tabselect_wait': 0.05, 'after_listviewselect_wait': 0.01, 'after_listviewcheck_wait': 0.001, 'listviewitemcontrol_timeout': 1.5, 'after_treeviewselect_wait': 0.1, 'after_toobarpressbutton_wait': 0.01, 'after_updownchange_wait': 0.1, 'after_movewindow_wait': 0, 'after_buttoncheck_wait': 0, 'after_comboboxselect_wait': 0.001, 'after_listboxselect_wait': 0, 'after_listboxfocuschange_wait': 0, 'after_editsetedittext_wait': 0, 'after_editselect_wait': 0.02, 'drag_n_drop_move_mouse_wait': 0.1, 'before_drag_wait': 0.2, 'before_drop_wait': 0.1, 'after_drag_n_drop_wait': 0.1, 'scroll_step_wait': 0.1, 'app_exit_timeout': 10.0, 'app_exit_retry': 0.1}
    assert __default_timing['window_find_timeout'] >= __default_timing['window_find_retry'] * 2
    _timings = __default_timing.copy()
    _cur_speed = 1

    def __getattribute__(self, attr):
        if False:
            print('Hello World!')
        'Get the value for a particular timing'
        if attr in ['__dict__', '__members__', '__methods__', '__class__']:
            return object.__getattribute__(self, attr)
        if attr in dir(TimeConfig):
            return object.__getattribute__(self, attr)
        if attr in self.__default_timing:
            return self._timings[attr]
        else:
            raise AttributeError('Unknown timing setting: {0}'.format(attr))

    def __setattr__(self, attr, value):
        if False:
            print('Hello World!')
        'Set a particular timing'
        if attr == '_timings':
            object.__setattr__(self, attr, value)
        elif attr in self.__default_timing:
            self._timings[attr] = value
        else:
            raise AttributeError('Unknown timing setting: {0}'.format(attr))

    def fast(self):
        if False:
            print('Hello World!')
        'Set fast timing values\n\n        Currently this changes the timing in the following ways:\n        timeouts = 1 second\n        waits = 0 seconds\n        retries = .001 seconds (minimum!)\n\n        (if existing times are faster then keep existing times)\n        '
        for setting in self.__default_timing:
            if '_timeout' in setting:
                self._timings[setting] = min(1, self._timings[setting])
            if '_wait' in setting:
                self._timings[setting] = self._timings[setting] / 2
            elif setting.endswith('_retry'):
                self._timings[setting] = 0.001

    def slow(self):
        if False:
            print('Hello World!')
        'Set slow timing values\n\n        Currently this changes the timing in the following ways:\n        timeouts = default timeouts * 10\n        waits = default waits * 3\n        retries = default retries * 3\n\n        (if existing times are slower then keep existing times)\n        '
        for setting in self.__default_timing:
            if '_timeout' in setting:
                self._timings[setting] = max(self.__default_timing[setting] * 10, self._timings[setting])
            if '_wait' in setting:
                self._timings[setting] = max(self.__default_timing[setting] * 3, self._timings[setting])
            elif setting.endswith('_retry'):
                self._timings[setting] = max(self.__default_timing[setting] * 3, self._timings[setting])
            if self._timings[setting] < 0.2:
                self._timings[setting] = 0.2

    def defaults(self):
        if False:
            while True:
                i = 10
        'Set all timings to the default time'
        self._timings = self.__default_timing.copy()
    Fast = deprecated(fast)
    Slow = deprecated(slow)
    Defaults = deprecated(defaults)
Timings = TimeConfig()

class TimeoutError(RuntimeError):
    pass
if six.PY3:
    _clock_func = time.perf_counter
else:
    _clock_func = time.clock

def timestamp():
    if False:
        while True:
            i = 10
    'Get a precise timestamp'
    return _clock_func()

def always_wait_until(timeout, retry_interval, value=True, op=operator.eq):
    if False:
        while True:
            i = 10
    'Decorator to call wait_until(...) every time for a decorated function/method'

    def wait_until_decorator(func):
        if False:
            return 10
        'Callable object that must be returned by the @always_wait_until decorator'

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            'pre-callback, target function call and post-callback'
            return wait_until(timeout, retry_interval, func, value, op, *args, **kwargs)
        return wrapper
    return wait_until_decorator

def wait_until(timeout, retry_interval, func, value=True, op=operator.eq, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Wait until ``op(function(*args, **kwargs), value)`` is True or until timeout expires\n\n    * **timeout**  how long the function will try the function\n    * **retry_interval**  how long to wait between retries\n    * **func** the function that will be executed\n    * **value**  the value to be compared against (defaults to True)\n    * **op** the comparison function (defaults to equality)\\\n    * **args** optional arguments to be passed to func when called\n    * **kwargs** optional keyword arguments to be passed to func when called\n\n    Returns the return value of the function\n    If the operation times out then the return value of the the function\n    is in the \'function_value\' attribute of the raised exception.\n\n    e.g. ::\n\n        try:\n            # wait a maximum of 10.5 seconds for the\n            # the objects item_count() method to return 10\n            # in increments of .5 of a second\n            wait_until(10.5, .5, self.item_count, 10)\n        except TimeoutError as e:\n            print("timed out")\n    '
    start = timestamp()
    func_val = func(*args, **kwargs)
    while not op(func_val, value):
        time_left = timeout - (timestamp() - start)
        if time_left > 0:
            time.sleep(min(retry_interval, time_left))
            func_val = func(*args, **kwargs)
        else:
            err = TimeoutError('timed out')
            err.function_value = func_val
            raise err
    return func_val
WaitUntil = deprecated(wait_until)

def always_wait_until_passes(timeout, retry_interval, exceptions=Exception):
    if False:
        print('Hello World!')
    'Decorator to call wait_until_passes(...) every time for a decorated function/method'

    def wait_until_passes_decorator(func):
        if False:
            while True:
                i = 10
        'Callable object that must be returned by the @always_wait_until_passes decorator'

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            'pre-callback, target function call and post-callback'
            return wait_until_passes(timeout, retry_interval, func, exceptions, *args, **kwargs)
        return wrapper
    return wait_until_passes_decorator

def wait_until_passes(timeout, retry_interval, func, exceptions=Exception, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Wait until ``func(*args, **kwargs)`` does not raise one of the exceptions\n\n    * **timeout**  how long the function will try the function\n    * **retry_interval**  how long to wait between retries\n    * **func** the function that will be executed\n    * **exceptions**  list of exceptions to test against (default: Exception)\n    * **args** optional arguments to be passed to func when called\n    * **kwargs** optional keyword arguments to be passed to func when called\n\n    Returns the return value of the function\n    If the operation times out then the original exception raised is in\n    the \'original_exception\' attribute of the raised exception.\n\n    e.g. ::\n\n        try:\n            # wait a maximum of 10.5 seconds for the\n            # window to be found in increments of .5 of a second.\n            # P.int a message and re-raise the original exception if never found.\n            wait_until_passes(10.5, .5, self.Exists, (ElementNotFoundError))\n        except TimeoutError as e:\n            print("timed out")\n            raise e.\n    '
    start = timestamp()
    while True:
        try:
            func_val = func(*args, **kwargs)
            break
        except exceptions as e:
            time_left = timeout - (timestamp() - start)
            if time_left > 0:
                time.sleep(min(retry_interval, time_left))
            else:
                err = TimeoutError()
                err.original_exception = e
                raise err
    return func_val
WaitUntilPasses = deprecated(wait_until_passes)