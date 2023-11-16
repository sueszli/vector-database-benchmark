from sentry.event_manager import EventManager
from sentry.event_manager import get_culprit as get_culprit_impl

def get_culprit(data):
    if False:
        while True:
            i = 10
    mgr = EventManager(data)
    mgr.normalize()
    return get_culprit_impl(mgr.get_data())

def test_cocoa_culprit():
    if False:
        i = 10
        return i + 15
    culprit = get_culprit({'platform': 'cocoa', 'exception': {'type': 'Crash', 'stacktrace': {'frames': [{'filename': 'foo/baz.c', 'package': '/foo/bar/baz.dylib', 'lineno': 1, 'in_app': True, 'function': '-[CRLCrashAsyncSafeThread crash]'}]}}})
    assert culprit == '-[CRLCrashAsyncSafeThread crash]'

def test_emoji_culprit():
    if False:
        print('Hello World!')
    culprit = get_culprit({'platform': 'native', 'exception': {'type': 'Crash', 'stacktrace': {'frames': [{'filename': 'foo/baz.c', 'package': '/foo/bar/baz.dylib', 'module': '😭', 'lineno': 1, 'in_app': True, 'function': '😍'}]}}})
    assert culprit == '😍'

def test_cocoa_strict_stacktrace():
    if False:
        while True:
            i = 10
    culprit = get_culprit({'platform': 'native', 'exception': {'type': 'Crash', 'stacktrace': {'frames': [{'filename': 'foo/baz.c', 'package': '/foo/bar/libswiftCore.dylib', 'lineno': 1, 'in_app': False, 'function': 'fooBar'}, {'package': '/foo/bar/MyApp', 'in_app': True, 'function': 'fooBar2'}, {'filename': 'Mycontroller.swift', 'package': '/foo/bar/MyApp', 'in_app': True, 'function': '-[CRLCrashAsyncSafeThread crash]'}]}}})
    assert culprit == '-[CRLCrashAsyncSafeThread crash]'

def test_culprit_for_synthetic_event():
    if False:
        i = 10
        return i + 15
    culprit = get_culprit({'platform': 'javascript', 'exception': {'type': 'Error', 'value': 'I threw up stringly', 'mechanism': {'type': 'string-error', 'synthetic': True}, 'stacktrace': {'frames': [{'filename': 'foo/baz.js', 'package': 'node_modules/blah/foo/bar.js', 'lineno': 42, 'in_app': True, 'function': 'fooBar'}]}}})
    assert culprit == ''

def test_culprit_for_javascript_event():
    if False:
        while True:
            i = 10
    culprit = get_culprit({'platform': 'javascript', 'exception': {'type': 'Error', 'value': 'I fail hard', 'stacktrace': {'frames': [{'filename': 'foo/baz.js', 'package': 'node_modules/blah/foo/bar.js', 'lineno': 42, 'in_app': True, 'function': 'fooBar'}]}}})
    assert culprit == 'fooBar(foo/baz.js)'

def test_culprit_for_python_event():
    if False:
        for i in range(10):
            print('nop')
    culprit = get_culprit({'platform': 'python', 'exception': {'type': 'ZeroDivisionError', 'value': 'integer division or modulo by zero', 'stacktrace': {'frames': [{'filename': 'foo/baz.py', 'module': 'foo.baz', 'package': 'foo/baz.py', 'lineno': 23, 'in_app': True, 'function': 'it_failed'}]}}})
    assert culprit == 'foo.baz in it_failed'