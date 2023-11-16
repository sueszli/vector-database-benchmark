import webview
from .util import assert_js, run_test

def test_expose_single():
    if False:
        print('Hello World!')
    window = webview.create_window('JSBridge test', html='<html><body>TEST</body></html>')
    window.expose(get_int)
    run_test(webview, window, expose_single)

def test_expose_multiple():
    if False:
        for i in range(10):
            print('nop')
    window = webview.create_window('JSBridge test', html='<html><body>TEST</body></html>')
    window.expose(get_int, get_float)
    run_test(webview, window, expose_multiple)

def test_expose_runtime():
    if False:
        while True:
            i = 10
    window = webview.create_window('JSBridge test', html='<html><body>TEST</body></html>')
    run_test(webview, window, expose_runtime)

def test_override():
    if False:
        while True:
            i = 10
    api = Api()
    window = webview.create_window('JSBridge test', js_api=api)
    window.expose(get_int)
    run_test(webview, window, expose_override)

def get_int():
    if False:
        i = 10
        return i + 15
    return 420

def get_float():
    if False:
        for i in range(10):
            print('nop')
    return 420.42

class Api:

    def get_int(self):
        if False:
            while True:
                i = 10
        return 421

def expose_single(window):
    if False:
        i = 10
        return i + 15
    assert_js(window, 'get_int', 420)

def expose_multiple(window):
    if False:
        return 10
    assert_js(window, 'get_int', 420)
    assert_js(window, 'get_float', 420.42)

def expose_runtime(window):
    if False:
        for i in range(10):
            print('nop')
    window.expose(get_int, get_float)
    assert_js(window, 'get_int', 420)

def expose_override(window):
    if False:
        print('Hello World!')
    assert_js(window, 'get_int', 420)