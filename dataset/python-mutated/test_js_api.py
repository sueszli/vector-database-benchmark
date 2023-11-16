from concurrent.futures.thread import ThreadPoolExecutor
import webview
from .util import assert_js, run_test

def test_js_bridge():
    if False:
        for i in range(10):
            print('nop')
    api = Api()
    window = webview.create_window('JSBridge test', js_api=api)
    run_test(webview, window, js_bridge)

def test_exception():
    if False:
        for i in range(10):
            print('nop')
    api = Api()
    window = webview.create_window('JSBridge test', js_api=api)
    run_test(webview, window, exception)

def test_concurrent():
    if False:
        for i in range(10):
            print('nop')
    api = Api()
    window = webview.create_window('JSBridge test', js_api=api)
    run_test(webview, window, concurrent)

class Api:

    class ApiTestException(Exception):
        pass

    def get_int(self):
        if False:
            for i in range(10):
                print('nop')
        return 420

    def get_float(self):
        if False:
            for i in range(10):
                print('nop')
        return 3.141

    def get_string(self):
        if False:
            i = 10
            return i + 15
        return 'test'

    def get_object(self):
        if False:
            while True:
                i = 10
        return {'key1': 'value', 'key2': 420}

    def get_objectlike_string(self):
        if False:
            print('Hello World!')
        return '{"key1": "value", "key2": 420}'

    def get_single_quote(self):
        if False:
            i = 10
            return i + 15
        return "te'st"

    def get_double_quote(self):
        if False:
            i = 10
            return i + 15
        return 'te"st'

    def raise_exception(self):
        if False:
            return 10
        raise Api.ApiTestException()

    def echo(self, param):
        if False:
            i = 10
            return i + 15
        return param

    def multiple(self, param1, param2, param3):
        if False:
            return 10
        return (param1, param2, param3)

def js_bridge(window):
    if False:
        while True:
            i = 10
    window.load_html('<html><body>TEST</body></html>')
    assert_js(window, 'get_int', 420)
    assert_js(window, 'get_float', 3.141)
    assert_js(window, 'get_string', 'test')
    assert_js(window, 'get_object', {'key1': 'value', 'key2': 420})
    assert_js(window, 'get_objectlike_string', '{"key1": "value", "key2": 420}')
    assert_js(window, 'get_single_quote', "te'st")
    assert_js(window, 'get_double_quote', 'te"st')
    assert_js(window, 'echo', 'test', 'test')
    assert_js(window, 'multiple', [1, 2, 3], 1, 2, 3)

def exception(window):
    if False:
        return 10
    assert_js(window, 'raise_exception', 'error')

def concurrent(window):
    if False:
        print('Hello World!')
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(5):
            future = executor.submit(assert_js, window, 'echo', i, i)
            futures.append(future)
    for e in filter(lambda r: r, [f.exception() for f in futures]):
        raise e