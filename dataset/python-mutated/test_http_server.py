import webview
from .util import assert_js, run_test

class Api:

    def test(self):
        if False:
            i = 10
            return i + 15
        return 'JS Api is working too'

def test_start():
    if False:
        for i in range(10):
            print('nop')
    api = Api()
    window = webview.create_window('Relative URL test', 'assets/test.html', js_api=api)
    run_test(webview, window, assert_func, start_args={'http_server': True})

def assert_func(window):
    if False:
        print('Hello World!')
    html_result = window.evaluate_js('document.getElementById("heading").innerText')
    assert html_result == 'Hello there!'
    css_result = window.evaluate_js('window.getComputedStyle(document.body, null).getPropertyValue("background-color")')
    assert css_result == 'rgb(255, 0, 0)'
    js_result = window.evaluate_js('window.testResult')
    assert js_result == 80085
    assert_js(window, 'test', 'JS Api is working too')