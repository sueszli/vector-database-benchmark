import pytest
import webview
from .util import assert_js, run_test

@pytest.fixture
def window():
    if False:
        print('Hello World!')
    return webview.create_window('Main window', html='<html><body><h1>Master window</h1></body></html>')

def test_bg_color():
    if False:
        while True:
            i = 10
    window = webview.create_window('Main window', background_color='#0000FF')
    run_test(webview, window, bg_color)

def test_load_html(window):
    if False:
        print('Hello World!')
    run_test(webview, window, load_html)

def test_load_url(window):
    if False:
        while True:
            i = 10
    run_test(webview, window, load_url)

def test_evaluate_js(window):
    if False:
        i = 10
        return i + 15
    run_test(webview, window, evaluate_js)

def test_js_bridge():
    if False:
        i = 10
        return i + 15

    class Api1:

        def test1(self):
            if False:
                i = 10
                return i + 15
            return 1
    window = webview.create_window('Multi-window js bridge test', html='<html><body><h1>Master window</h1></body></html>', js_api=Api1())
    run_test(webview, window, js_bridge)

def bg_color(window):
    if False:
        for i in range(10):
            print('nop')
    child_window = webview.create_window('Window #2', background_color='#0000FF')
    assert child_window.uid != 'MainWindow'
    child_window.destroy()

def js_bridge(window):
    if False:
        print('Hello World!')

    class Api2:

        def test2(self):
            if False:
                for i in range(10):
                    print('nop')
            return 2
    api2 = Api2()
    child_window = webview.create_window('Window #2', js_api=api2)
    assert child_window.uid != 'MainWindow'
    child_window.load_html('<html><body><h1>Secondary window</h1></body></html>')
    assert_js(window, 'test1', 1)
    assert_js(child_window, 'test2', 2)
    child_window.destroy()

def evaluate_js(window):
    if False:
        for i in range(10):
            print('nop')
    child_window = webview.create_window('Window #2', 'https://pywebview.flowrl.com')
    assert child_window.uid != 'MainWindow'
    result1 = window.evaluate_js("\n        document.body.style.backgroundColor = '#212121';\n        // comment\n        function test() {\n            return 2 + 5;\n        }\n        test();\n    ")
    assert result1 == 7
    result2 = child_window.evaluate_js("\n        document.body.style.backgroundColor = '#212121';\n        // comment\n        function test() {\n            return 2 + 2;\n        }\n        test();\n    ")
    assert result2 == 4
    child_window.destroy()

def load_html(window):
    if False:
        return 10
    child_window = webview.create_window('Window #2', html='<body style="background: red;"><h1>Master Window</h1></body>')
    assert child_window != 'MainWindow'
    child_window.destroy()

def load_url(window):
    if False:
        i = 10
        return i + 15
    child_window = webview.create_window('Window #2')
    assert child_window != 'MainWindow'
    child_window.load_url('https://woot.fi')
    child_window.destroy()