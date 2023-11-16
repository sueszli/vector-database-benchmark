import webview
from .util import run_test

def test_simple_browser():
    if False:
        for i in range(10):
            print('nop')
    window = webview.create_window('Simple browser test', 'https://www.example.org')
    run_test(webview, window)