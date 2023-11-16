import webview
from .util import run_test

def test_frameless():
    if False:
        for i in range(10):
            print('nop')
    window = webview.create_window('Frameless test', 'https://www.example.org', frameless=True)
    run_test(webview, window)