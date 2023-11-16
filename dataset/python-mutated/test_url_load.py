import webview
from .util import run_test

def test_url_load():
    if False:
        for i in range(10):
            print('nop')
    window = webview.create_window('URL change test', 'https://www.example.org')
    run_test(webview, window, url_load)

def url_load(window):
    if False:
        print('Hello World!')
    window.load_url('https://pywebview.flowrl.com')