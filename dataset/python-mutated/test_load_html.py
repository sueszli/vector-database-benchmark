import webview
from .util import run_test

def test_load_html():
    if False:
        return 10
    window = webview.create_window('Load HTML test')
    run_test(webview, window, load_html)

def load_html(window):
    if False:
        return 10
    window.load_html('<h1>This is dynamically loaded HTML</h1>')