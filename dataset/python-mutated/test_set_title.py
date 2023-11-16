import webview
from .util import run_test

def test_set_title():
    if False:
        print('Hello World!')
    window = webview.create_window('Set title test', 'https://www.example.org')
    run_test(webview, window, set_title)

def set_title(window):
    if False:
        i = 10
        return i + 15
    window.set_title('New title')