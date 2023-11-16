import webview
from .util import run_test

def test_toggle_fullscreen():
    if False:
        while True:
            i = 10
    window = webview.create_window('Toggle fullscreen test', 'https://www.example.org')
    run_test(webview, window, toggle_fullscreen)

def toggle_fullscreen(window):
    if False:
        while True:
            i = 10
    window.toggle_fullscreen()