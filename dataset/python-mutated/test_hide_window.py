import webview
from .util import run_test

def test_hide_show_window():
    if False:
        print('Hello World!')
    window = webview.create_window('Hide/show window test', 'https://www.example.org', hidden=True)
    run_test(webview, window, hide_show_window)

def hide_show_window(window):
    if False:
        return 10
    window.show()
    window.hide()