import pytest
import webview
from .util import run_test

def test_bg_color():
    if False:
        for i in range(10):
            print('nop')
    window = webview.create_window('Background color test', 'https://www.example.org', background_color='#0000FF')
    run_test(webview, window)

def test_invalid_bg_color():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        webview.create_window('Background color test', 'https://www.example.org', background_color='#dsg0000FF')
    with pytest.raises(ValueError):
        webview.create_window('Background color test', 'https://www.example.org', background_color='FF00FF')
    with pytest.raises(ValueError):
        webview.create_window('Background color test', 'https://www.example.org', background_color='#ac')
    with pytest.raises(ValueError):
        webview.create_window('Background color test', 'https://www.example.org', background_color='#EFEFEH')
    with pytest.raises(ValueError):
        webview.create_window('Background color test', 'https://www.example.org', background_color='#0000000')