import pytest
import webview
from .util import run_test
html = '\n  <html>\n    <body>\n      <h1 id="heading">Heading</h1>\n      <div class="content">Content 1</div>\n      <div class="content">Content 2</div>\n    </body>\n  </html>\n'

@pytest.fixture
def window():
    if False:
        while True:
            i = 10
    return webview.create_window('Get elements test', html=html)

def test_single(window):
    if False:
        while True:
            i = 10
    run_test(webview, window, single_test)

def test_multiple(window):
    if False:
        for i in range(10):
            print('nop')
    run_test(webview, window, multiple_test)

def test_none(window):
    if False:
        i = 10
        return i + 15
    run_test(webview, window, none_test)

def single_test(window):
    if False:
        while True:
            i = 10
    try:
        elements = window.get_elements('#heading')
        assert len(elements) == 1
        assert elements[0]['innerHTML'] == 'Heading'
    except NotImplementedError:
        pass

def multiple_test(window):
    if False:
        while True:
            i = 10
    try:
        elements = window.get_elements('.content')
        assert len(elements) == 2
        assert elements[0]['innerHTML'] == 'Content 1'
        assert elements[1]['innerHTML'] == 'Content 2'
    except NotImplementedError:
        pass

def none_test(window):
    if False:
        while True:
            i = 10
    try:
        elements = window.get_elements('.adgdfg')
        assert len(elements) == 0
    except NotImplementedError:
        pass