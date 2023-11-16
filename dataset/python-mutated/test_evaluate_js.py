import pytest
import webview
from .util import run_test

@pytest.fixture
def window():
    if False:
        while True:
            i = 10
    return webview.create_window('Evaluate JS test')

def test_mixed(window):
    if False:
        return 10
    run_test(webview, window, mixed_test)

def test_array(window):
    if False:
        print('Hello World!')
    run_test(webview, window, array_test)

def test_object(window):
    if False:
        for i in range(10):
            print('nop')
    run_test(webview, window, object_test)

def test_string(window):
    if False:
        for i in range(10):
            print('nop')
    run_test(webview, window, string_test)

def test_int(window):
    if False:
        while True:
            i = 10
    run_test(webview, window, int_test)

def test_float(window):
    if False:
        return 10
    run_test(webview, window, float_test)

def test_undefined(window):
    if False:
        for i in range(10):
            print('nop')
    run_test(webview, window, undefined_test)

def test_null(window):
    if False:
        print('Hello World!')
    run_test(webview, window, null_test)

def test_nan(window):
    if False:
        while True:
            i = 10
    run_test(webview, window, nan_test)

def mixed_test(window):
    if False:
        for i in range(10):
            print('nop')
    result = window.evaluate_js("\n        document.body.style.backgroundColor = '#212121';\n        // comment\n        function test() {\n            return 2 + 2;\n        }\n        test();\n    ")
    assert result == 4

def array_test(window):
    if False:
        i = 10
        return i + 15
    result = window.evaluate_js("\n    function getValue() {\n        return [undefined, 1, 'two', 3.00001, {four: true}]\n    }\n    getValue()\n    ")
    assert result == [None, 1, 'two', 3.00001, {'four': True}]

def object_test(window):
    if False:
        for i in range(10):
            print('nop')
    result = window.evaluate_js("\n    function getValue() {\n        return {1: 2, 'test': true, obj: {2: false, 3: 3.1}}\n    }\n\n    getValue()\n    ")
    assert result == {'1': 2, 'test': True, 'obj': {'2': False, '3': 3.1}}

def string_test(window):
    if False:
        for i in range(10):
            print('nop')
    result = window.evaluate_js('\n    function getValue() {\n        return "this is only a test"\n    }\n\n    getValue()\n    ')
    assert result == 'this is only a test'

def int_test(window):
    if False:
        i = 10
        return i + 15
    result = window.evaluate_js('\n    function getValue() {\n        return 23\n    }\n\n    getValue()\n    ')
    assert result == 23

def float_test(window):
    if False:
        print('Hello World!')
    result = window.evaluate_js('\n    function getValue() {\n        return 23.23443\n    }\n\n    getValue()\n    ')
    assert result == 23.23443

def undefined_test(window):
    if False:
        i = 10
        return i + 15
    result = window.evaluate_js('\n    function getValue() {\n        return undefined\n    }\n\n    getValue()\n    ')
    assert result is None

def null_test(window):
    if False:
        for i in range(10):
            print('nop')
    result = window.evaluate_js('\n    function getValue() {\n        return null\n    }\n\n    getValue()\n    ')
    assert result is None

def nan_test(window):
    if False:
        while True:
            i = 10
    result = window.evaluate_js('\n    function getValue() {\n        return NaN\n    }\n\n    getValue()\n    ')
    assert result is None