import re
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Any
import pytest
from pytest_pyodide import run_in_pyodide
from pytest_pyodide.fixture import selenium_standalone_noload_common
from pytest_pyodide.server import spawn_web_server
from conftest import DIST_PATH, ROOT_PATH, strip_assertions_stderr
from pyodide.code import CodeRunner, eval_code, find_imports, should_quiet
from pyodide_build.build_env import get_pyodide_root

def test_find_imports():
    if False:
        for i in range(10):
            print('nop')
    res = find_imports('\n        import numpy as np\n        from scipy import sparse\n        import matplotlib.pyplot as plt\n        ')
    assert set(res) == {'numpy', 'scipy', 'matplotlib'}
    res = find_imports('\n        import numpy as np\n        from scipy import sparse\n        import matplotlib.pyplot as plt\n        for x in [1,2,3]\n        ')
    assert res == []

def test_ffi_import_star():
    if False:
        while True:
            i = 10
    exec('from pyodide.ffi import *', {})

def test_pyimport(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        let platform = pyodide.pyimport("platform");\n        assert(() => platform.machine() === "wasm32");\n        assert(() => !pyodide.globals.has("platform"))\n        assertThrows(() => pyodide.pyimport("platform;"), "PythonError", "ModuleNotFoundError: No module named \'platform;\'");\n        platform.destroy();\n        ')

def test_code_runner():
    if False:
        i = 10
        return i + 15
    assert should_quiet('1+1;')
    assert not should_quiet('1+1#;')
    assert not should_quiet('5-2  # comment with trailing semicolon ;')
    assert CodeRunner('1+1').compile().run() == 2
    assert CodeRunner('1+1\n1+1').compile().run() == 2
    assert CodeRunner('x + 7').compile().run({'x': 3}) == 10
    cr = CodeRunner('x + 7')
    import ast
    l = cr.ast.body[0].value.left
    cr.ast.body[0].value.left = ast.BinOp(left=l, op=ast.Mult(), right=ast.Constant(value=2))
    assert cr.compile().run({'x': 3}) == 13
    assert cr.code
    cr.code = cr.code.replace(co_consts=(0, 3, 5, None))
    assert cr.run({'x': 4}) == 17

def test_code_runner_mode():
    if False:
        return 10
    from codeop import PyCF_DONT_IMPLY_DEDENT
    assert CodeRunner('1+1\n1+1', mode='exec').compile().run() == 2
    with pytest.raises(SyntaxError, match='invalid syntax'):
        CodeRunner('1+1\n1+1', mode='eval').compile().run()
    with pytest.raises(SyntaxError, match='multiple statements found while compiling a single statement'):
        CodeRunner('1+1\n1+1', mode='single').compile().run()
    with pytest.raises(SyntaxError, match='invalid syntax'):
        CodeRunner('def f():\n  1', mode='single', flags=PyCF_DONT_IMPLY_DEDENT).compile().run()

def test_eval_code():
    if False:
        for i in range(10):
            print('nop')
    ns: dict[str, Any] = {}
    assert eval_code('\n        def f(x):\n            return 2*x + 5\n        f(77)\n    ', ns) == 2 * 77 + 5
    assert ns['f'](7) == 2 * 7 + 5
    assert eval_code('(x:=4)', ns) == 4
    assert ns['x'] == 4
    assert eval_code('x=7', ns) is None
    assert ns['x'] == 7
    assert eval_code('1+1;', ns) is None
    assert eval_code('1+1#;', ns) == 2
    assert eval_code('5-2  # comment with trailing semicolon ;', ns) == 3
    assert eval_code('4//2\n', ns) == 2
    assert eval_code('2**1\n\n', ns) == 2
    assert eval_code('4//2;\n', ns) is None
    assert eval_code('2**1;\n\n', ns) is None
    assert eval_code('1 + 1', ns, return_mode='last_expr_or_assign') == 2
    assert eval_code('x = 1 + 1', ns, return_mode='last_expr_or_assign') == 2
    assert eval_code('a = 5 ; a += 1', ns, return_mode='last_expr_or_assign') == 6
    assert eval_code('a = 5 ; a += 1;', ns, return_mode='last_expr_or_assign') is None
    assert eval_code('l = [1, 1, 2] ; l[0] = 0', ns, return_mode='last_expr_or_assign') is None
    assert eval_code('a = b = 2', ns, return_mode='last_expr_or_assign') == 2
    assert eval_code('1 + 1', ns, return_mode='none') is None
    assert eval_code('x = 1 + 1', ns, return_mode='none') is None
    assert eval_code('a = 5 ; a += 1', ns, return_mode='none') is None
    assert eval_code('a = 5 ; a += 1;', ns, return_mode='none') is None
    assert eval_code('l = [1, 1, 2] ; l[0] = 0', ns, return_mode='none') is None
    assert eval_code('1+1;', ns, quiet_trailing_semicolon=False) == 2
    assert eval_code('1+1#;', ns, quiet_trailing_semicolon=False) == 2
    assert eval_code('5-2  # comment with trailing semicolon ;', ns, quiet_trailing_semicolon=False) == 3
    assert eval_code('4//2\n', ns, quiet_trailing_semicolon=False) == 2
    assert eval_code('2**1\n\n', ns, quiet_trailing_semicolon=False) == 2
    assert eval_code('4//2;\n', ns, quiet_trailing_semicolon=False) == 2
    assert eval_code('2**1;\n\n', ns, quiet_trailing_semicolon=False) == 2

def test_eval_code_locals():
    if False:
        while True:
            i = 10
    globals: dict[str, Any] = {}
    eval_code('x=2', globals, {})
    with pytest.raises(NameError):
        eval_code('x', globals, {})
    locals: dict[str, Any] = {}
    eval_code('import sys; sys.getrecursionlimit()', globals, locals)
    with pytest.raises(NameError):
        eval_code('sys.getrecursionlimit()', globals, {})
    eval_code('sys.getrecursionlimit()', globals, locals)
    eval_code('from importlib import invalidate_caches; invalidate_caches()', globals, locals)
    with pytest.raises(NameError):
        eval_code('invalidate_caches()', globals, globals)
    eval_code('invalidate_caches()', globals, locals)
    with pytest.raises(NameError):
        eval_code('print(self)')
    res = eval_code('\n        var = "Hello"\n        def test():\n            return var\n        test()\n        ')
    assert res == 'Hello'

def test_unpack_archive(selenium_standalone):
    if False:
        while True:
            i = 10
    selenium = selenium_standalone
    js_error = selenium.run_js('\n        var error = "";\n        try {\n            pyodide.unpackArchive([1, 2, 3], "zip", "abc");\n        } catch (te) {\n            error = te.toString();\n        }\n        return error\n        ')
    expected_err_msg = "TypeError: Expected argument 'buffer' to be an ArrayBuffer or an ArrayBuffer view"
    assert js_error == expected_err_msg

@run_in_pyodide
def test_dup_pipe(selenium):
    if False:
        i = 10
        return i + 15
    import os
    [fdr1, fdw1] = os.pipe()
    fdr2 = os.dup(fdr1)
    fdw2 = os.dup2(fdw1, 50)
    s1 = b'some stuff'
    s2 = b'other stuff to write'
    os.write(fdw1, s1)
    assert os.read(fdr2, 100) == s1
    os.write(fdw2, s2)
    assert os.read(fdr1, 100) == s2

@run_in_pyodide
def test_dup_temp_file(selenium):
    if False:
        for i in range(10):
            print('nop')
    import os
    from tempfile import TemporaryFile
    tf = TemporaryFile(buffering=0)
    fd1 = os.dup(tf.fileno())
    os.dup2(tf.fileno(), 50)
    s = b'hello there!'
    tf.write(s)
    tf2 = open(fd1, 'w+')
    assert tf2.tell() == len(s)
    assert os.read(fd1, 50) == b''
    tf2.seek(1)
    assert tf.tell() == 1
    assert tf.read(100) == b'ello there!'

@run_in_pyodide
def test_dup_stdout(selenium):
    if False:
        print('Hello World!')
    import os
    import sys
    from tempfile import TemporaryFile
    tf = TemporaryFile(buffering=0)
    save_stdout = os.dup(sys.stdout.fileno())
    os.dup2(tf.fileno(), sys.stdout.fileno())
    print('hi!!')
    print('there...')
    assert tf.tell() == len('hi!!\nthere...\n')
    os.dup2(save_stdout, sys.stdout.fileno())
    print('not captured')
    os.dup2(tf.fileno(), sys.stdout.fileno())
    print('captured')
    assert tf.tell() == len('hi!!\nthere...\ncaptured\n')
    os.dup2(save_stdout, sys.stdout.fileno())
    os.close(save_stdout)
    tf.seek(0)
    assert tf.read(1000).decode() == 'hi!!\nthere...\ncaptured\n'

@pytest.mark.skip_pyproxy_check
def test_monkeypatch_eval_code(selenium):
    if False:
        while True:
            i = 10
    try:
        selenium.run('\n            import pyodide\n            old_eval_code = pyodide.code.eval_code\n            x = 3\n            def eval_code(code, globals=None, locals=None):\n                return [globals["x"], old_eval_code(code, globals, locals)]\n            pyodide.code.eval_code = eval_code\n            ')
        assert selenium.run('x = 99; 5') == [3, 5]
        assert selenium.run('7') == [99, 7]
    finally:
        selenium.run('\n            pyodide.code.eval_code = old_eval_code\n            ')

def test_promise_check(selenium):
    if False:
        i = 10
        return i + 15
    for s in ['0', '1', "'x'", "''", 'false', 'undefined', 'null', 'NaN', '0n', '[0,1,2]', '[]', '{}', '{a : 2}', '(()=>{})', '((x) => x*x)', '(function(x, y){ return x*x + y*y; })', 'Array', 'Map', 'Set', 'Promise', 'new Array()', 'new Map()', 'new Set()']:
        assert selenium.run_js(f'return pyodide._api.isPromise({s}) === false;')
    if not selenium.browser == 'node':
        assert selenium.run_js('return pyodide._api.isPromise(document.all) === false;')
    assert selenium.run_js('return pyodide._api.isPromise(Promise.resolve()) === true;')
    assert selenium.run_js('\n        return pyodide._api.isPromise(new Promise((resolve, reject) => {}));\n        ')
    assert not selenium.run_js('\n        let d = pyodide.runPython("{}");\n        try {\n            return pyodide._api.isPromise(d);\n        } finally {\n            d.destroy();\n        }\n        ')

def test_keyboard_interrupt(selenium):
    if False:
        return 10
    x = selenium.run_js("\n        let x = new Int8Array(1);\n        pyodide.setInterruptBuffer(x);\n        self.triggerKeyboardInterrupt = function(){\n            x[0] = 2;\n        }\n        try {\n            pyodide.runPython(`\n                from js import triggerKeyboardInterrupt\n                for x in range(100000):\n                    if x == 2000:\n                        triggerKeyboardInterrupt()\n            `);\n        } catch(e){}\n        pyodide.setInterruptBuffer(undefined);\n        return pyodide.globals.get('x');\n        ")
    assert 2000 < x < 2500

def test_run_python_async_toplevel_await(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        await pyodide.runPythonAsync(`\n            from js import fetch\n            resp = await fetch("pyodide-lock.json")\n            json = (await resp.json()).to_py()["packages"]\n            assert "micropip" in json\n        `);\n        ')

def test_run_python_proxy_leak(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        pyodide.runPython("")\n        await pyodide.runPythonAsync("")\n        ')

def test_run_python_last_exc(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        try {\n            pyodide.runPython("x = ValueError(77); raise x");\n        } catch(e){}\n        pyodide.runPython(`\n            import sys\n            assert sys.last_value is x\n            assert sys.last_type is type(x)\n            assert sys.last_traceback is x.__traceback__\n        `);\n        ')

def test_check_interrupt(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        pyodide.setInterruptBuffer(undefined);\n        pyodide.checkInterrupt();\n        ')
    assert selenium.run_js('\n        let buffer = new Uint8Array(1);\n        let x = 0;\n        pyodide.setInterruptBuffer(buffer);\n        function test(){\n            buffer[0] = 2;\n            pyodide.checkInterrupt();\n            x = 1;\n        }\n        self.test = test;\n        let err;\n        try {\n            pyodide.runPython(`\n                from js import test;\n                try:\n                    test();\n                finally:\n                    del test\n            `);\n        } catch(e){\n            err = e;\n        }\n        return x === 0 && err.message.includes("KeyboardInterrupt");\n        ')
    assert selenium.run_js('\n        let buffer = new Uint8Array(1);\n        pyodide.setInterruptBuffer(buffer);\n        buffer[0] = 2;\n        let err_code = 0;\n        for(let i = 0; i < 1000; i++){\n            err_code = err_code || pyodide._module._PyErr_CheckSignals();\n        }\n        let err_occurred = pyodide._module._PyErr_Occurred();\n        console.log({err_code, err_occurred});\n        pyodide._module._PyErr_Clear();\n        return buffer[0] === 0 && err_code === -1 && err_occurred !== 0;\n        ')

def test_check_interrupt_no_gil(selenium):
    if False:
        return 10
    'Check interrupt has a special case for GIL not held.\n    Make sure that it works.\n    '
    selenium.run_js('\n        // release GIL\n        const tstate = pyodide._module._PyEval_SaveThread();\n\n        try {\n            // check that checkInterrupt works when interrupt buffer not defined\n            // it should do nothing.\n            pyodide.setInterruptBuffer(undefined);\n            pyodide.checkInterrupt();\n            ib = new Int32Array(1);\n            pyodide.setInterruptBuffer(ib);\n            pyodide.checkInterrupt();\n\n            ib[0] = 2;\n            let err;\n            try {\n                pyodide.checkInterrupt();\n            } catch(e) {\n                err = e;\n            }\n            assert(() => err instanceof pyodide.FS.ErrnoError);\n            assert(() => err.errno === pyodide.ERRNO_CODES.EINTR);\n            assert(() => ib[0] === 2);\n            ib[0] = 0;\n        } finally {\n            // acquire GIL\n            pyodide._module._PyEval_RestoreThread(tstate)\n        }\n        ')

def test_check_interrupt_custom_signal_handler(selenium):
    if False:
        print('Hello World!')
    try:
        selenium.run_js('\n            pyodide.runPython(`\n                import signal\n                interrupt_occurred = False\n                def signal_handler(*args):\n                    global interrupt_occurred\n                    interrupt_occurred = True\n                signal.signal(signal.SIGINT, signal_handler)\n                None\n            `);\n            ')
        selenium.run_js('\n            let buffer = new Uint8Array(1);\n            let x = 0;\n            pyodide.setInterruptBuffer(buffer);\n            function test(){\n                buffer[0] = 2;\n                pyodide.checkInterrupt();\n                x = 1;\n            }\n            self.test = test;\n            let err;\n            pyodide.runPython(`\n                interrupt_occurred = False\n                from js import test\n                test()\n                assert interrupt_occurred == True\n                del test\n            `);\n            ')
        assert selenium.run_js('\n            pyodide.runPython(`\n                interrupt_occurred = False\n            `);\n            let buffer = new Uint8Array(1);\n            pyodide.setInterruptBuffer(buffer);\n            buffer[0] = 2;\n            let err_code = 0;\n            for(let i = 0; i < 1000; i++){\n                err_code = err_code || pyodide._module._PyErr_CheckSignals();\n            }\n            let interrupt_occurred = pyodide.globals.get("interrupt_occurred");\n\n            return buffer[0] === 0 && err_code === 0 && interrupt_occurred;\n            ')
    finally:
        selenium.run_js('\n            pyodide.runPython(`\n                import signal\n                signal.signal(signal.SIGINT, signal.default_int_handler)\n                None\n            `);\n            ')

def test_async_leak(selenium):
    if False:
        return 10
    assert 0 == selenium.run_js('\n        pyodide.runPython(`d = 888.888`);\n        pyodide.runPython(`async def test(): return d`);\n        async function test(){\n            let t = pyodide.runPython(`test()`);\n            await t;\n            t.destroy();\n        }\n        await test();\n        let init_refcount = pyodide.runPython(`from sys import getrefcount; getrefcount(d)`);\n        await test(); await test(); await test(); await test();\n        let new_refcount = pyodide.runPython(`getrefcount(d)`);\n        return new_refcount - init_refcount;\n        ')

def test_run_python_js_error(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        function throwError(){\n            throw new Error("blah!");\n        }\n        self.throwError = throwError;\n        pyodide.runPython(`\n            from js import throwError\n            from unittest import TestCase\n            from pyodide.ffi import JsException\n            raises = TestCase().assertRaisesRegex\n            with raises(JsException, "blah!"):\n                throwError()\n        `);\n        ')

@pytest.mark.xfail_browsers(node='No DOMException in node')
@run_in_pyodide
def test_run_python_dom_error(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from js import DOMException
    from pyodide.ffi import JsException
    with pytest.raises(JsException, match='oops'):
        raise DOMException.new('oops')

def test_run_python_locals(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        let dict = pyodide.globals.get("dict");\n        let locals = dict([["x", 7]]);\n        let globals = dict([["x", 5], ["y", 29]]);\n        dict.destroy();\n        let result = pyodide.runPython("z = 13; x + y", {locals, globals});\n        assert(() => locals.get("z") === 13);\n        assert(() => locals.has("x"));\n        let result2 = pyodide.runPython("del x; x + y", {locals, globals});\n        assert(() => !locals.has("x"));\n        assert(() => result === 7 + 29);\n        assert(() => result2 === 5 + 29);\n        locals.destroy();\n        globals.destroy();\n        ')

def test_create_once_callable(selenium):
    if False:
        return 10
    selenium.run_js('\n        self.call7 = function call7(f){\n            return f(7);\n        }\n        pyodide.runPython(`\n            from pyodide.ffi import create_once_callable, JsException\n            from js import call7;\n            from unittest import TestCase\n            raises = TestCase().assertRaisesRegex\n            class Square:\n                def __call__(self, x):\n                    return x*x\n\n                def __del__(self):\n                    global destroyed\n                    destroyed = True\n\n            f = Square()\n            import sys\n            assert sys.getrefcount(f) == 2\n            proxy = create_once_callable(f)\n            assert sys.getrefcount(f) == 3\n            assert call7(proxy) == 49\n            assert sys.getrefcount(f) == 2\n            with raises(JsException, "can only be called once"):\n                call7(proxy)\n            destroyed = False\n            del f\n            assert destroyed == True\n            del proxy\n        `);\n        ')

@run_in_pyodide
def test_create_proxy(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    from pyodide.ffi import create_proxy
    [testAddListener, testCallListener, testRemoveListener] = run_js('\n        function testAddListener(f){\n            self.listener = f;\n        }\n        function testCallListener(f){\n            return self.listener();\n        }\n        function testRemoveListener(f){\n            return self.listener === f;\n        }\n        [testAddListener, testCallListener, testRemoveListener]\n        ')
    destroyed = False

    class Test:

        def __call__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 7

        def __del__(self):
            if False:
                print('Hello World!')
            nonlocal destroyed
            destroyed = True
    f = Test()
    import sys
    assert sys.getrefcount(f) == 2
    proxy = create_proxy(f)
    assert sys.getrefcount(f) == 3
    assert proxy() == 7
    testAddListener(proxy)
    assert sys.getrefcount(f) == 3
    assert testCallListener() == 7
    assert sys.getrefcount(f) == 3
    assert testCallListener() == 7
    assert sys.getrefcount(f) == 3
    assert testRemoveListener(proxy)
    assert sys.getrefcount(f) == 3
    proxy.destroy()
    assert sys.getrefcount(f) == 2
    destroyed = False
    del f
    assert destroyed

@run_in_pyodide
def test_create_proxy_capture_this(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    from pyodide.ffi import create_proxy
    o = run_js('({})')

    def f(self):
        if False:
            return 10
        assert self == o
    o.f = create_proxy(f, capture_this=True)
    run_js('(o) => { o.f(); o.f.destroy(); }')(o)

@run_in_pyodide
def test_create_proxy_roundtrip(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    from pyodide.ffi import JsDoubleProxy, create_proxy
    f = {}
    o = run_js('({})')
    o.f = create_proxy(f, roundtrip=True)
    assert isinstance(o.f, JsDoubleProxy)
    assert o.f.unwrap() is f
    o.f.destroy()
    o.f = create_proxy(f, roundtrip=False)
    assert o.f is f
    run_js('(o) => { o.f.destroy(); }')(o)

@run_in_pyodide
def test_return_destroyed_value(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsException, create_proxy
    f = run_js('(function(x){ return x; })')
    p = create_proxy([])
    p.destroy()
    with pytest.raises(JsException, match='Object has already been destroyed'):
        f(p)

def test_docstrings_a():
    if False:
        while True:
            i = 10
    from _pyodide._core_docs import _instantiate_token
    from _pyodide.docstring import dedent_docstring, get_cmeth_docstring
    from pyodide.ffi import JsPromise
    jsproxy = JsPromise(_instantiate_token)
    c_docstring = get_cmeth_docstring(jsproxy.then)
    assert c_docstring == 'then(onfulfilled, onrejected)\n--\n\n' + dedent_docstring(jsproxy.then.__doc__)

def test_docstrings_b(selenium):
    if False:
        while True:
            i = 10
    from _pyodide._core_docs import _instantiate_token
    from _pyodide.docstring import dedent_docstring
    from pyodide.ffi import JsPromise, create_once_callable
    jsproxy = JsPromise(_instantiate_token)
    ds_then_should_equal = dedent_docstring(jsproxy.then.__doc__)
    sig_then_should_equal = '(onfulfilled, onrejected)'
    ds_once_should_equal = dedent_docstring(create_once_callable.__doc__)
    sig_once_should_equal = '(obj, /)'
    selenium.run_js('self.a = Promise.resolve();')
    [ds_then, sig_then, ds_once, sig_once] = selenium.run('\n        from js import a\n        from pyodide.ffi import create_once_callable as b\n        [\n            a.then.__doc__, a.then.__text_signature__,\n            b.__doc__, b.__text_signature__\n        ]\n        ')
    assert ds_then == ds_then_should_equal
    assert sig_then == sig_then_should_equal
    assert ds_once == ds_once_should_equal
    assert sig_once == sig_once_should_equal

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
def test_restore_state(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        pyodide.registerJsModule("a", {somefield : 82});\n        pyodide.registerJsModule("b", { otherfield : 3 });\n        pyodide.runPython("x = 7; from a import somefield");\n        let state = pyodide._api.saveState();\n\n        pyodide.registerJsModule("c", { thirdfield : 9 });\n        pyodide.runPython("y = 77; from b import otherfield; import c;");\n        pyodide._api.restoreState(state);\n        state.destroy();\n        ')
    selenium.run('\n        from unittest import TestCase\n        raises = TestCase().assertRaises\n        import sys\n\n        assert x == 7\n        assert "a" in sys.modules\n        assert somefield == 82\n        with raises(NameError):\n            y\n        with raises(NameError):\n            otherfield\n        assert "b" not in sys.modules\n        import b\n        with raises(ModuleNotFoundError):\n            import c\n        ')

@pytest.mark.xfail_browsers(safari='TODO: traceback is not the same on Safari')
@pytest.mark.skip_refcount_check
def test_fatal_error(selenium_standalone):
    if False:
        return 10
    assert selenium_standalone.run_js('\n        try {\n            pyodide.runPython(`\n                from _pyodide_core import trigger_fatal_error\n                def f():\n                    g()\n                def g():\n                    h()\n                def h():\n                    trigger_fatal_error()\n                f()\n            `);\n        } catch(e){\n            return e.toString();\n        }\n        ')
    import re

    def strip_stack_trace(x):
        if False:
            i = 10
            return i + 15
        x = re.sub('\n.*site-packages.*', '', x)
        x = re.sub('/lib/python.*/', '', x)
        x = re.sub('/lib/python.*/', '', x)
        x = re.sub('warning: no [bB]lob.*\n', '', x)
        x = re.sub('Error: intentionally triggered fatal error!\n', '', x)
        x = re.sub(' +at .*\n', '', x)
        x = re.sub('.*@https?://[0-9.:]*/.*\n', '', x)
        x = re.sub('.*@debugger.*\n', '', x)
        x = re.sub('.*@chrome.*\n', '', x)
        x = re.sub('line [0-9]*', 'line xxx', x)
        x = x.replace('\n\n', '\n')
        return x
    err_msg = strip_stack_trace(selenium_standalone.logs)
    err_msg = ''.join(strip_assertions_stderr(err_msg.splitlines(keepends=True)))
    assert err_msg == dedent(strip_stack_trace('\n                Pyodide has suffered a fatal error. Please report this to the Pyodide maintainers.\n                The cause of the fatal error was:\n                Stack (most recent call first):\n                  File "<exec>", line 8 in h\n                  File "<exec>", line 6 in g\n                  File "<exec>", line 4 in f\n                  File "<exec>", line 9 in <module>\n                  File "/lib/pythonxxx/pyodide/_base.py", line 242 in run\n                  File "/lib/pythonxxx/pyodide/_base.py", line 344 in eval_code\n                ')).strip()
    selenium_standalone.run_js('\n        assertThrows(() => pyodide.runPython, "Error", "Pyodide already fatally failed and can no longer be used.")\n        assertThrows(() => pyodide.globals, "Error", "Pyodide already fatally failed and can no longer be used.")\n        ')

@pytest.mark.skip_refcount_check
def test_exit_error(selenium_standalone):
    if False:
        for i in range(10):
            print('nop')
    x = selenium_standalone.run_js('\n        try {\n            pyodide.runPython(`\n                import os\n                def f():\n                    g()\n                def g():\n                    h()\n                def h():\n                    os._exit(0)\n                f()\n            `);\n        } catch(e){\n            return e.toString();\n        }\n        ')
    assert x == 'Exit: Program terminated with exit(0)'

def test_reentrant_error(selenium):
    if False:
        i = 10
        return i + 15
    caught = selenium.run_js('\n        function raisePythonKeyboardInterrupt(){\n            pyodide.globals.get("pyfunc")();\n        }\n        let caught = false;\n        try {\n            pyodide.runPython(`\n                def pyfunc():\n                    raise KeyboardInterrupt\n                from js import raisePythonKeyboardInterrupt\n                try:\n                    raisePythonKeyboardInterrupt()\n                except Exception as e:\n                    pass\n            `);\n        } catch(e){\n            caught = true;\n        }\n        return caught;\n        ')
    assert caught

@pytest.mark.xfail_browsers(safari='TODO: traceback is not exactly the same on Safari')
def test_js_stackframes(selenium):
    if False:
        return 10
    res = selenium.run_js('\n        self.b = function b(){\n            pyodide.pyimport("???");\n        }\n        self.d1 = function d1(){\n            pyodide.runPython("c2()");\n        }\n        self.d2 = function d2(){\n            d1();\n        }\n        self.d3 = function d3(){\n            d2();\n        }\n        self.d4 = function d4(){\n            d3();\n        }\n        pyodide.runPython(`\n            def c1():\n                from js import b\n                b()\n            def c2():\n                c1()\n            def e():\n                from js import d4\n                from pyodide.ffi import to_js\n                from traceback import extract_tb\n                try:\n                    d4()\n                except Exception as ex:\n                    return to_js([[x.filename, x.name] for x in extract_tb(ex.__traceback__)])\n        `);\n        let e = pyodide.globals.get("e");\n        let res = e();\n        e.destroy();\n        return res;\n        ')

    def normalize_tb(t):
        if False:
            while True:
                i = 10
        res = []
        for [file, name] in t:
            if file.endswith(('.js', '.html')):
                file = file.rpartition('/')[-1]
            if file.endswith('.py'):
                file = '/'.join(file.split('/')[-2:])
            if re.fullmatch('\\:[0-9]*', file) or file == 'evalmachine.<anonymous>' or file == 'debugger eval code':
                file = 'test.html'
            res.append([file, name])
        return res
    frames = [['<exec>', 'e'], ['test.html', 'd4'], ['test.html', 'd3'], ['test.html', 'd2'], ['test.html', 'd1'], ['pyodide.asm.js', 'runPython'], ['_pyodide/_base.py', 'eval_code'], ['_pyodide/_base.py', 'run'], ['<exec>', '<module>'], ['<exec>', 'c2'], ['<exec>', 'c1'], ['test.html', 'b'], ['pyodide.asm.js', 'pyimport'], ['importlib/__init__.py', 'import_module']]
    assert normalize_tb(res[:len(frames)]) == frames

def test_reentrant_fatal(selenium_standalone):
    if False:
        while True:
            i = 10
    selenium = selenium_standalone
    assert selenium.run_js('\n        function f(){\n            pyodide.globals.get("trigger_fatal_error")();\n        }\n        self.success = true;\n        try {\n            pyodide.runPython(`\n                from _pyodide_core import trigger_fatal_error\n                from js import f\n                try:\n                    f()\n                except Exception as e:\n                    # This code shouldn\'t be executed\n                    import js\n                    js.success = False\n            `);\n        } catch(e){}\n        return success;\n        ')

def test_weird_throws(selenium):
    if False:
        print('Hello World!')
    'Throw strange Javascript garbage and make sure we survive.'
    selenium.run_js('\n        self.funcs = {\n            null(){ throw null; },\n            undefined(){ throw undefined; },\n            obj(){ throw {}; },\n            obj_null_proto(){ throw Object.create(null); },\n            string(){ throw "abc"; },\n            func(){ throw self.funcs.func; },\n            number(){ throw 12; },\n            bigint(){ throw 12n; },\n        };\n        pyodide.runPython(`\n            from js import funcs\n            from unittest import TestCase\n            from pyodide.ffi import JsException\n            raises = TestCase().assertRaisesRegex\n            msgs = {\n                "null" : [\'type object .* tag .object Null.\', \'"""null"""\',  \'fails\'],\n                "undefined" : [\'type undefined .* tag .object Undefined.\', \'"""undefined"""\',  \'fails\'],\n                "obj" : [\'type object .* tag .object Object.\', \'""".object Object."""\',  \'""".object Object."""\'],\n                "obj_null_proto" : [\'type object .* tag .object Object.\', \'fails\',  \'fails\'],\n                "string" : ["Error: abc"],\n                "func" : [\'type function .* tag .object Function.\', \'throw self.funcs.func\',  \'throw self.funcs.func\'],\n                "number" : [\'type number .* tag .object Number.\'],\n                "bigint" : [\'type bigint .* tag .object BigInt.\'],\n            }\n            for name, f in funcs.object_entries():\n                msg = \'.*\\\\n.*\'.join(msgs.get(name, ["xx"]))\n                with raises(JsException, msg):\n                    f()\n        `);\n        ')

@pytest.mark.skip_refcount_check
@pytest.mark.parametrize('to_throw', ['Object.create(null);', "'Some message'", 'null'])
def test_weird_fatals(selenium_standalone, to_throw):
    if False:
        i = 10
        return i + 15
    expected_message = {'Object.create(null);': 'Error: A value of type object with tag [object Object] was thrown as an error!', "'Some message'": 'Error: Some message', 'null': 'Error: A value of type object with tag [object Null] was thrown as an error!'}[to_throw]
    msg = selenium_standalone.run_js(f'\n        self.f = function(){{ throw {to_throw} }};\n        \n        try {{\n            pyodide.runPython(`\n                from _pyodide_core import raw_call\n                from js import f\n                raw_call(f)\n            `);\n        }} catch(e){{\n            return e.toString();\n        }}\n        ')
    print('msg', msg[:len(expected_message)])
    print('expected_message', expected_message)
    assert msg[:len(expected_message)] == expected_message

def test_restore_error(selenium):
    if False:
        print('Hello World!')
    selenium.run_js("\n        self.f = function(){\n            pyodide.runPython(`\n                err = Exception('hi')\n                raise err\n            `);\n        }\n        pyodide.runPython(`\n            from js import f\n            import sys\n            try:\n                f()\n            except Exception as e:\n                assert err == e\n                assert e == sys.last_value\n            finally:\n                del err\n            assert sys.getrefcount(sys.last_value) == 2\n        `);\n        ")

def test_home_directory(selenium_standalone_noload):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_standalone_noload
    selenium.run_js('\n        const homedir = "/home/custom_home";\n        const pyodide = await loadPyodide({\n            homedir,\n        });\n        return pyodide.runPython(`\n            import os\n            os.getcwd() == "${homedir}"\n        `)\n        ')
    assert 'The homedir argument to loadPyodide is deprecated' in selenium.logs

def test_env(selenium_standalone_noload):
    if False:
        return 10
    selenium = selenium_standalone_noload
    hashval = selenium.run_js('\n        let pyodide = await loadPyodide({\n            env : {PYTHONHASHSEED : 1},\n        });\n        return pyodide.runPython(`\n            hash((1,2,3))\n        `)\n        ')
    assert hashval == -2022708474

def test_version_variable(selenium):
    if False:
        i = 10
        return i + 15
    js_version = selenium.run_js('\n        return pyodide.version\n        ')
    core_version = selenium.run_js('\n        return pyodide._api.version\n        ')
    from pyodide import __version__ as py_version
    assert js_version == py_version == core_version

@run_in_pyodide
def test_default_sys_path(selenium):
    if False:
        i = 10
        return i + 15
    import sys
    from sys import version_info
    major = version_info[0]
    minor = version_info[1]
    prefix = sys.prefix
    platlibdir = sys.platlibdir
    paths = [f'{prefix}{platlibdir}/python{major}{minor}.zip', f'{prefix}{platlibdir}/python{major}.{minor}', f'{prefix}{platlibdir}/python{major}.{minor}/lib-dynload', f'{prefix}{platlibdir}/python{major}.{minor}/site-packages']
    for path in paths:
        assert path in sys.path

def test_sys_path0(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        pyodide.runPython(`\n            import sys\n            import os\n            assert os.getcwd() == sys.path[0]\n        `)\n        ')

@pytest.mark.requires_dynamic_linking
def test_fullstdlib(selenium_standalone_noload):
    if False:
        while True:
            i = 10
    selenium = selenium_standalone_noload
    selenium.run_js('\n        let pyodide = await loadPyodide({\n            fullStdLib: true,\n        });\n\n        await pyodide.loadPackage("micropip");\n\n        pyodide.runPython(`\n            import pyodide_js\n            import micropip\n            loaded_packages = micropip.list()\n            assert all((lib in micropip.list()) for lib in pyodide_js._api.lockfile_unvendored_stdlibs)\n        `);\n        ')

def test_loadPyodide_relative_index_url(selenium_standalone_noload):
    if False:
        while True:
            i = 10
    'Check that loading Pyodide with a relative URL works'
    selenium_standalone_noload.run_js('\n        self.pyodide = await loadPyodide({ indexURL: "./" });\n        ')

@run_in_pyodide
def test_run_js(selenium):
    if False:
        i = 10
        return i + 15
    from unittest import TestCase
    from pyodide.code import run_js
    raises = TestCase().assertRaises
    with raises(TypeError, msg="argument should have type 'string' not type 'int'"):
        run_js(3)
    assert run_js('(x)=> x+1')(7) == 8
    assert run_js('[1,2,3]')[2] == 3
    run_js('globalThis.x = 77')
    from js import x
    assert x == 77

@run_in_pyodide
def test_pickle_jsexception(selenium):
    if False:
        return 10
    import pickle
    from pyodide.code import run_js
    pickle.dumps(run_js("new Error('hi');"))

def test_raises_jsexception(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.ffi import JsException

    @run_in_pyodide
    def raise_jsexception(selenium):
        if False:
            while True:
                i = 10
        from pyodide.code import run_js
        run_js("throw new Error('hi');")
    with pytest.raises(JsException, match='Error: hi'):
        raise_jsexception(selenium)

@pytest.mark.xfail_browsers(node='Some problem with the logs in node')
def test_deprecations(selenium_standalone):
    if False:
        return 10
    selenium = selenium_standalone
    selenium.run_js('\n        let a = pyodide.PyBuffer;\n        let b = pyodide.PyBuffer;\n        assert(() => a === b);\n        ')
    assert selenium.logs.count('pyodide.PyBuffer is deprecated. Use `pyodide.ffi.PyBufferView` instead.') == 1
    selenium.run_js('\n        let a = pyodide.PyProxyBuffer;\n        let b = pyodide.PyProxyBuffer;\n        assert(() => a === b);\n        ')
    assert selenium.logs.count('pyodide.PyProxyBuffer is deprecated. Use `pyodide.ffi.PyBuffer` instead.') == 1
    selenium.run_js('\n        assert(() => pyodide.isPyProxy(pyodide.globals));\n        assert(() => pyodide.isPyProxy(pyodide.globals));\n        assert(() => !pyodide.isPyProxy({}));\n        ')
    selenium.run_js('\n        assert(() => !pyodide.globals.isAwaitable());\n        assert(() => !pyodide.globals.isAwaitable());\n        assert(() => !pyodide.globals.isBuffer());\n        assert(() => !pyodide.globals.isBuffer());\n        assert(() => !pyodide.globals.isCallable());\n        assert(() => !pyodide.globals.isCallable());\n        assert(() => pyodide.globals.isIterable());\n        assert(() => pyodide.globals.isIterable());\n        assert(() => !pyodide.globals.isIterator());\n        assert(() => !pyodide.globals.isIterator());\n        assert(() => pyodide.globals.supportsGet());\n        assert(() => pyodide.globals.supportsGet());\n        assert(() => pyodide.globals.supportsSet());\n        assert(() => pyodide.globals.supportsSet());\n        assert(() => pyodide.globals.supportsHas());\n        assert(() => pyodide.globals.supportsHas());\n        ')
    for name in ['isPyProxy', 'isAwaitable', 'isBuffer', 'isCallable', 'isIterable', 'isIterator', 'supportsGet', 'supportsSet', 'supportsHas']:
        assert sum((f'{name}() is deprecated. Use' in s for s in selenium.logs.split('\n'))) == 1

@run_in_pyodide(packages=['pytest'])
def test_module_not_found_hook(selenium_standalone):
    if False:
        for i in range(10):
            print('nop')
    import importlib
    import pytest
    unvendored_stdlibs = ['test', 'ssl', 'lzma', 'sqlite3', '_hashlib']
    removed_stdlibs = ['pwd', 'turtle', 'tkinter']
    lockfile_packages = ['micropip', 'packaging', 'regex']
    for lib in unvendored_stdlibs:
        with pytest.raises(ModuleNotFoundError, match='unvendored from the Python standard library'):
            importlib.import_module(lib)
    for lib in removed_stdlibs:
        with pytest.raises(ModuleNotFoundError, match='removed from the Python standard library'):
            importlib.import_module(lib)
    with pytest.raises(ModuleNotFoundError, match='No module named'):
        importlib.import_module('urllib.there_is_no_such_module')
    for lib in lockfile_packages:
        with pytest.raises(ModuleNotFoundError, match='included in the Pyodide distribution'):
            importlib.import_module(lib)
    with pytest.raises(ModuleNotFoundError, match='No module named'):
        importlib.import_module('pytest.there_is_no_such_module')
    for pkg in ['liblzma', 'openssl']:
        with pytest.raises(ModuleNotFoundError, match='No module named'):
            importlib.import_module(pkg)
    with pytest.raises(ModuleNotFoundError, match='loadPackage\\("hashlib"\\)'):
        importlib.import_module('_hashlib')

def test_args(selenium_standalone_noload):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_standalone_noload
    assert selenium.run_js("\n            self.stdoutStrings = [];\n            self.stderrStrings = [];\n            function stdout(s){\n                stdoutStrings.push(s);\n            }\n            function stderr(s){\n                stderrStrings.push(s);\n            }\n            let pyodide = await loadPyodide({\n                fullStdLib: false,\n                jsglobals : self,\n                stdout,\n                stderr,\n                args: ['-c', 'print([x*x+1 for x in range(10)])']\n            });\n            self.pyodide = pyodide;\n            globalThis.pyodide = pyodide;\n            pyodide._module._run_main();\n            return stdoutStrings.pop()\n            ") == repr([x * x + 1 for x in range(10)])

def test_args_OO(selenium_standalone_noload):
    if False:
        return 10
    selenium = selenium_standalone_noload
    doc = selenium.run_js("\n        let pyodide = await loadPyodide({\n            args: ['-OO']\n        });\n        pyodide.runPython(`import sys; sys.__doc__`)\n        ")
    assert not doc

@pytest.mark.xfail_browsers(chrome='Node only', firefox='Node only', safari='Node only')
def test_relative_index_url(selenium, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    tmp_dir = Path(tmp_path)
    subprocess.run(['node', '-v'], capture_output=True, encoding='utf8')
    shutil.copy(ROOT_PATH / 'dist/pyodide.js', tmp_dir / 'pyodide.js')
    result = subprocess.run(['node', '-e', f'''\n            const loadPyodide = require("{tmp_dir / 'pyodide.js'}").loadPyodide;\n            async function main(){{\n                py = await loadPyodide({{indexURL: "./dist"}});\n                console.log("\\n");\n                console.log(py._module.API.config.indexURL);\n            }}\n            main();\n            '''], cwd=ROOT_PATH, capture_output=True, encoding='utf8')
    import textwrap

    def print_result(result):
        if False:
            return 10
        if result.stdout:
            print('  stdout:')
            print(textwrap.indent(result.stdout, '    '))
        if result.stderr:
            print('  stderr:')
            print(textwrap.indent(result.stderr, '    '))
    if result.returncode:
        print_result(result)
        result.check_returncode()
    try:
        assert result.stdout.strip().split('\n')[-1] == str(ROOT_PATH / 'dist') + '/'
    finally:
        print_result(result)

@pytest.mark.xfail_browsers(chrome='Node only', firefox='Node only', safari='Node only')
def test_index_url_calculation_source_map(selenium):
    if False:
        while True:
            i = 10
    import os
    node_options = ['--enable-source-maps']
    result = subprocess.run(['node', '-v'], capture_output=True, encoding='utf8')
    DIST_DIR = str(Path.cwd() / 'dist')
    env = os.environ.copy()
    env['DIST_DIR'] = DIST_DIR
    result = subprocess.run(['node', *node_options, '-e', '\n            const { loadPyodide } = require(`${process.env.DIST_DIR}/pyodide`);\n            async function main() {\n                const py = await loadPyodide();\n                console.log("indexURL:", py._module.API.config.indexURL);\n            }\n            main();\n            '], env=env, capture_output=True, encoding='utf8')
    assert f'indexURL: {DIST_DIR}' in result.stdout

@pytest.mark.xfail_browsers(chrome='Node only', firefox='Node only', safari='Node only')
@pytest.mark.parametrize('filename, import_stmt', [('index.js', "const { loadPyodide } = require('%s/pyodide.js')"), ('index.mjs', "import { loadPyodide } from '%s/pyodide.mjs'")])
def test_default_index_url_calculation_node(selenium, tmp_path, filename, import_stmt):
    if False:
        while True:
            i = 10
    Path(tmp_path / filename).write_text(import_stmt % DIST_PATH + '\n' + '\n        async function main() {\n            const py = await loadPyodide();\n            console.log("indexURL:", py._module.API.config.indexURL);\n        }\n        main();\n        ')
    result = subprocess.run(['node', filename], capture_output=True, encoding='utf8', cwd=tmp_path)
    assert f'indexURL: {DIST_PATH}' in result.stdout

@pytest.mark.xfail_browsers(node='Browser only', safari="Safari doesn't support wasm-unsafe-eval")
def test_csp(selenium_standalone_noload):
    if False:
        i = 10
        return i + 15
    selenium = selenium_standalone_noload
    target_path = DIST_PATH / 'test_csp.html'
    try:
        shutil.copy(get_pyodide_root() / 'src/templates/test_csp.html', target_path)
        selenium.goto(f'{selenium.base_url}/test_csp.html')
        selenium.javascript_setup()
        selenium.load_pyodide()
    finally:
        target_path.unlink()

def test_static_import(request, runtime, web_server_main, playwright_browsers, tmp_path):
    if False:
        i = 10
        return i + 15
    if runtime == 'node':
        pytest.xfail('static import test is browser-only')
    shutil.copytree(ROOT_PATH / 'dist', tmp_path, dirs_exist_ok=True)
    hiding_dir = 'hide_pyodide_asm_for_test'
    (tmp_path / hiding_dir).mkdir()
    shutil.move(tmp_path / 'pyodide.asm.js', tmp_path / hiding_dir / 'pyodide.asm.js')
    test_html = (ROOT_PATH / 'src/templates/module_static_import_test.html').read_text()
    test_html = test_html.replace('./pyodide.asm.js', f'./{hiding_dir}/pyodide.asm.js')
    (tmp_path / 'module_static_import_test.html').write_text(test_html)
    with spawn_web_server(tmp_path) as web_server, selenium_standalone_noload_common(request, runtime, web_server, playwright_browsers) as selenium:
        selenium.goto(f'{selenium.base_url}/module_static_import_test.html')
        selenium.javascript_setup()
        selenium.load_pyodide()
        selenium.run_js("\n            pyodide.runPython(`\n                print('Static import works')\n            `);\n            ")

def test_python_error(selenium):
    if False:
        i = 10
        return i + 15
    [msg, ty] = selenium.run_js('\n        try {\n            pyodide.runPython("raise TypeError(\'oops\')");\n        } catch(e) {\n            return [e.message, e.type];\n        }\n        ')
    assert msg.endswith('TypeError: oops\n')
    assert ty == 'TypeError'

def test_python_version(selenium):
    if False:
        return 10
    selenium.run_js('\n        sys = pyodide.pyimport("sys");\n        assert(() => sys.version_info.major === pyodide._module._py_version_major());\n        assert(() => sys.version_info.minor === pyodide._module._py_version_minor());\n        assert(() => sys.version_info.micro === pyodide._module._py_version_micro());\n        sys.destroy();\n        ')

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
def test_custom_python_stdlib_URL(selenium_standalone_noload, runtime):
    if False:
        print('Hello World!')
    selenium = selenium_standalone_noload
    stdlib_target_path = ROOT_PATH / 'dist/python_stdlib2.zip'
    shutil.copy(ROOT_PATH / 'dist/python_stdlib.zip', stdlib_target_path)
    try:
        selenium.run_js('\n            let pyodide = await loadPyodide({\n                fullStdLib: false,\n                stdLibURL: "./python_stdlib2.zip",\n            });\n            // Check that we can import stdlib library modules\n            let statistics = pyodide.pyimport(\'statistics\');\n            assert(() => statistics.median([2, 3, 1]) === 2)\n            ')
    finally:
        stdlib_target_path.unlink()

def test_pickle_internal_error(selenium):
    if False:
        while True:
            i = 10

    @run_in_pyodide
    def helper(selenium):
        if False:
            while True:
                i = 10
        from pyodide.ffi import InternalError
        raise InternalError('oops!')
    from pyodide.ffi import InternalError
    with pytest.raises(InternalError):
        helper(selenium)

@pytest.mark.parametrize('run_python', ['pyodide.runPython', 'await pyodide.runPythonAsync'])
def test_runpython_filename(selenium, run_python):
    if False:
        for i in range(10):
            print('nop')
    msg = selenium.run_js('\n        try {\n            %s(`\n                def f1():\n                    f2()\n\n                def f2():\n                    raise Exception("oops")\n\n                f1()\n            `, {filename: "a.py"});\n        } catch(e) {\n            return e.message\n        }\n        ' % run_python)
    expected = dedent('\n        File "a.py", line 8, in <module>\n          f1()\n        File "a.py", line 3, in f1\n          f2()\n        File "a.py", line 6, in f2\n          raise Exception("oops")\n        ').strip()
    assert dedent('\n'.join(msg.splitlines()[-7:-1])) == expected
    msg = selenium.run_js('\n        let f1;\n        try {\n            f1 = pyodide.globals.get("f1");\n            f1();\n        } catch(e) {\n            console.log(e);\n            return e.message;\n        } finally {\n            f1.destroy();\n        }\n        ')
    assert dedent('\n'.join(msg.splitlines()[1:-1])) == '\n'.join(expected.splitlines()[2:])

@pytest.mark.requires_dynamic_linking
@run_in_pyodide
def test_hiwire_invalid_ref(selenium):
    if False:
        return 10
    import pytest
    import pyodide_js
    from pyodide.code import run_js
    from pyodide.ffi import JsException
    _hiwire_get = pyodide_js._module._hiwire_get
    _hiwire_incref = pyodide_js._module._hiwire_incref
    _hiwire_decref = pyodide_js._module._hiwire_decref
    _api = pyodide_js._api
    _hiwire_incref(0)
    assert not _api.fail_test
    _hiwire_decref(0)
    assert not _api.fail_test
    expected = 'Pyodide internal error: Argument to hiwire_get is falsy \\(but error indicator is not set\\)\\.'
    with pytest.raises(JsException, match=expected):
        _hiwire_get(0)
    assert _api.fail_test
    _api.fail_test = False
    with pytest.raises(AssertionError, match='This is a message'):
        run_js('\n            const msgptr = pyodide._module.stringToNewUTF8("This is a message");\n            const AssertionError = pyodide._module.HEAP32[pyodide._module._PyExc_AssertionError/4];\n            pyodide._module._PyErr_SetString(AssertionError, msgptr);\n            pyodide._module._free(msgptr);\n            try {\n                pyodide._module._hiwire_get(0);\n            } finally {\n                pyodide._module._PyErr_Clear();\n            }\n            ')
    msg = 'hiwire_{} on invalid reference 77. This is most likely due to use after free. It may also be due to memory corruption.'
    with pytest.raises(JsException, match=msg.format('get')):
        _hiwire_get(77)
    assert _api.fail_test
    _api.fail_test = False
    with pytest.raises(JsException, match=msg.format('incref')):
        _hiwire_incref(77)
    assert _api.fail_test
    _api.fail_test = False
    with pytest.raises(JsException, match=msg.format('decref')):
        _hiwire_decref(77)
    assert _api.fail_test
    _api.fail_test = False