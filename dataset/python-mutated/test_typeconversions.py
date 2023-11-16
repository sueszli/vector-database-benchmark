import io
import pickle
from typing import Any
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import text
from pytest_pyodide import run_in_pyodide
from pytest_pyodide.fixture import selenium_context_manager
from pytest_pyodide.hypothesis import any_equal_to_self_strategy, any_strategy, std_hypothesis_settings

class NoHypothesisUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if False:
            print('Hello World!')
        if module == 'hypothesis':
            raise pickle.UnpicklingError()
        return super().find_class(module, name)

def no_hypothesis(x):
    if False:
        return 10
    try:
        NoHypothesisUnpickler(io.BytesIO(pickle.dumps(x))).load()
        return True
    except Exception:
        return False

@given(s=text())
@settings(deadline=10000)
@example('\ufeff')
def test_string_conversion(selenium_module_scope, s):
    if False:
        print('Hello World!')

    @run_in_pyodide
    def main(selenium, sbytes):
        if False:
            print('Hello World!')
        from pyodide.code import run_js
        run_js('self.encoder = new TextEncoder()')
        run_js("self.decoder = new TextDecoder('utf8', {ignoreBOM: true})")
        spy = bytes(sbytes).decode()
        sjs = run_js('\n            (sbytes) => {\n                self.sjs = self.decoder.decode(new Uint8Array(sbytes));\n                return sjs;\n            }\n            ')(sbytes)
        assert sjs == spy
        assert run_js('(spy) => spy === self.sjs')(spy)
    with selenium_context_manager(selenium_module_scope) as selenium:
        sbytes = list(s.encode())
        main(selenium, sbytes)

@given(s=text())
@std_hypothesis_settings
@example('\ufeff')
@run_in_pyodide
def test_string_conversion2(selenium, s):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    run_js('self.encoder = new TextEncoder()')
    run_js("self.decoder = new TextDecoder('utf8', {ignoreBOM: true})")
    s_encoded = s.encode()
    sjs = run_js('\n        (s_encoded) => {\n            let buf = s_encoded.getBuffer();\n            self.sjs = self.decoder.decode(buf.data);\n            buf.release();\n            return sjs\n        }\n        ')(s_encoded)
    assert sjs == s
    assert run_js('(spy) => spy === self.sjs')(s)

def blns():
    if False:
        while True:
            i = 10
    import base64
    import json
    with open('./src/tests/blns.base64.json') as f:
        BLNS = json.load(f)
    for s in BLNS:
        yield base64.b64decode(s).decode(errors='ignore')

@pytest.mark.driver_timeout(60)
def test_string_conversion_blns(selenium):
    if False:
        i = 10
        return i + 15

    @run_in_pyodide
    def _string_conversion_blns_internal(selenium, s):
        if False:
            for i in range(10):
                print('nop')
        from pyodide.code import run_js
        run_js('self.encoder = new TextEncoder()')
        run_js("self.decoder = new TextDecoder('utf8', {ignoreBOM: true})")
        s_encoded = s.encode()
        sjs = run_js('\n            (s_encoded) => {\n                let buf = s_encoded.getBuffer();\n                self.sjs = self.decoder.decode(buf.data);\n                buf.release();\n                return sjs\n            }\n            ')(s_encoded)
        assert sjs == s
        assert run_js('(spy) => spy === self.sjs')(s)
    strings = blns()
    for s in strings:
        _string_conversion_blns_internal(selenium, s)

@run_in_pyodide
def test_large_string_conversion(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    longstr = run_js('"ab".repeat(200_000)')
    res = longstr.count('ab')
    assert res == 200000
    run_js('\n        (s) => {\n            assert(() => s.length === 40_000);\n            for(let n = 0; n < 20_000; n++){\n                assert(() => s.slice(2*n, 2*n+2) === "ab");\n            }\n        }\n        ')('ab' * 20000)

@given(n=st.one_of(st.integers(), st.floats(allow_nan=False)))
@std_hypothesis_settings
@example(2 ** 53)
@example(2 ** 53 - 1)
@example(2 ** 53 + 1)
@example(-2 ** 53)
@example(-2 ** 53 - 1)
@example(-2 ** 53 + 1)
@run_in_pyodide
def test_number_conversions(selenium_module_scope, n):
    if False:
        while True:
            i = 10
    import json
    from pyodide.code import run_js
    x_js = run_js('(s) => self.x_js = eval(s)')(json.dumps(n))
    run_js('(x_py) => Number(x_py) === x_js')(n)
    if type(x_js) is float:
        assert x_js == float(n)
    else:
        assert x_js == n

@given(n=st.floats())
@std_hypothesis_settings
@run_in_pyodide
def test_number_conversions_2(selenium_module_scope, n):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    assert run_js('(n) => typeof n')(n) == 'number'
    from math import isinf, isnan
    if isnan(n):
        return
    import json
    n_js = run_js('(s) => eval(s)')(json.dumps(n))
    if not isinf(n) and float(int(n)) == n and (-2 ** 53 < n < 2 ** 53):
        assert isinstance(n_js, int)
    else:
        assert isinstance(n_js, float)

@given(n=st.integers())
@std_hypothesis_settings
@example(2 ** 53)
@example(2 ** 53 - 1)
@example(2 ** 53 + 1)
@example(-2 ** 53)
@example(-2 ** 53 - 1)
@example(-2 ** 53 + 1)
@run_in_pyodide
def test_number_conversions_3(selenium_module_scope, n):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    jsty = run_js('(n) => typeof n')(n)
    if -2 ** 53 + 1 < n < 2 ** 53 - 1:
        assert jsty == 'number'
    else:
        assert jsty == 'bigint'
    import json
    n_js = run_js('(s) => eval(s)')(json.dumps(n))
    if -2 ** 53 < n < 2 ** 53:
        assert isinstance(n_js, int)
    else:
        assert isinstance(n_js, float)

@run_in_pyodide
def test_nan_conversions(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    jsnan = run_js('NaN')
    from math import isnan
    assert isnan(jsnan)
    assert run_js('\n        let mathmod = pyodide.pyimport("math");\n        const res = Number.isNaN(mathmod.nan);\n        mathmod.destroy();\n        res\n        ')

@given(n=st.integers())
@std_hypothesis_settings
def test_bigint_conversions(selenium_module_scope, n):
    if False:
        i = 10
        return i + 15
    with selenium_context_manager(selenium_module_scope) as selenium:
        h = hex(n)
        selenium.run_js(f'self.h = {h!r};')
        selenium.run_js('\n            let negative = false;\n            let h2 = h;\n            if(h2.startsWith(\'-\')){\n                h2 = h2.slice(1);\n                negative = true;\n            }\n            self.n = BigInt(h2);\n            if(negative){\n                self.n = -n;\n            }\n            pyodide.runPython(`\n                from js import n, h\n                n2 = int(h, 16)\n                assert n == n2\n            `);\n            let n2 = pyodide.globals.get("n2");\n            let n3 = Number(n2);\n            if(Number.isSafeInteger(n3)){\n                assert(() => typeof n2 === "number");\n                assert(() => n2 === Number(n));\n            } else {\n                assert(() => typeof n2 === "bigint");\n                assert(() => n2 === n);\n            }\n            ')

@given(n=st.one_of(st.integers(min_value=2 ** 53 + 1), st.integers(max_value=-2 ** 53 - 1)))
@std_hypothesis_settings
def test_big_int_conversions2(selenium_module_scope, n):
    if False:
        for i in range(10):
            print('nop')

    @run_in_pyodide
    def main(selenium, s):
        if False:
            print('Hello World!')
        import json
        from pyodide.code import run_js
        x_py = json.loads(s)
        (x_js, check) = run_js("\n            (s, x_py) => {\n                let x_js = eval(s + 'n');\n\n                return [x_js, x_py === x_js];\n            }\n            ")(s, x_py)
        assert check
        assert x_js == x_py
    with selenium_context_manager(selenium_module_scope) as selenium:
        import json
        s = json.dumps(n)
        main(selenium, s)

@given(n=st.integers(), exp=st.integers(min_value=1, max_value=10))
@std_hypothesis_settings
def test_big_int_conversions3(selenium_module_scope, n, exp):
    if False:
        return 10

    @run_in_pyodide
    def main(selenium, s):
        if False:
            while True:
                i = 10
        import json
        from pyodide.code import run_js
        x_py = json.loads(s)
        x_js = run_js(f"\n            self.x_js = eval('{s}n'); // JSON.parse apparently doesn't work\n            ")
        [x1, x2] = run_js('\n            (x_py) => [x_py.toString(), x_js.toString()]\n            ')(x_py)
        assert x1 == x2
        check = run_js('\n            (x) => {\n                const [a, b] = x.toJs();\n                return a === b;\n            }\n            ')([str(x_js), str(x_py)])
        assert check
    with selenium_context_manager(selenium_module_scope) as selenium:
        val = 2 ** (32 * exp) - n
        import json
        s = json.dumps(val)
        main(selenium, s)

@given(obj=any_equal_to_self_strategy.filter(no_hypothesis))
@std_hypothesis_settings
@run_in_pyodide
def test_hyp_py2js2py(selenium, obj):
    if False:
        print('Hello World!')
    import __main__
    from pyodide.code import run_js
    __main__.obj = obj
    try:
        run_js('self.obj2 = pyodide.globals.get("obj"); 0;')
        from js import obj2
        assert obj2 == obj
        run_js('\n            if(self.obj2 && self.obj2.destroy){\n                self.obj2.destroy();\n            }\n            delete self.obj2\n            ')
    finally:
        del __main__.obj

@given(obj=any_equal_to_self_strategy.filter(no_hypothesis))
@std_hypothesis_settings
@run_in_pyodide
def test_hyp_py2js2py_2(selenium, obj):
    if False:
        while True:
            i = 10
    import __main__
    from pyodide.code import run_js
    __main__.o = obj
    try:
        assert obj == run_js("pyodide.globals.get('o')")
    finally:
        del __main__.o

@pytest.mark.parametrize('a', [9992361673228537, -9992361673228537])
@run_in_pyodide
def test_big_integer_py2js2py(selenium, a):
    if False:
        i = 10
        return i + 15
    import __main__
    from pyodide.code import run_js
    __main__.a = a
    try:
        b = run_js("pyodide.globals.get('a')")
        assert a == b
    finally:
        del __main__.a

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
@given(obj=any_strategy.filter(no_hypothesis))
@std_hypothesis_settings
@run_in_pyodide
def test_hyp_tojs_no_crash(selenium, obj):
    if False:
        for i in range(10):
            print('nop')
    import __main__
    from pyodide.code import run_js
    __main__.x = obj
    try:
        run_js('\n            let x = pyodide.globals.get("x");\n            if(x && x.toJs){\n                x.toJs();\n            }\n            ')
    finally:
        del __main__.x

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
@given(obj=any_strategy.filter(no_hypothesis))
@example(obj=range(0, 2147483648))
@settings(std_hypothesis_settings, max_examples=25)
@run_in_pyodide
def test_hypothesis(selenium_standalone, obj):
    if False:
        print('Hello World!')
    from pyodide.ffi import to_js
    to_js(obj)

@pytest.mark.parametrize('py,js', [(None, 'undefined'), (True, 'true'), (False, 'false'), (42, '42'), (3.14, '3.14'), ('ascii', "'ascii'"), ('ιωδιούχο', "'ιωδιούχο'"), ('碘化物', "'碘化物'"), ('🐍', "'🐍'")])
@run_in_pyodide
def test_python2js1(selenium, py, js):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    assert run_js(f'\n        (py) => py === {js}\n        ')(py)

@run_in_pyodide
def test_python2js2(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    assert list(run_js('\n                (x) => {\n                    x = x.toJs();\n                    return [x.constructor.name, x.length, x[0]];\n                }\n                ')(b'bytes')) == ['Uint8Array', 5, 98]

@run_in_pyodide
def test_python2js3(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    l = [7, 9, 13]
    result = run_js('\n        (proxy) => {\n            x = proxy.toJs();\n            return [proxy.type, x.constructor.name, x.length, x[0], x[1], x[2]]\n        }\n        ')(l)
    assert list(result) == ['list', 'Array', 3, *l]

@run_in_pyodide
def test_python2js4(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    assert list(run_js('\n                (proxy) => {\n                    let typename = proxy.type;\n                    let x = proxy.toJs();\n                    return [proxy.type, x.constructor.name, x.get(42)];\n                }\n                ')({42: 64})) == ['dict', 'Map', 64]

@run_in_pyodide
def test_python2js5(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    assert run_js('(x) => x.tell()')(open('/foo.txt', 'wb')) == 0
    from tempfile import TemporaryFile
    with TemporaryFile(mode='w+') as f:
        contents = ['a\n', 'b\n', 'hello there!\n']
        f.writelines(contents)
        assert run_js('(f) => f.tell()')(f) == 17
        assert run_js('\n                (f) => {\n                    f.seek(0);\n                    return [f.readline(), f.readline(), f.readline()];\n                }\n                ')(f).to_py() == contents

def test_python2js_track_proxies(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        let x = pyodide.runPython(`\n            class T:\n                pass\n            [[T()],[T()], [[[T()],[T()]],[T(), [], [[T()]], T()], T(), T()], T()]\n        `);\n        let proxies = [];\n        let result = x.toJs({ pyproxies : proxies });\n        assert(() => proxies.length === 10);\n        for(let x of proxies){\n            x.destroy();\n        }\n        function check(l){\n            for(let x of l){\n                if(x instanceof pyodide.ffi.PyProxy){\n                    assert(() => !pyodide._api.pyproxyIsAlive(x));\n                } else {\n                    check(x);\n                }\n            }\n        }\n        check(result);\n        assertThrows(() => x.toJs({create_pyproxies : false}), "PythonError", "pyodide.ffi.ConversionError");\n        x.destroy();\n        ')

@run_in_pyodide
def test_wrong_way_track_proxies(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    checkDestroyed = run_js('\n        function checkDestroyed(l){\n            for(let e of l){\n                if(e instanceof pyodide.ffi.PyProxy){\n                    assert(() => !pyodide._api.pyproxyIsAlive(e));\n                } else {\n                    checkDestroyed(e);\n                }\n            }\n        };\n        checkDestroyed\n        ')
    from unittest import TestCase
    from js import Array, Object
    from pyodide.ffi import ConversionError, destroy_proxies, to_js
    raises = TestCase().assertRaises

    class T:
        pass
    x = [[T()], [T()], [[[T()], [T()]], [T(), [], [[T()]], T()], T(), T()], T()]
    proxylist = Array.new()
    r = to_js(x, pyproxies=proxylist)
    assert len(proxylist) == 10
    destroy_proxies(proxylist)
    checkDestroyed(r)
    with raises(TypeError):
        to_js(x, pyproxies=[])
    with raises(TypeError):
        to_js(x, pyproxies=Object.new())
    with raises(ConversionError):
        to_js(x, create_pyproxies=False)

def test_wrong_way_conversions1(selenium):
    if False:
        return 10
    selenium.run_js('\n        assert(() => pyodide.toPy(5) === 5);\n        assert(() => pyodide.toPy(5n) === 5n);\n        assert(() => pyodide.toPy("abc") === "abc");\n        class Test {};\n        let t = new Test();\n        assert(() => pyodide.toPy(t) === t);\n\n        self.a1 = [1,2,3];\n        self.b1 = pyodide.toPy(a1);\n        self.a2 = { a : 1, b : 2, c : 3};\n        self.b2 = pyodide.toPy(a2);\n        pyodide.runPython(`\n            from js import a1, b1, a2, b2\n            assert a1.to_py() == b1\n            assert a2.to_py() == b2\n        `);\n        self.b1.destroy();\n        self.b2.destroy();\n        ')

@run_in_pyodide
def test_wrong_way_conversions2(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    [astr, bstr] = run_js('\n        (a) => {\n            b = [1,2,3];\n            return [JSON.stringify(a), JSON.stringify(b)]\n        }\n        ')(to_js([1, 2, 3]))
    assert astr == bstr

@run_in_pyodide
def test_wrong_way_conversions3(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    from pyodide.ffi import to_js

    class Test:
        pass
    t1 = Test()
    t2 = to_js(t1)
    t3 = run_js('(t2) => t2.copy()')(t2)
    assert t1 is t3
    t2.destroy()

@run_in_pyodide
def test_wrong_way_conversions4(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.ffi import to_js
    s = 'avafhjpa'
    t = 55
    assert to_js(s) is s
    assert to_js(t) is t

@run_in_pyodide
def test_dict_converter1(selenium):
    if False:
        while True:
            i = 10
    import json
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    arrayFrom = run_js('Array.from')
    d = {x: x + 2 for x in range(5)}
    res = to_js(d, dict_converter=arrayFrom)
    (constructor, serialized) = run_js('\n        (res) => {\n            return [res.constructor.name, JSON.stringify(res)];\n        }\n        ')(res)
    assert constructor == 'Array'
    assert json.loads(serialized) == [list(x) for x in d.items()]

@run_in_pyodide
def test_dict_converter2(selenium):
    if False:
        for i in range(10):
            print('nop')
    import json
    from pyodide.code import run_js
    d = {x: x + 2 for x in range(5)}
    (constructor, serialized) = run_js('\n        (d) => {\n            const res = d.toJs({dict_converter : Array.from});\n            return [res.constructor.name, JSON.stringify(res)];\n        }\n        ')(d)
    assert constructor == 'Array'
    assert json.loads(serialized) == [list(x) for x in d.items()]

@run_in_pyodide
def test_dict_converter3(selenium):
    if False:
        print('Hello World!')
    import json
    from js import Object
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    d = {x: x + 2 for x in range(5)}
    res = to_js(d, dict_converter=Object.fromEntries)
    (constructor, serialized) = run_js('\n        (res) => [res.constructor.name, JSON.stringify(res)]\n        ')(res)
    assert constructor == 'Object'
    assert json.loads(serialized) == {str(k): v for (k, v) in d.items()}

@run_in_pyodide
def test_dict_converter4(selenium):
    if False:
        while True:
            i = 10
    import json
    from pyodide.code import run_js
    d = {x: x + 2 for x in range(5)}
    (constructor, serialized) = run_js('\n        (px) => {\n            let res = px.toJs({dict_converter : Object.fromEntries});\n            return [res.constructor.name, JSON.stringify(res)];\n        }\n        ')(d)
    assert constructor == 'Object'
    assert json.loads(serialized) == {str(k): v for (k, v) in d.items()}

@pytest.mark.parametrize('formula', ['2**30', '2**31', '2**30 - 1 + 2**30', '2**32 / 2**4', '-2**30', '-2**31'])
def test_python2js_long_ints(selenium, formula):
    if False:
        for i in range(10):
            print('nop')
    assert selenium.run(formula) == eval(formula)

@run_in_pyodide
def test_python2js_long_ints2(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    assert run_js('\n        (x) => x === 2n**64n;\n        ')(2 ** 64)
    assert run_js('\n        (x) => x === -(2n**64n);\n        ')(-2 ** 64)

def test_pythonexc2js(selenium):
    if False:
        print('Hello World!')
    msg = 'ZeroDivisionError'
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run_js('return pyodide.runPython("5 / 0")')

@run_in_pyodide
def test_js2python_null(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    assert run_js('null') is None
    assert run_js('[null]')[0] is None
    assert run_js('() => null')() is None
    assert run_js('({a: null})').a is None
    assert run_js("new Map([['a', null]])")['a'] is None
    assert run_js('[null, null, null]').to_py() == [None, None, None]
    assert run_js("new Map([['a', null]])").to_py() == {'a': None}

@run_in_pyodide
def test_js2python_basic(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    t = run_js('\n        ({\n            jsstring_ucs1 : "pyodidé",\n            jsstring_ucs2 : "碘化物",\n            jsstring_ucs4 : "🐍",\n            jsnumber0 : 42,\n            jsnumber1 : 42.5,\n            jsundefined : undefined,\n            jsnull : null,\n            jstrue : true,\n            jsfalse : false,\n            jsarray0 : [],\n            jsarray1 : [1, 2, 3],\n            jspython : pyodide.globals.get("open"),\n            jsbytes : new Uint8Array([1, 2, 3]),\n            jsfloats : new Float32Array([1, 2, 3]),\n            jsobject : new TextDecoder(),\n        });\n        ')
    assert t.jsstring_ucs1 == 'pyodidé'
    assert t.jsstring_ucs2 == '碘化物'
    assert t.jsstring_ucs4 == '🐍'
    assert t.jsnumber0 == 42 and isinstance(t.jsnumber0, int)
    assert t.jsnumber1 == 42.5 and isinstance(t.jsnumber1, float)
    assert t.jsundefined is None
    assert t.jsnull is None
    assert t.jstrue is True
    assert t.jsfalse is False
    assert t.jspython is open
    jsbytes = t.jsbytes.to_py()
    assert jsbytes.tolist() == [1, 2, 3] and jsbytes.tobytes() == b'\x01\x02\x03'
    jsfloats = t.jsfloats.to_py()
    import struct
    expected = struct.pack('fff', 1, 2, 3)
    assert jsfloats.tolist() == [1, 2, 3] and jsfloats.tobytes() == expected
    assert str(t.jsobject) == '[object TextDecoder]'
    assert bool(t.jsobject) is True
    assert bool(t.jsarray0) is False
    assert bool(t.jsarray1) is True
    run_js('(t) => t.jspython.destroy()')(t)

@pytest.mark.parametrize('jsval, is_truthy', [('()=>{}', True), ('new Map()', False), ('new Map([[0, 1]])', True), ('new Set()', False), ('new Set([0])', True)])
@run_in_pyodide
def test_js2python_bool(selenium, jsval, is_truthy):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    assert bool(run_js(jsval)) is is_truthy

@pytest.mark.parametrize('jstype, pytype', (('Int8Array', 'b'), ('Uint8Array', 'B'), ('Uint8ClampedArray', 'B'), ('Int16Array', 'h'), ('Uint16Array', 'H'), ('Int32Array', 'i'), ('Uint32Array', 'I'), ('Float32Array', 'f'), ('Float64Array', 'd')))
@run_in_pyodide
def test_typed_arrays(selenium, jstype, pytype):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    array = run_js(f'new {jstype}([1, 2, 3, 4]);').to_py()
    print(array.format, array.tolist(), array.tobytes())
    assert array.format == pytype
    assert array.tolist() == [1, 2, 3, 4]
    import struct
    assert array.tobytes() == struct.pack(pytype * 4, 1, 2, 3, 4)

@run_in_pyodide
def test_array_buffer(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    array = run_js('new ArrayBuffer(100);').to_py()
    assert len(array.tobytes()) == 100

def assert_js_to_py_to_js(selenium, name):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js(f'self.obj = {name};')
    selenium.run('from js import obj')
    assert selenium.run_js('\n        let pyobj = pyodide.globals.get("obj");\n        return pyobj === obj;\n        ')

def assert_py_to_js_to_py(selenium, name):
    if False:
        return 10
    selenium.run_js(f"\n        self.obj = pyodide.runPython('{name}');\n        pyodide.runPython(`\n            from js import obj\n            assert obj is {name}\n        `);\n        obj.destroy();\n        ")

@run_in_pyodide
def test_recursive_list_to_js(selenium):
    if False:
        while True:
            i = 10
    x: Any = []
    x.append(x)
    from pyodide.ffi import to_js
    to_js(x)

@run_in_pyodide
def test_recursive_dict_to_js(selenium):
    if False:
        i = 10
        return i + 15
    x: Any = {}
    x[0] = x
    from pyodide.ffi import to_js
    to_js(x)

def test_list_js2py2js(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('self.x = [1,2,3];')
    assert_js_to_py_to_js(selenium, 'x')

def test_dict_js2py2js(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('self.x = { a : 1, b : 2, 0 : 3 };')
    assert_js_to_py_to_js(selenium, 'x')

def test_error_js2py2js(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js("self.err = new Error('hello there?');")
    assert_js_to_py_to_js(selenium, 'err')
    if selenium.browser == 'node':
        return
    selenium.run_js("self.err = new DOMException('hello there?');")
    assert_js_to_py_to_js(selenium, 'err')

def test_error_py2js2py(selenium):
    if False:
        return 10
    selenium.run("err = Exception('hello there?');")
    assert_py_to_js_to_py(selenium, 'err')

def test_list_py2js2py(selenium):
    if False:
        while True:
            i = 10
    selenium.run("x = ['a', 'b']")
    assert_py_to_js_to_py(selenium, 'x')

def test_dict_py2js2py(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run("x = {'a' : 5, 'b' : 1}")
    assert_py_to_js_to_py(selenium, 'x')

@run_in_pyodide
def test_jsproxy_attribute_error(selenium):
    if False:
        print('Hello World!')
    import pytest
    from pyodide.code import run_js
    point = run_js('\n        class Point {\n            constructor(x, y) {\n                this.x = x;\n                this.y = y;\n            }\n        }\n        new Point(42, 43);\n        ')
    assert point.y == 43
    with pytest.raises(AttributeError, match='z'):
        point.z
    del point.y
    with pytest.raises(AttributeError, match='y'):
        point.y
    assert run_js('(point) => point.y;')(point) is None

def test_javascript_error(selenium):
    if False:
        while True:
            i = 10
    msg = 'JsException: Error: This is a js error'
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run('\n            from js import Error\n            err = Error.new("This is a js error")\n            err2 = Error.new("This is another js error")\n            raise err\n            ')

@run_in_pyodide
def test_javascript_error_back_to_js(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    err = run_js('self.err = new Error("This is a js error"); err')
    assert type(err).__name__ == 'JsException'
    assert run_js('\n        (py_err) => py_err === err;\n        ')(err)

def test_memoryview_conversion(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run('\n        import array\n        a = array.array("Q", [1,2,3])\n        b = array.array("u", "123")\n        ')
    selenium.run_js('\n        pyodide.runPython("a").destroy()\n        // Implicit assertion: this doesn\'t leave python error indicator set\n        // (automatically checked in conftest.py)\n        ')
    selenium.run_js('\n        pyodide.runPython("b").destroy()\n        // Implicit assertion: this doesn\'t leave python error indicator set\n        // (automatically checked in conftest.py)\n        ')

def test_python2js_with_depth(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        const x = pyodide.runPython(`\n            class Test: pass\n            [Test(), [Test(), [Test(), [Test()]]]]\n        `);\n        const Module = pyodide._module;\n        const proxies = [];\n        const result = Module._python2js_with_depth(Module.PyProxy_getPtr(x), -1, proxies);\n        assert(() => proxies.length === 4);\n        const result_proxies = [result[0], result[1][0], result[1][1][0], result[1][1][1][0]];\n        const sortFunc = (x, y) => Module.PyProxy_getPtr(x) < Module.PyProxy_getPtr(y);\n        proxies.sort(sortFunc);\n        result_proxies.sort(sortFunc);\n        for(let i = 0; i < 4; i++){\n            assert(() => proxies[i] == result_proxies[i]);\n        }\n        x.destroy();\n        for(const px of proxies){\n            px.destroy();\n        }\n        ')

@pytest.mark.parametrize('ty', [list, tuple])
@run_in_pyodide
def test_tojs1(selenium, ty):
    if False:
        print('Hello World!')
    import json
    from pyodide.code import run_js
    l = [1, 2, 3]
    x = ty(l)
    assert run_js('x => Array.isArray(x.toJs())')(x)
    serialized = run_js('x => JSON.stringify(x.toJs())')(x)
    assert l == json.loads(serialized)

@run_in_pyodide
def test_tojs2(selenium):
    if False:
        return 10
    import json
    from pyodide.code import run_js
    o = [(1, 2), (3, 4), [5, 6], {2: 3, 4: 9}]
    assert run_js('(o) => Array.isArray(o.toJs())')(o)
    serialized = run_js('(o) => JSON.stringify(o.toJs())')(o)
    assert json.loads(serialized) == [[1, 2], [3, 4], [5, 6], {}]
    serialized = run_js('(o) => JSON.stringify(Array.from(o.toJs()[3].entries()))')(o)
    assert json.loads(serialized) == [[2, 3], [4, 9]]

def test_tojs4(selenium):
    if False:
        return 10
    selenium.run_js('\n        let a = pyodide.runPython("[1,[2,[3,[4,[5,[6,[7]]]]]]]")\n        for(let i=0; i < 7; i++){\n            let x = a.toJs({depth : i});\n            for(let j=0; j < i; j++){\n                assert(() => Array.isArray(x), `i: ${i}, j: ${j}`);\n                x = x[1];\n            }\n            assert(() => x instanceof pyodide.ffi.PyProxy, `i: ${i}, j: ${i}`);\n            x.destroy();\n        }\n        a.destroy()\n        ')

def test_tojs5(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        let a = pyodide.runPython("[1, (2, (3, [4, (5, (6, [7]))]))]")\n        for(let i=0; i < 7; i++){\n            let x = a.toJs({depth : i});\n            for(let j=0; j < i; j++){\n                assert(() => Array.isArray(x), `i: ${i}, j: ${j}`);\n                x = x[1];\n            }\n            assert(() => x instanceof pyodide.ffi.PyProxy, `i: ${i}, j: ${i}`);\n            x.destroy();\n        }\n        a.destroy()\n        ')

def test_tojs6(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        let respy = pyodide.runPython(`\n            a = [1, 2, 3, 4, 5]\n            b = [a, a, a, a, a]\n            [b, b, b, b, b]\n        `);\n        let total_refs = pyodide._module._hiwire_num_refs();\n        let res = respy.toJs();\n        let new_total_refs = pyodide._module._hiwire_num_refs();\n        respy.destroy();\n        assert(() => total_refs === new_total_refs);\n        assert(() => res[0] === res[1]);\n        assert(() => res[0][0] === res[1][1]);\n        assert(() => res[4][0] === res[1][4]);\n        ')

def test_tojs7(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        let respy = pyodide.runPython(`\n            a = [["b"]]\n            b = [1,2,3, a[0]]\n            a[0].append(b)\n            a.append(b)\n            a\n        `);\n        let total_refs = pyodide._module._hiwire_num_refs();\n        let res = respy.toJs();\n        let new_total_refs = pyodide._module._hiwire_num_refs();\n        respy.destroy();\n        assert(() => total_refs === new_total_refs);\n        assert(() => res[0][0] === "b");\n        assert(() => res[1][2] === 3);\n        assert(() => res[1][3] === res[0]);\n        assert(() => res[0][1] === res[1]);\n        ')

@pytest.mark.skip_pyproxy_check
@run_in_pyodide
def test_tojs8(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from pyodide.ffi import ConversionError, to_js
    msg = 'Cannot use \\(2, 2\\) as a key for a Javascript'
    with pytest.raises(ConversionError, match=msg):
        to_js({(2, 2): 0})
    with pytest.raises(ConversionError, match=msg):
        to_js({(2, 2)})

def test_tojs9(selenium):
    if False:
        for i in range(10):
            print('nop')
    assert set(selenium.run_js('\n                return Array.from(pyodide.runPython(`\n                    from pyodide.ffi import to_js\n                    to_js({ 1, "1" })\n                `).values())\n                ')) == {1, '1'}
    assert dict(selenium.run_js('\n                return Array.from(pyodide.runPython(`\n                    from pyodide.ffi import to_js\n                    to_js({ 1 : 7, "1" : 9 })\n                `).entries())\n                ')) == {1: 7, '1': 9}

@run_in_pyodide
def test_to_py1(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    a = run_js('\n        let a = new Map([[1, [1,2,new Set([1,2,3])]], [2, new Map([[1,2],[2,7]])]]);\n        a.get(2).set("a", a);\n        a;\n        ')
    result = [repr(a.to_py(depth=i)) for i in range(4)]
    assert result == ['[object Map]', '{1: 1,2,[object Set], 2: [object Map]}', "{1: [1, 2, [object Set]], 2: {1: 2, 2: 7, 'a': [object Map]}}", "{1: [1, 2, {1, 2, 3}], 2: {1: 2, 2: 7, 'a': {...}}}"]

@run_in_pyodide
def test_to_py2(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    a = run_js('\n        let a = { "x" : 2, "y" : 7, "z" : [1,2] };\n        a.z.push(a);\n        a\n        ')
    result = [repr(a.to_py(depth=i)) for i in range(4)]
    assert result == ['[object Object]', "{'x': 2, 'y': 7, 'z': 1,2,[object Object]}", "{'x': 2, 'y': 7, 'z': [1, 2, [object Object]]}", "{'x': 2, 'y': 7, 'z': [1, 2, {...}]}"]

@run_in_pyodide
def test_to_py3(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    a = run_js('\n        class Temp {\n            constructor(){\n                this.x = 2;\n                this.y = 7;\n            }\n        }\n        new Temp();\n        ')
    assert repr(type(a.to_py())) == "<class 'pyodide.ffi.JsProxy'>"

@pytest.mark.parametrize('obj, msg', [('Map([[[1,1], 2]])', 'Cannot use key of type Array as a key to a Python dict'), ('Set([[1,1]])', 'Cannot use key of type Array as a key to a Python set'), ('Map([[0, 2], [false, 3]])', 'contains both 0 and false'), ('Set([0, false])', 'contains both 0 and false'), ('Map([[1, 2], [true, 3]])', 'contains both 1 and true'), ('Set([1, true])', 'contains both 1 and true')])
@run_in_pyodide
def test_to_py4(selenium, obj, msg):
    if False:
        i = 10
        return i + 15
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import ConversionError, JsException
    a = run_js(f'new {obj}')
    with pytest.raises((ConversionError, JsException), match=msg):
        a.to_py()

@run_in_pyodide
def test_to_py_default_converter(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    [r1, r2] = run_js('\n        class Pair {\n            constructor(first, second){\n                this.first = first;\n                this.second = second;\n            }\n        }\n        let l = [1,2,3];\n        const r1 = new Pair(l, [l]);\n        const r2 = new Pair(l, [l]);\n        r2.first = r2;\n        const opts = {defaultConverter(value, converter, cache){\n            if(value.constructor.name !== "Pair"){\n                return value;\n            }\n            let list = pyodide.globals.get("list");\n            l = list();\n            list.destroy();\n            cache(value, l);\n            const first = converter(value.first);\n            const second = converter(value.second);\n            l.append(first);\n            l.append(second);\n            first.destroy();\n            second.destroy();\n            return l;\n        }};\n        pyodide.toPy([r1, r2], opts);\n        ')
    assert isinstance(r1, list)
    assert r1[0] is r1[1][0]
    assert r1[0] == [1, 2, 3]
    assert r2[0] is r2

@run_in_pyodide
def test_to_py_default_converter2(selenium):
    if False:
        print('Hello World!')
    from typing import Any
    from pyodide.code import run_js
    [p1, p2] = run_js('\n        class Pair {\n            constructor(first, second){\n                this.first = first;\n                this.second = second;\n            }\n        }\n        const l = [1,2,3];\n        const r1 = new Pair(l, [l]);\n        const r2 = new Pair(l, [l]);\n        r2.first = r2;\n        [r1, r2]\n        ')

    def default_converter(value, converter, cache):
        if False:
            while True:
                i = 10
        if value.constructor.name != 'Pair':
            return value
        l: list[Any] = []
        cache(value, l)
        l.append(converter(value.first))
        l.append(converter(value.second))
        return l
    r1 = p1.to_py(default_converter=default_converter)
    assert isinstance(r1, list)
    assert r1[0] is r1[1][0]
    assert r1[0] == [1, 2, 3]
    r2 = p2.to_py(default_converter=default_converter)
    assert r2[0] is r2

def test_to_js_default_converter(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        p = pyodide.runPython(`\n        class Pair:\n            def __init__(self, first, second):\n                self.first = first\n                self.second = second\n        p = Pair(1,2)\n        p\n        `);\n        let res = p.toJs({ default_converter(x, convert, cacheConversion){\n            let result = [];\n            cacheConversion(x, result);\n            result.push(convert(x.first));\n            result.push(convert(x.second));\n            return result;\n        }});\n        assert(() => res[0] === 1);\n        assert(() => res[1] === 2);\n        p.first = p;\n        let res2 = p.toJs({ default_converter(x, convert, cacheConversion){\n            let result = [];\n            cacheConversion(x, result);\n            result.push(convert(x.first));\n            result.push(convert(x.second));\n            return result;\n        }});\n        assert(() => res2[0] === res2);\n        assert(() => res2[1] === 2);\n        p.destroy();\n        ')

@run_in_pyodide
def test_to_js_default_converter2(selenium):
    if False:
        return 10
    import json
    import pytest
    from js import JSON, Array
    from pyodide.code import run_js
    from pyodide.ffi import JsException, to_js

    class Pair:
        __slots__ = ('first', 'second')

        def __init__(self, first, second):
            if False:
                while True:
                    i = 10
            self.first = first
            self.second = second
    p1 = Pair(1, 2)
    p2 = Pair(1, 2)
    p2.first = p2

    def default_converter(value, convert, cacheConversion):
        if False:
            return 10
        result = Array.new()
        cacheConversion(value, result)
        result.push(convert(value.first))
        result.push(convert(value.second))
        return result
    p1js = to_js(p1, default_converter=default_converter)
    p2js = to_js(p2, default_converter=default_converter)
    assert json.loads(JSON.stringify(p1js)) == [1, 2]
    with pytest.raises(JsException, match='TypeError'):
        JSON.stringify(p2js)
    assert run_js('(x) => x[0] === x')(p2js)
    assert run_js('(x) => x[1] === 2')(p2js)

def test_buffer_format_string(selenium):
    if False:
        for i in range(10):
            print('nop')
    errors = [['aaa', "Expected format string to have length <= 2, got 'aaa'"], ['II', 'Unrecognized alignment character I.'], ['x', "Unrecognized format character 'x'."], ['x', "Unrecognized format character 'x'."], ['e', 'Javascript has no Float16 support.']]
    for (fmt, msg) in errors:
        with pytest.raises(selenium.JavascriptException, match=msg):
            selenium.run_js(f'\n                pyodide._module.processBufferFormatString({fmt!r});\n                ')
    format_tests = [('c', 'Uint8'), ('b', 'Int8'), ('B', 'Uint8'), ('?', 'Uint8'), ('h', 'Int16'), ('H', 'Uint16'), ('i', 'Int32'), ('I', 'Uint32'), ('l', 'Int32'), ('L', 'Uint32'), ('n', 'Int32'), ('N', 'Uint32'), ('q', 'BigInt64'), ('Q', 'BigUint64'), ('f', 'Float32'), ('d', 'Float64'), ('s', 'Uint8'), ('p', 'Uint8'), ('P', 'Uint32')]

    def process_fmt_string(fmt):
        if False:
            print('Hello World!')
        return selenium.run_js(f'\n            let [array, is_big_endian] = pyodide._module.processBufferFormatString({fmt!r});\n            if(!array || typeof array.name !== "string" || !array.name.endsWith("Array")){{\n                throw new Error("Unexpected output on input {fmt}: " + array);\n            }}\n            let arrayName = array.name.slice(0, -"Array".length);\n            return [arrayName, is_big_endian];\n            ')
    for (fmt, expected_array_name) in format_tests:
        [array_name, is_big_endian] = process_fmt_string(fmt)
        assert not is_big_endian
        assert array_name == expected_array_name
    endian_tests = [('@h', 'Int16', False), ('=H', 'Uint16', False), ('<i', 'Int32', False), ('>I', 'Uint32', True), ('!l', 'Int32', True)]
    for (fmt, expected_array_name, expected_is_big_endian) in endian_tests:
        [array_name, is_big_endian] = process_fmt_string(fmt)
        assert is_big_endian == expected_is_big_endian
        assert array_name == expected_array_name

def test_dict_converter_cache1(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js("\n        let d1 = pyodide.runPython('d={0: {1: 2}}; d[1]=d[0]; d');\n        let d = d1.toJs({dict_converter: Object.fromEntries});\n        d1.destroy();\n        assert(() => d[0] === d[1]);\n        ")

@pytest.mark.xfail(reason='TODO: Fix me')
def test_dict_converter_cache2(selenium):
    if False:
        print('Hello World!')
    selenium.run_js("\n        let d1 = pyodide.runPython('d={0: {1: 2}}; d[1]=d[0]; d[2] = d; d');\n        let d = d1.toJs({dict_converter: Object.fromEntries});\n        assert(() => d[2] === d);\n        ")

@run_in_pyodide
def test_dict_and_default_converter(selenium):
    if False:
        i = 10
        return i + 15
    from js import Object
    from pyodide.ffi import to_js

    def default_converter(_obj, c, _):
        if False:
            return 10
        return c({'a': 2})

    class A:
        pass
    res = to_js(A, dict_converter=Object.fromEntries, default_converter=default_converter)
    assert res.a == 2

@pytest.mark.parametrize('n', [1 << 31, 1 << 32, 1 << 33, 1 << 63, 1 << 64, 1 << 65])
@run_in_pyodide
def test_very_large_length(selenium, n):
    if False:
        while True:
            i = 10
    from unittest import TestCase
    from pyodide.code import run_js
    raises = TestCase().assertRaises(OverflowError, msg=f'length {n} of object is larger than INT_MAX (2147483647)')
    o = run_js(f'({{length : {n}}})')
    with raises:
        len(o)
    a = run_js(f"({{[Symbol.toStringTag] : 'NodeList', length: {n}}})")
    with raises:
        a[-1]

@pytest.mark.parametrize('n', [-1, -2, -3, -100, -1 << 31, -1 << 32, -1 << 33, -1 << 63, -1 << 64, -1 << 65])
@run_in_pyodide
def test_negative_length(selenium, n):
    if False:
        return 10
    from unittest import TestCase
    from pyodide.code import run_js
    raises = TestCase().assertRaises(ValueError, msg=f'length {n} of object is negative')
    o = run_js(f'({{length : {n}}})')
    with raises:
        len(o)
    a = run_js(f"({{[Symbol.toStringTag] : 'NodeList', length: {n}}})")
    with raises:
        a[-1]

@std_hypothesis_settings
@given(l=st.lists(st.integers()), slice=st.slices(50))
@example(l=[0, 1], slice=slice(None, None, -1))
@example(l=list(range(4)), slice=slice(None, None, -2))
@example(l=list(range(10)), slice=slice(-1, 12))
@example(l=list(range(10)), slice=slice(12, -1))
@example(l=list(range(10)), slice=slice(12, -1, -1))
@example(l=list(range(10)), slice=slice(-1, 12, 2))
@example(l=list(range(10)), slice=slice(12, -1, -1))
@example(l=list(range(10)), slice=slice(12, -1, -2))
@run_in_pyodide
def test_array_slices(selenium, l, slice):
    if False:
        return 10
    expected = l[slice]
    from pyodide.ffi import JsArray, to_js
    jsl = to_js(l)
    assert isinstance(jsl, JsArray)
    result = jsl[slice]
    assert result.to_py() == expected

@std_hypothesis_settings
@given(l=st.lists(st.integers()), slice=st.slices(50))
@example(l=[0, 1], slice=slice(None, None, -1))
@example(l=list(range(4)), slice=slice(None, None, -2))
@example(l=list(range(10)), slice=slice(-1, 12))
@example(l=list(range(10)), slice=slice(12, -1))
@example(l=list(range(10)), slice=slice(12, -1, -1))
@example(l=list(range(10)), slice=slice(-1, 12, 2))
@example(l=list(range(10)), slice=slice(12, -1, -1))
@example(l=list(range(10)), slice=slice(12, -1, -2))
@run_in_pyodide
def test_array_slice_del(selenium, l, slice):
    if False:
        print('Hello World!')
    from pyodide.ffi import JsArray, to_js
    jsl = to_js(l)
    assert isinstance(jsl, JsArray)
    del l[slice]
    del jsl[slice]
    assert jsl.to_py() == l

@st.composite
def list_slice_and_value(draw):
    if False:
        return 10
    l = draw(st.lists(st.integers()))
    step_one = draw(st.booleans())
    if step_one:
        start = draw(st.integers(0, max(len(l) - 1, 0)) | st.none())
        stop = draw(st.integers(start, len(l)) | st.none())
        if draw(st.booleans()) and start is not None:
            start -= len(l)
        if draw(st.booleans()) and stop is not None:
            stop -= len(l)
        s = slice(start, stop)
        vals = draw(st.lists(st.integers()))
    else:
        s = draw(st.slices(50))
        vals_len = len(l[s])
        vals = draw(st.lists(st.integers(), min_size=vals_len, max_size=vals_len))
    return (l, s, vals)

@std_hypothesis_settings
@given(lsv=list_slice_and_value())
@example(lsv=(list(range(5)), slice(5, 2), []))
@example(lsv=(list(range(5)), slice(2, 5, -1), []))
@example(lsv=(list(range(5)), slice(5, 2), [-1, -2, -3]))
@run_in_pyodide
def test_array_slice_assign_1(selenium, lsv):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.ffi import JsArray, to_js
    [l, s, v] = lsv
    jsl = to_js(l)
    assert isinstance(jsl, JsArray)
    l[s] = v
    jsl[s] = v
    assert jsl.to_py() == l

@run_in_pyodide
def test_array_slice_assign_2(selenium):
    if False:
        print('Hello World!')
    import pytest
    from pyodide.ffi import JsArray, to_js
    l = list(range(10))
    with pytest.raises(ValueError) as exc_info_1a:
        l[0:4:2] = [1, 2, 3, 4]
    jsl = to_js(l)
    assert isinstance(jsl, JsArray)
    with pytest.raises(ValueError) as exc_info_1b:
        jsl[0:4:2] = [1, 2, 3, 4]
    l = list(range(10))
    with pytest.raises(ValueError) as exc_info_2a:
        l[0:4:2] = []
    with pytest.raises(ValueError) as exc_info_2b:
        jsl[0:4:2] = []
    with pytest.raises(TypeError) as exc_info_3a:
        l[:] = 1
    with pytest.raises(TypeError) as exc_info_3b:
        jsl[:] = 1
    assert exc_info_1a.value.args == exc_info_1b.value.args
    assert exc_info_2a.value.args == exc_info_2b.value.args
    assert exc_info_3a.value.args == exc_info_3b.value.args

@std_hypothesis_settings
@given(l1=st.lists(st.integers()), l2=st.lists(st.integers()))
@example(l1=[], l2=[])
@example(l1=[], l2=[1])
@run_in_pyodide
def test_array_extend(selenium_module_scope, l1, l2):
    if False:
        while True:
            i = 10
    from pyodide.ffi import to_js
    l1js1 = to_js(l1)
    l1js1.extend(l2)
    l1js2 = to_js(l1)
    l1js2 += l2
    l1.extend(l2)
    assert l1 == l1js1.to_py()
    assert l1 == l1js2.to_py()

@run_in_pyodide
def test_typed_array(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    a = run_js('self.a = new Uint8Array([1,2,3,4]); a')
    assert a[0] == 1
    assert a[-1] == 4
    a[-2] = 7
    assert run_js('self.a[2]') == 7
    import pytest
    with pytest.raises(TypeError, match="does ?n[o']t support item deletion"):
        del a[0]
    msg = "Slice subscripting isn't implemented for typed arrays"
    with pytest.raises(NotImplementedError, match=msg):
        a[:]
    msg = "Slice assignment isn't implemented for typed arrays"
    with pytest.raises(NotImplementedError, match=msg):
        a[:] = [-1, -2, -3, -4]
    assert not hasattr(a, 'extend')
    with pytest.raises(TypeError):
        a += [1, 2, 3]

@pytest.mark.xfail_browsers(node='No document in node')
@run_in_pyodide
def test_html_array(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    x = run_js("document.querySelectorAll('*')")
    assert run_js('(a, b) => a === b[0]')(x[0], x)
    assert run_js('(a, b) => a === Array.from(b).pop()')(x[-1], x)
    import pytest
    with pytest.raises(TypeError, match="does ?n[o']t support item assignment"):
        x[0] = 0
    with pytest.raises(TypeError, match="does ?n[o']t support item deletion"):
        del x[0]