import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pytest_pyodide import run_in_pyodide
from pytest_pyodide.hypothesis import std_hypothesis_settings

def test_jsproxy_dir(selenium):
    if False:
        return 10
    result = selenium.run_js('\n        self.a = { x : 2, y : "9" };\n        self.b = function(){};\n        let pyresult = pyodide.runPython(`\n            from js import a\n            from js import b\n            [dir(a), dir(b)]\n        `);\n        let result = pyresult.toJs();\n        pyresult.destroy();\n        return result;\n        ')
    jsproxy_items = {'__bool__', '__class__', '__defineGetter__', '__defineSetter__', '__delattr__', 'constructor', 'toString', 'typeof', 'valueOf'}
    a_items = {'x', 'y'}
    callable_items = {'__call__', 'new'}
    set0 = set(result[0])
    set1 = set(result[1])
    assert set0.issuperset(jsproxy_items)
    assert set0.isdisjoint(callable_items)
    assert set0.issuperset(a_items)
    assert set1.issuperset(jsproxy_items)
    assert set1.issuperset(callable_items)
    assert set1.isdisjoint(a_items)
    selenium.run_js('\n        self.a = [0,1,2,3,4,5,6,7,8,9];\n        a[27] = 0;\n        a[":"] = 0;\n        a["/"] = 0;\n        a.abcd = 0;\n        a.α = 0;\n\n        pyodide.runPython(`\n            from js import a\n            d = dir(a)\n            assert \'0\' not in d\n            assert \'9\' not in d\n            assert \'27\' not in d\n            assert \':\' in d\n            assert \'/\' in d\n            assert \'abcd\' in d\n            assert \'α\' in d\n        `);\n        ')

def test_jsproxy_getattr(selenium):
    if False:
        print('Hello World!')
    assert selenium.run_js('\n            self.a = { x : 2, y : "9", typeof : 7 };\n            let pyresult = pyodide.runPython(`\n                from js import a\n                [ a.x, a.y, a.typeof ]\n            `);\n            let result = pyresult.toJs();\n            pyresult.destroy();\n            return result;\n            ') == [2, '9', 'object']

@run_in_pyodide
def test_jsproxy_getattr_errors(selenium):
    if False:
        return 10
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsException
    o = run_js("({get a() { throw new Error('oops'); } })")
    with pytest.raises(AttributeError):
        o.x
    with pytest.raises(JsException):
        o.a

@pytest.mark.xfail_browsers(node='No document in node')
@run_in_pyodide
def test_jsproxy_document(selenium):
    if False:
        return 10
    from js import document
    el = document.createElement('div')
    assert el.tagName == 'DIV'
    assert bool(el)
    assert not document.body.children
    document.body.appendChild(el)
    assert document.body.children
    assert len(document.body.children) == 1
    assert document.body.children[0] == el
    assert repr(document) == '[object HTMLDocument]'
    assert len(dir(el)) >= 200
    assert 'appendChild' in dir(el)

@pytest.mark.parametrize('js,result', [('{}', True), ('{a:1}', True), ('[]', False), ('[1]', True), ('new Map()', False), ('new Map([[0, 0]])', True), ('new Set()', False), ('new Set([0])', True), ('class T {}', True), ('new (class T {})', True), ('new Uint8Array(0)', False), ('new Uint8Array(1)', True), ('new ArrayBuffer(0)', False), ('new ArrayBuffer(1)', True)])
@run_in_pyodide
def test_jsproxy_bool(selenium, js, result):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    assert bool(run_js(f'({js})')) == result

@pytest.mark.xfail_browsers(node='No document in node')
@pytest.mark.parametrize('js,result', [("document.createElement('div')", True), ("document.createElement('select')", True), ("document.createElement('p')", True), ("document.createElement('style')", True), ("document.createElement('ul')", True), ("document.createElement('ul').style", True), ("document.querySelectorAll('x')", False), ("document.querySelectorAll('body')", True), ('document.all', False)])
@run_in_pyodide
def test_jsproxy_bool_html(selenium, js, result):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    assert bool(run_js(js)) == result

@pytest.mark.xfail_browsers(node='No ImageData in node')
@run_in_pyodide
def test_jsproxy_imagedata(selenium):
    if False:
        i = 10
        return i + 15
    from js import ImageData
    assert ImageData.new(64, 64).width == 64
    assert ImageData.typeof == 'function'

def test_jsproxy_function(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('self.square = function (x) { return x*x; };')
    assert selenium.run('\n            from js import square\n            square(2)\n            ') == 4

def test_jsproxy_class(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        class Point {\n          constructor(x, y) {\n            this.x = x;\n            this.y = y;\n          }\n        }\n        self.TEST = new Point(42, 43);\n        ')
    assert selenium.run("\n            from js import TEST\n            del TEST.y\n            hasattr(TEST, 'y')\n            ") is False

@run_in_pyodide
def test_jsproxy_map(selenium):
    if False:
        for i in range(10):
            print('nop')
    import pytest
    from pyodide.code import run_js
    TEST = run_js('new Map([["x", 42], ["y", 43]])')
    assert 'y' in TEST
    del TEST['y']
    assert 'y' not in TEST
    with pytest.raises(KeyError):
        del TEST['y']
    assert TEST == TEST
    assert TEST != 'foo'
    TEST = run_js("({foo: 'bar', baz: 'bap'})")
    assert dict(TEST.object_entries()) == {'foo': 'bar', 'baz': 'bap'}

def test_jsproxy_iter(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        function makeIterator(array) {\n          let nextIndex = 0;\n          return {\n            next: function() {\n              return nextIndex < array.length ?\n                {value: array[nextIndex++], done: false} :\n                {done: true};\n            }\n          };\n        }\n        self.ITER = makeIterator([1, 2, 3]);')
    assert selenium.run('from js import ITER\nlist(ITER)') == [1, 2, 3]

def test_jsproxy_implicit_iter(selenium):
    if False:
        return 10
    selenium.run_js('\n        self.ITER = [1, 2, 3];\n        ')
    assert selenium.run('from js import ITER, Object\nlist(ITER)') == [1, 2, 3]
    assert selenium.run('from js import ITER, Object\nlist(ITER.values())') == [1, 2, 3]
    assert selenium.run('from js import ITER, Object\nlist(Object.values(ITER))') == [1, 2, 3]

def test_jsproxy_call1(selenium):
    if False:
        print('Hello World!')
    assert selenium.run_js('\n            self.f = function(){ return arguments.length; };\n            let pyresult = pyodide.runPython(\n                `\n                from js import f\n                [f(*range(n)) for n in range(10)]\n                `\n            );\n            let result = pyresult.toJs();\n            pyresult.destroy();\n            return result;\n            ') == list(range(10))

@run_in_pyodide
def test_jsproxy_call2(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    f = run_js('(function(){ return arguments.length; })')
    assert [f(*range(n)) for n in range(10)] == list(range(10))

def test_jsproxy_call_kwargs(selenium):
    if False:
        return 10
    assert selenium.run_js('\n            self.kwarg_function = ({ a = 1, b = 1 }) => {\n                return [a, b];\n            };\n            return pyodide.runPython(\n                `\n                from js import kwarg_function\n                kwarg_function(b = 2, a = 10)\n                `\n            );\n            ') == [10, 2]

@pytest.mark.xfail
def test_jsproxy_call_meth_py(selenium):
    if False:
        return 10
    assert selenium.run_js('\n        self.a = {};\n        return pyodide.runPython(\n            `\n            from js import a\n            def f(self):\n                return self\n            a.f = f\n            a.f() == a\n            `\n        );\n        ')

def test_jsproxy_call_meth_js(selenium):
    if False:
        return 10
    assert selenium.run_js('\n        self.a = {};\n        function f(){return this;}\n        a.f = f;\n        return pyodide.runPython(\n            `\n            from js import a\n            a.f() == a\n            `\n        );\n        ')

def test_jsproxy_call_meth_js_kwargs(selenium):
    if False:
        for i in range(10):
            print('nop')
    assert selenium.run_js('\n        self.a = {};\n        function f({ x = 1, y = 1 }){\n            return [this, x, y];\n        }\n        a.f = f;\n        return pyodide.runPython(\n            `\n            from js import a\n            [r0, r1, r2] = a.f(y=10, x=2)\n            r0 == a and r1 == 2 and r2 == 10\n            `\n        );\n        ')

def test_call_pyproxy_destroy_args(selenium):
    if False:
        return 10
    selenium.run_js('\n        let y;\n        pyodide.setDebug(true);\n        self.f = function(x){ y = x; }\n        pyodide.runPython(`\n            from js import f\n            f({})\n            f([])\n        `);\n        assertThrows(() => y.length, "Error",\n            "This borrowed proxy was automatically destroyed at the end of a function call.*\\n" +\n            \'The object was of type "list" and had repr "\\\\[\\\\]"\'\n        );\n        ')
    selenium.run_js('\n        let y;\n        pyodide.setDebug(false);\n        self.f = function(x){ y = x; }\n        pyodide.runPython(`\n            from js import f\n            f({})\n            f([])\n        `);\n        assertThrows(() => y.length, "Error",\n            "This borrowed proxy was automatically destroyed at the end of a function call.*\\n" +\n            \'For more information about the cause of this error, use `pyodide.setDebug.true.`\'\n        );\n        ')
    selenium.run_js('\n        let y;\n        self.f = async function(x){\n            await sleep(5);\n            y = x;\n        }\n        await pyodide.runPythonAsync(`\n            from js import f\n            await f({})\n            await f([])\n        `);\n        assertThrows(() => y.length, "Error", "This borrowed proxy was automatically destroyed");\n        ')

def test_call_pyproxy_set_global(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        self.setGlobal = function(x){\n            if(self.myGlobal instanceof pyodide.ffi.PyProxy){\n                self.myGlobal.destroy();\n            }\n            if(x instanceof pyodide.ffi.PyProxy){\n                x = x.copy();\n            }\n            self.myGlobal = x;\n        }\n        pyodide.runPython(`\n            from js import setGlobal\n            setGlobal(2)\n            setGlobal({})\n            setGlobal([])\n            setGlobal(3)\n        `);\n        ')
    selenium.run_js('\n        self.setGlobal = async function(x){\n            await sleep(5);\n            if(self.myGlobal instanceof pyodide.ffi.PyProxy){\n                self.myGlobal.destroy();\n            }\n            if(x instanceof pyodide.ffi.PyProxy){\n                x = x.copy();\n            }\n            self.myGlobal = x;\n        }\n        await pyodide.runPythonAsync(`\n            from js import setGlobal\n            await setGlobal(2)\n            await setGlobal({})\n            await setGlobal([])\n            await setGlobal(3)\n        `);\n        ')

def test_call_pyproxy_destroy_result(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        self.f = function(){\n            let dict = pyodide.globals.get("dict");\n            let result = dict();\n            dict.destroy();\n            return result;\n        }\n        pyodide.runPython(`\n            from js import f\n            import sys\n            d = f()\n            assert sys.getrefcount(d) == 2\n        `);\n        ')
    selenium.run_js('\n        self.f = async function(){\n            await sleep(5);\n            let dict = pyodide.globals.get("dict");\n            let result = dict();\n            dict.destroy();\n            return result;\n        }\n        await pyodide.runPythonAsync(`\n            from js import f\n            import sys\n            d = await f()\n        `);\n        pyodide.runPython(`\n            assert sys.getrefcount(d) == 2\n        `);\n        ')

@pytest.mark.skip_refcount_check
def test_call_pyproxy_return_arg(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        self.f = function f(x){\n            return x;\n        }\n        pyodide.runPython(`\n            from js import f\n            l = [1,2,3]\n            x = f(l)\n            assert x is l\n            import sys\n            assert sys.getrefcount(x) == 3\n        `);\n        ')
    selenium.run_js('\n        self.f = async function f(x){\n            await sleep(5);\n            return x;\n        }\n        await pyodide.runPythonAsync(`\n            from js import f\n            l = [1,2,3]\n            x = await f(l)\n            assert x is l\n        `);\n        pyodide.runPython(`\n            import sys\n            assert sys.getrefcount(x) == 3\n        `);\n        ')

@run_in_pyodide
def test_import_invocation(selenium):
    if False:
        for i in range(10):
            print('nop')
    import js

    def temp():
        if False:
            return 10
        print('okay?')
    from pyodide.ffi import create_once_callable
    js.setTimeout(create_once_callable(temp), 100)
    js.fetch('pyodide-lock.json')

@run_in_pyodide
def test_import_bind(selenium):
    if False:
        while True:
            i = 10
    from js import fetch
    fetch('pyodide-lock.json')

@run_in_pyodide
def test_nested_attribute_access(selenium):
    if False:
        return 10
    import js
    from js import self
    assert js.Float64Array.BYTES_PER_ELEMENT == 8
    assert self.Float64Array.BYTES_PER_ELEMENT == 8

def test_destroy_attribute(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        let test = pyodide.runPython(`\n            class Test:\n                a = {}\n            test = Test()\n            test\n        `);\n        pyodide.runPython(`\n            import sys\n            assert sys.getrefcount(test) == 3\n            assert sys.getrefcount(test.a) == 2\n        `);\n        test.a;\n        pyodide.runPython(`\n            assert sys.getrefcount(test) == 3\n            assert sys.getrefcount(test.a) == 3\n        `);\n        test.a.destroy();\n        pyodide.runPython(`\n            assert sys.getrefcount(test) == 3\n            assert sys.getrefcount(test.a) == 2\n        `);\n        test.a;\n        pyodide.runPython(`\n            assert sys.getrefcount(test) == 3\n            assert sys.getrefcount(test.a) == 3\n        `);\n        test.destroy();\n        pyodide.runPython(`\n            assert sys.getrefcount(test) == 2\n            assert sys.getrefcount(test.a) == 2\n        `);\n        ')

@run_in_pyodide
def test_window_isnt_super_weird_anymore(selenium):
    if False:
        i = 10
        return i + 15
    import js
    from js import Array, self
    assert self.Array != self
    assert self.Array == Array
    assert self.self.self.self == self
    assert js.self.Array == Array
    assert js.self.self.self.self == self
    assert self.self.self.self.Array == Array

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
def test_mount_object(selenium_standalone):
    if False:
        i = 10
        return i + 15
    selenium = selenium_standalone
    result = selenium.run_js('\n        function x1(){\n            return "x1";\n        }\n        function x2(){\n            return "x2";\n        }\n        function y(){\n            return "y";\n        }\n        let a = { x : x1, y, s : 3, t : 7};\n        let b = { x : x2, y, u : 3, t : 7};\n        pyodide.registerJsModule("a", a);\n        pyodide.registerJsModule("b", b);\n        let result_proxy = pyodide.runPython(`\n            from a import x\n            from b import x as x2\n            result = [x(), x2()]\n            import a\n            import b\n            result += [a.s, dir(a), dir(b)]\n            result\n        `);\n        let result = result_proxy.toJs()\n        result_proxy.destroy();\n        return result;\n        ')
    assert result[:3] == ['x1', 'x2', 3]
    assert {x for x in result[3] if len(x) == 1} == {'x', 'y', 's', 't'}
    assert {x for x in result[4] if len(x) == 1} == {'x', 'y', 'u', 't'}
    selenium.run_js('\n        pyodide.unregisterJsModule("a");\n        pyodide.unregisterJsModule("b");\n        ')
    selenium.run('\n        import sys\n        del sys.modules["a"]\n        del sys.modules["b"]\n        ')

def test_unregister_jsmodule(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        let a = new Map(Object.entries({ s : 7 }));\n        let b = new Map(Object.entries({ t : 3 }));\n        pyodide.registerJsModule("a", a);\n        pyodide.registerJsModule("a", b);\n        pyodide.unregisterJsModule("a");\n        await pyodide.runPythonAsync(`\n            from unittest import TestCase\n            raises = TestCase().assertRaises\n            with raises(ImportError):\n                import a\n        `);\n        ')

def test_unregister_jsmodule_error(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        try {\n            pyodide.unregisterJsModule("doesnotexist");\n            throw new Error("unregisterJsModule should have thrown an error.");\n        } catch(e){\n            if(!e.message.includes("Cannot unregister \'doesnotexist\': no Javascript module with that name is registered")){\n                throw e;\n            }\n        }\n        ')

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
@run_in_pyodide
def test_jsmod_import_star1(selenium):
    if False:
        return 10
    import sys
    from typing import Any
    from pyodide.code import run_js
    run_js("pyodide.registerJsModule('xx', {a: 2, b: 7, f(x){ return x + 1; }});")
    g: dict[str, Any] = {}
    exec('from xx import *', g)
    try:
        assert 'a' in g
        assert 'b' in g
        assert 'f' in g
        assert '__all__' not in g
        assert g['a'] == 2
        assert g['b'] == 7
        assert g['f'](9) == 10
    finally:
        sys.modules.pop('xx', None)
        run_js("pyodide.unregisterJsModule('xx');")

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
@run_in_pyodide
def test_jsmod_import_star2(selenium):
    if False:
        while True:
            i = 10
    import sys
    from typing import Any
    from pyodide.code import run_js
    run_js("pyodide.registerJsModule('xx', {a: 2, b: 7, f(x){ return x + 1; }, __all__ : pyodide.toPy(['a'])});")
    g: dict[str, Any] = {}
    exec('from xx import *', g)
    try:
        assert 'a' in g
        assert 'b' not in g
        assert 'f' not in g
        assert '__all__' not in g
        assert g['a'] == 2
    finally:
        sys.modules.pop('xx', None)
        run_js("pyodide.unregisterJsModule('xx');")

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
def test_nested_import(selenium_standalone):
    if False:
        while True:
            i = 10
    selenium = selenium_standalone
    assert selenium.run_js('\n            self.a = { b : { c : { d : 2 } } };\n            return pyodide.runPython("from js.a.b import c; c.d");\n            ') == 2
    selenium.run('\n        import sys\n        del sys.modules["js.a"]\n        del sys.modules["js.a.b"]\n        ')

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
def test_register_jsmodule_docs_example(selenium_standalone):
    if False:
        return 10
    selenium = selenium_standalone
    selenium.run_js('\n        let my_module = {\n        f : function(x){\n            return x*x + 1;\n        },\n        g : function(x){\n            console.log(`Calling g on argument ${x}`);\n            return x;\n        },\n        submodule : {\n            h : function(x) {\n            return x*x - 1;\n            },\n            c  : 2,\n        },\n        };\n        pyodide.registerJsModule("my_js_module", my_module);\n        ')
    selenium.run('\n        import my_js_module\n        from my_js_module.submodule import h, c\n        assert my_js_module.f(7) == 50\n        assert h(9) == 80\n        assert c == 2\n        import sys\n        del sys.modules["my_js_module"]\n        del sys.modules["my_js_module.submodule"]\n        ')

@run_in_pyodide
def test_object_entries_keys_values(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    x = run_js('({ a : 2, b : 3, c : 4 })')
    assert x.object_entries().to_py() == [['a', 2], ['b', 3], ['c', 4]]
    assert x.object_keys().to_py() == ['a', 'b', 'c']
    assert x.object_values().to_py() == [2, 3, 4]

def test_mixins_feature_presence(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        let fields = [\n            [{ [Symbol.iterator](){} }, "__iter__"],\n            [{ next(){} }, "__next__", "__iter__"],\n            [{ length : 1 }, "__len__"],\n            [{ get(){} }, "__getitem__"],\n            [{ set(){} }, "__setitem__", "__delitem__"],\n            [{ has(){} }, "__contains__"],\n            [{ then(){} }, "__await__"]\n        ];\n\n        let test_object = pyodide.runPython(`\n            from js import console\n            def test_object(obj, keys_expected):\n                for [key, expected_val] in keys_expected.object_entries():\n                    actual_val = hasattr(obj, key)\n                    if actual_val != expected_val:\n                        console.log(obj)\n                        console.log(key)\n                        console.log(actual_val)\n                        assert False\n            test_object\n        `);\n\n        for(let flags = 0; flags < (1 << fields.length); flags ++){\n            let o = {};\n            let keys_expected = {};\n            for(let [idx, [obj, ...keys]] of fields.entries()){\n                if(flags & (1<<idx)){\n                    Object.assign(o, obj);\n                }\n                for(let key of keys){\n                    keys_expected[key] = keys_expected[key] || !!(flags & (1<<idx));\n                }\n            }\n            test_object(o, keys_expected);\n        }\n        test_object.destroy();\n        ')

def test_mixins_calls(selenium):
    if False:
        print('Hello World!')
    result = selenium.run_js('\n        self.testObjects = {};\n        testObjects.iterable = { *[Symbol.iterator](){\n            yield 3; yield 5; yield 7;\n        } };\n        testObjects.iterator = testObjects.iterable[Symbol.iterator]();\n        testObjects.has_len1 = { length : 7, size : 10 };\n        testObjects.has_len2 = { length : 7 };\n        testObjects.has_get = { get(x){ return x; } };\n        testObjects.has_getset = new Map();\n        testObjects.has_has = { has(x){ return typeof(x) === "string" && x.startsWith("x") } };\n        testObjects.has_includes = { includes(x){ return typeof(x) === "string" && x.startsWith("a") } };\n        testObjects.has_has_includes = {\n            includes(x){ return typeof(x) === "string" && x.startsWith("a") },\n            has(x){ return typeof(x) === "string" && x.startsWith("x") }\n        };\n        testObjects.awaitable = { then(cb){ cb(7); } };\n\n        let pyresult = await pyodide.runPythonAsync(`\n            from js import testObjects as obj\n            result = []\n            result.append(["iterable1", list(iter(obj.iterable)), [3, 5, 7]])\n            result.append(["iterable2", [*obj.iterable], [3, 5, 7]])\n            it = obj.iterator\n            result.append(["iterator", [next(it), next(it), next(it)], [3, 5, 7]])\n            result.append(["has_len1", len(obj.has_len1), 10])\n            result.append(["has_len2", len(obj.has_len2), 7])\n            result.append(["has_get1", obj.has_get[10], 10])\n            result.append(["has_get2", obj.has_get[11], 11])\n            m = obj.has_getset\n            m[1] = 6\n            m[2] = 77\n            m[3] = 9\n            m[2] = 5\n            del m[3]\n            result.append(["has_getset", [x.to_py() for x in m.entries()], [[1, 6], [2, 5]]])\n            result.append(["has_has", [n in obj.has_has for n in ["x9", "a9"]], [True, False]])\n            result.append(["has_includes", [n in obj.has_includes for n in ["x9", "a9"]], [False, True]])\n            result.append(["has_has_includes", [n in obj.has_has_includes for n in ["x9", "a9"]], [True, False]])\n            result.append(["awaitable", await obj.awaitable, 7])\n            result\n        `);\n        let result = pyresult.toJs();\n        pyresult.destroy();\n        return result;\n        ')
    for [desc, a, b] in result:
        assert a == b, desc

def test_mixins_errors_1(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        self.a = [];\n        self.b = {\n            has(){ return false; },\n            get(){ return undefined; },\n            set(){ return false; },\n            delete(){ return false; },\n        };\n        pyodide.runPython(`\n            from unittest import TestCase\n            raises = TestCase().assertRaises\n            from js import a, b\n            with raises(IndexError):\n                a[0]\n            with raises(IndexError):\n                del a[0]\n            with raises(KeyError):\n                b[0]\n            with raises(KeyError):\n                del b[0]\n        `);\n        ')

def test_mixins_errors_2(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        self.c = {\n            next(){},\n            length : 1,\n            get(){},\n            set(){},\n            has(){},\n            then(){}\n        };\n        self.d = {\n            [Symbol.iterator](){},\n        };\n        pyodide.runPython("from js import c, d");\n        delete c.next;\n        delete c.length;\n        delete c.get;\n        delete c.set;\n        delete c.has;\n        delete c.then;\n        delete d[Symbol.iterator];\n        pyodide.runPython(`\n            from contextlib import contextmanager\n            from unittest import TestCase\n            @contextmanager\n            def raises(exc, match=None):\n                with TestCase().assertRaisesRegex(exc, match) as e:\n                    yield e\n\n            from pyodide.ffi import JsException\n            msg = "^TypeError:.* is not a function.*"\n            with raises(JsException, match=msg):\n                next(c)\n            with raises(JsException, match=msg):\n                iter(d)\n            with raises(TypeError, match="object does not have a valid length"):\n                len(c)\n            with raises(JsException, match=msg):\n                c[0]\n            with raises(JsException, match=msg):\n                c[0] = 7\n            with raises(JsException, match=msg):\n                del c[0]\n        `)\n\n        await pyodide.runPythonAsync(`\n            with raises(TypeError, match="can\'t be used in \'await\' expression"):\n                await c\n        `);\n        ')

@run_in_pyodide
def test_mixins_errors_3(selenium):
    if False:
        i = 10
        return i + 15
    from unittest import TestCase
    from pyodide.code import run_js
    raises = TestCase().assertRaises
    l = run_js('\n        const l = [0, false, NaN, undefined, null];\n        l[6] = 7;\n        l\n        ')
    with raises(IndexError):
        l[10]
    with raises(IndexError):
        l[5]
    assert len(l) == 7
    l[0]
    l[1]
    l[2]
    l[3]
    l[4]
    l[6]
    del l[1]
    with raises(IndexError):
        l[4]
    l[5]
    del l[4]
    l[3]
    l[4]

@run_in_pyodide
def test_mixins_errors_4(selenium):
    if False:
        return 10
    from unittest import TestCase
    from pyodide.code import run_js
    raises = TestCase().assertRaises
    m = run_js('\n        l = [0, false, NaN, undefined, null];\n        l[6] = 7;\n        let a = Array.from(self.l.entries());\n        a.splice(5, 1);\n        m = new Map(a);\n        m\n        ')
    with raises(KeyError):
        m[10]
    with raises(KeyError):
        m[5]
    assert len(m) == 6
    m[0]
    m[1]
    m[2]
    m[3]
    m[4]
    m[6]
    del m[1]
    with raises(KeyError):
        m[1]
    assert len(m) == 5

def test_buffer(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        self.a = new Uint32Array(Array.from({length : 10}, (_,idx) => idx));\n        pyodide.runPython(`\n            from js import a\n            b = a.to_py()\n            b[4] = 7\n            assert b[8] == 8\n            a.assign_to(b)\n            assert b[4] == 4\n            b[4] = 7\n            a.assign(b)\n            assert a[4] == 7\n        `);\n        if(a[4] !== 7){\n            throw Error();\n        }\n        ')
    selenium.run_js('\n        self.a = new Uint32Array(Array.from({length : 10}, (_,idx) => idx));\n        pyodide.runPython(`\n            import js\n            from unittest import TestCase\n            raises = TestCase().assertRaisesRegex\n            from array import array\n            from js import a\n            c = array(\'b\', range(30))\n            d = array(\'b\', range(40))\n            with raises(ValueError, "cannot assign to TypedArray"):\n                a.assign(c)\n\n            with raises(ValueError, "cannot assign from TypedArray"):\n                a.assign_to(c)\n\n            with raises(ValueError, "incompatible formats"):\n                a.assign(d)\n\n            with raises(ValueError, "incompatible formats"):\n                a.assign_to(d)\n\n            e = array(\'I\', range(10, 20))\n            a.assign(e)\n        `);\n        for(let [k, v] of a.entries()){\n            if(v !== k + 10){\n                throw new Error([v, k]);\n            }\n        }\n        ')

@run_in_pyodide
def test_buffer_to_file(selenium):
    if False:
        i = 10
        return i + 15
    from js import Uint8Array
    a = Uint8Array.new(range(10))
    from tempfile import TemporaryFile
    with TemporaryFile() as f:
        a.to_file(f)
        f.seek(0)
        assert f.read() == a.to_bytes()
        b = b'abcdef'
        f.write(b)
        f.seek(-len(b), 1)
        a.from_file(f)
        assert list(a.subarray(0, len(b)).to_bytes()) == list(b)

@run_in_pyodide
def test_buffer_into_file(selenium):
    if False:
        print('Hello World!')
    from js import Uint8Array
    a = Uint8Array.new(range(10))
    from tempfile import TemporaryFile
    with TemporaryFile() as f:
        b = a.to_bytes()
        a._into_file(f)
        f.seek(0)
        assert f.read() == b

@run_in_pyodide
def test_buffer_into_file2(selenium):
    if False:
        for i in range(10):
            print('nop')
    'Check that no copy occurred.'
    import pyodide_js
    from js import Uint8Array
    a = Uint8Array.new(range(10))
    from tempfile import TemporaryFile
    with TemporaryFile() as f:
        a._into_file(f)
        assert pyodide_js.FS.streams[f.fileno()].node.contents.buffer == a.buffer

def test_buffer_assign_back(selenium):
    if False:
        return 10
    result = selenium.run_js('\n        self.jsarray = new Uint8Array([1, 2, 3, 4, 5, 6]);\n        pyodide.runPython(`\n            from js import jsarray\n            array = jsarray.to_py()\n            array[1::2] = bytes([20, 77, 9])\n            jsarray.assign(array)\n        `);\n        return Array.from(jsarray)\n        ')
    assert result == [1, 20, 3, 77, 5, 9]

@run_in_pyodide
def test_buffer_conversions(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    (s, jsbytes) = run_js('\n        const s = "abcဴ";\n        const jsbytes = new TextEncoder().encode(s);\n        [s, jsbytes]\n        ')
    memoryview_conversion = jsbytes.to_memoryview()
    bytes_conversion = jsbytes.to_bytes()
    assert bytes_conversion.decode() == s
    assert bytes(memoryview_conversion) == bytes_conversion

@run_in_pyodide
def test_tostring_encoding(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    bytes = run_js('\n        // windows-1251 encoded "Привет, мир!" which is Russian for "Hello, world!"\n        new Uint8Array([207, 240, 232, 226, 229, 242, 44, 32, 236, 232, 240, 33]);\n        ')
    assert bytes.to_string('windows-1251') == 'Привет, мир!'

@run_in_pyodide
def test_tostring_error(selenium):
    if False:
        i = 10
        return i + 15
    from unittest import TestCase
    from pyodide.code import run_js
    raises = TestCase().assertRaises
    bytes = run_js('\n        // windows-1251 encoded "Привет, мир!" which is Russian for "Hello, world!"\n        new Uint8Array([207, 240, 232, 226, 229, 242, 44, 32, 236, 232, 240, 33]);\n        ')
    with raises(ValueError):
        bytes.to_string()

@run_in_pyodide
def test_duck_buffer_method_presence(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    bytes = run_js('new Uint8Array([207, 240, 232, 226, 229, 242, 44, 32, 236, 232, 240, 33])')
    other = run_js('{}')
    buffer_methods = {'assign', 'assign_to', 'to_string', 'to_memoryview', 'to_bytes'}
    assert buffer_methods < set(dir(bytes))
    assert not set(dir(other)).intersection(buffer_methods)

def test_memory_leaks(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        self.a = [1,2,3];\n        pyodide.runPython(`\n            from js import a\n            repr(a)\n            [*a]\n            None\n        `);\n        ')

@run_in_pyodide
def test_raise_js_error(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsException
    e = run_js("new Error('hi')")
    with pytest.raises(JsException):
        raise e

@run_in_pyodide
def test_js_id(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    [x, y, z] = run_js('let a = {}; let b = {}; [a, a, b]')
    assert x.js_id == y.js_id
    assert x is not y
    assert x.js_id != z.js_id

@run_in_pyodide
def test_object_with_null_constructor(selenium):
    if False:
        print('Hello World!')
    from unittest import TestCase
    from pyodide.code import run_js
    o = run_js('Object.create(null)')
    with TestCase().assertRaises(TypeError):
        repr(o)

@pytest.mark.parametrize('n', [1 << 31, 1 << 32, 1 << 33, 1 << 63, 1 << 64, 1 << 65])
@run_in_pyodide
def test_very_large_length(selenium, n):
    if False:
        print('Hello World!')
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

@run_in_pyodide
def test_jsarray_reversed(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    l = [5, 7, 9, -1, 3, 5]
    a = run_js(repr(l))
    b = run_js(f'new Int8Array({repr(l)})')
    it1 = reversed(l)
    it2 = reversed(a)
    it3 = reversed(b)
    for _ in range(len(l)):
        v = next(it1)
        assert next(it2) == v
        assert next(it3) == v
    import pytest
    with pytest.raises(StopIteration):
        next(it1)
    with pytest.raises(StopIteration):
        next(it2)
    with pytest.raises(StopIteration):
        next(it3)

@run_in_pyodide
def test_jsarray_reverse(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    l = [5, 7, 9, 0, 3, 1]
    a = run_js(repr(l))
    b = run_js(f'new Int8Array({repr(l)})')
    l.reverse()
    a.reverse()
    b.reverse()
    assert a.to_py() == l
    assert b.to_bytes() == bytes(l)

@run_in_pyodide
def test_array_empty_slot(selenium):
    if False:
        return 10
    import pytest
    from pyodide.code import run_js
    a = run_js('[1,,2]')
    with pytest.raises(IndexError):
        a[1]
    assert a.to_py() == [1, None, 2]
    del a[1]
    assert a.to_py() == [1, 2]

@run_in_pyodide
def test_array_pop(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    a = run_js('[1, 2, 3]')
    assert a.pop() == 3
    assert a.pop(0) == 1

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
        for i in range(10):
            print('nop')
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
        while True:
            i = 10
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
        while True:
            i = 10
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
        return 10
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
        i = 10
        return i + 15
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
        return 10
    from pyodide.code import run_js
    a = run_js('self.a = new Uint8Array([1,2,3,4]); a')
    assert a[0] == 1
    assert a[-1] == 4
    a[-2] = 7
    assert run_js('self.a[2]') == 7
    import pytest
    with pytest.raises(TypeError, match="object doesn't support item deletion"):
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

@pytest.mark.parametrize('sequence_converter', ['(x) => x', '(x) => new Uint8Array(x)', "(x) => Object.create({[Symbol.toStringTag] : 'NodeList'}, Object.getOwnPropertyDescriptors(x))"])
@pytest.mark.requires_dynamic_linking
@run_in_pyodide
def test_array_sequence_methods(selenium, sequence_converter):
    if False:
        print('Hello World!')
    from pytest import raises
    from js import ArrayBuffer
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    x = to_js([77, 65, 23])
    l = run_js(sequence_converter)(x)
    from ctypes import c_bool, c_ssize_t, py_object, pythonapi
    pythonapi.PySequence_Check.argtypes = [py_object]
    pythonapi.PySequence_Check.restype = c_bool
    pythonapi.PySequence_Length.argtypes = [py_object]
    pythonapi.PySequence_GetItem.argtypes = [py_object, c_ssize_t]
    pythonapi.PySequence_GetItem.restype = py_object
    pythonapi.PySequence_SetItem.argtypes = [py_object, c_ssize_t, py_object]
    pythonapi.PySequence_DelItem.argtypes = [py_object, c_ssize_t]
    assert pythonapi.PySequence_Check(l)
    assert pythonapi.PySequence_Length(l) == 3
    assert pythonapi.PySequence_GetItem(l, 0) == 77
    node_list = 'NodeList' in str(l)
    typed_array = ArrayBuffer.isView(l)
    is_mutable = not node_list
    supports_del = not (node_list or typed_array)
    if typed_array:
        with raises(TypeError, match='unsupported operand type\\(s\\) for \\+'):
            l + [4, 5, 6]
    else:
        assert (l + [4, 5, 6]).to_py() == [77, 65, 23, 4, 5, 6]
    if is_mutable:
        pythonapi.PySequence_SetItem(l, 1, 29)
        assert l[1] == 29
        l[1] = 65
    else:
        with raises(TypeError, match="does ?n[o']t support item assignment"):
            pythonapi.PySequence_SetItem(l, 1, 29)
        assert l[1] == 65
    if supports_del:
        pythonapi.PySequence_DelItem(l, 1)
        assert l.to_py() == [77, 23]
    else:
        with raises(TypeError, match="does ?n[o']t support item deletion"):
            pythonapi.PySequence_DelItem(l, 1)
        assert list(l) == [77, 65, 23]

@run_in_pyodide
def test_array_sequence_repeat(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.ffi import JsArray, to_js
    a = [77, 65, 23]
    l: JsArray[int] = to_js(a)
    assert (l * 0).to_py() == a * 0
    assert (l * 1).to_py() == a * 1
    assert (l * 2).to_py() == a * 2
    l *= 0
    assert list(l) == a * 0
    l = to_js(a)
    l *= 1
    assert list(l) == a * 1
    l = to_js(a)
    l *= 2
    assert list(l) == a * 2

@run_in_pyodide
def test_jsproxy_match(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    x: int
    y: int
    l: list[int]
    a = run_js('[1, 2, 3]')
    match a:
        case [x, y, 3]:
            pass
    assert x == 1
    assert y == 2
    b = run_js('new Uint8Array([7, 3, 9, 10])')
    match b:
        case [x, y, *l]:
            pass
    assert x == 7
    assert y == 3
    assert l == [9, 10]
    c = run_js('new Map([[1,2], [3,4]])')
    match c:
        case {1: x, 3: y}:
            pass
    assert x == 2
    assert y == 4
    c = run_js('({a: 2, b: 5})').as_object_map()
    match c:
        case {'a': x, 'b': y}:
            pass
    assert x == 2
    assert y == 5
    default = False
    match c:
        case {'a': x, 'b': y, 'd': _}:
            pass
        case _:
            default = True
    assert default

@run_in_pyodide
def test_jsarray_index(selenium):
    if False:
        while True:
            i = 10
    import pytest
    from pyodide.code import run_js
    a = run_js('[5, 7, 9, -1, 3, 5]')
    assert a.index(5) == 0
    assert a.index(5, 1) == 5
    with pytest.raises(ValueError, match='5 is not in list'):
        assert a.index(5, 1, -1) == 5
    a.append([1, 2, 3])
    assert a.index([1, 2, 3]) == 6
    run_js('(a) => a.pop().destroy()')(a)

@run_in_pyodide
def test_jsarray_count(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    l = [5, 7, 9, -1, 3, 5]
    a = run_js(repr(l))
    assert a.count(1) == 0
    assert a.count(-1) == 1
    assert a.count(5) == 2
    b = run_js(f'new Int8Array({repr(l)})')
    assert b.count(1) == 0
    assert b.count(-1) == 1
    assert b.count(5) == 2
    a.append([])
    a.append([1])
    a.append([])
    assert a.count([]) == 2
    assert a.count([1]) == 1
    assert a.count([2]) == 0
    run_js('(a) => {\n            a.pop().destroy();\n            a.pop().destroy();\n            a.pop().destroy();\n        }\n        ')(a)

@run_in_pyodide
def test_jsproxy_descr_get(selenium):
    if False:
        return 10
    from pyodide.code import run_js

    class T:
        a: int
        b: int
        f = run_js('function f(x) {return this[x]; }; f')
    t = T()
    t.a = 7
    t.b = 66
    assert t.f('a') == 7
    assert t.f('b') == 66
    assert t.f('c') is None

@run_in_pyodide
def test_mappings(selenium):
    if False:
        while True:
            i = 10
    import pytest
    from pyodide.code import run_js
    m = run_js('new Map([[1,2], [3,4]])')
    assert set(m) == {1, 3}
    assert 1 in m.keys()
    assert m.keys() | {2} == {1, 2, 3}
    assert 2 in m.values()
    assert set(m.values()) == {2, 4}
    assert (1, 2) in m.items()
    assert set(m.items()) == {(1, 2), (3, 4)}
    assert m.get(1, 7) == 2
    assert m.get(2, 7) == 7
    assert m.pop(1) == 2
    assert m.pop(1, 7) == 7
    m[1] = 2
    assert m.pop(1, 7) == 2
    assert m.pop(1, 7) == 7
    assert 1 not in m
    with pytest.raises(KeyError):
        m.pop(1)
    assert m.setdefault(1, 8) == 8
    assert m.setdefault(3, 8) == 4
    assert m.setdefault(3) == 4
    assert m.setdefault(4) is None
    assert 1 in m
    assert m[1] == 8
    m.update({6: 7, 8: 9})
    assert dict(m) == {1: 8, 3: 4, 4: None, 6: 7, 8: 9}
    assert m.popitem() in set({1: 8, 3: 4, 4: None, 6: 7, 8: 9}.items())
    assert len(m) == 4
    m.clear()
    assert dict(m) == {}

@run_in_pyodide
def test_jsproxy_as_object_map(selenium):
    if False:
        for i in range(10):
            print('nop')
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsMutableMap
    o1 = run_js('({a : 2, b: 3, c: 77, 1 : 9})')
    with pytest.raises(TypeError, match='object is not subscriptable'):
        o1['a']
    o = o1.as_object_map()
    assert not isinstance(o1, JsMutableMap)
    assert isinstance(o, JsMutableMap)
    del o1
    assert len(o) == 4
    assert set(o) == {'a', 'b', 'c', '1'}
    assert 'a' in o
    assert 'b' in o
    assert '1' in o
    assert 1 not in o
    assert o['a'] == 2
    assert o['1'] == 9
    del o['a']
    assert 'a' not in o
    assert not hasattr(o, 'a')
    assert hasattr(o, 'b')
    assert len(o) == 3
    assert set(o) == {'b', 'c', '1'}
    o['d'] = 36
    assert len(o) == 4
    with pytest.raises(TypeError, match='Can only assign keys of type string to JavaScript object map'):
        o[1] = 2
    assert len(o) == 4
    assert set(o) == {'b', 'c', 'd', '1'}
    assert o['d'] == 36
    assert 'constructor' not in o
    assert o.to_py() == {'b': 3, 'c': 77, 'd': 36, '1': 9}
    with pytest.raises(KeyError):
        del o[1]

@run_in_pyodide
def test_object_map_mapping_methods(selenium):
    if False:
        print('Hello World!')
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsMap, JsMutableMap
    m = run_js('({1:2, 3:4})').as_object_map()
    assert isinstance(m, JsMap)
    assert isinstance(m, JsMutableMap)
    assert set(m) == {'1', '3'}
    assert '1' in m.keys()
    assert 1 not in m.keys()
    assert m.keys() | {'2'} == {'1', '2', '3'}
    assert 2 in m.values()
    assert set(m.values()) == {2, 4}
    assert ('1', 2) in m.items()
    assert set(m.items()) == {('1', 2), ('3', 4)}
    assert m.get('1', 7) == 2
    assert m.get('2', 7) == 7
    assert m.pop('1') == 2
    assert m.pop('1', 7) == 7
    m['1'] = 2
    assert m.pop('1', 7) == 2
    assert m.pop('1', 7) == 7
    assert '1' not in m
    with pytest.raises(KeyError):
        m.pop('1')
    assert m.setdefault('1', 8) == 8
    assert m.setdefault('3', 8) == 4
    assert m.setdefault('3') == 4
    assert m.setdefault('4') is None
    assert '1' in m
    assert m['1'] == 8
    m.update({'6': 7, '8': 9})
    assert dict(m) == {'1': 8, '3': 4, '4': None, '6': 7, '8': 9}
    assert m.popitem() in set({'1': 8, '3': 4, '4': None, '6': 7, '8': 9}.items())
    assert len(m) == 4
    m.clear()
    assert dict(m) == {}

@run_in_pyodide
def test_as_object_map_heritable(selenium):
    if False:
        return 10
    import pytest
    from pyodide.code import run_js
    o = run_js('({1:{2: 9, 3: 77}, 3:{6: 5, 12: 3, 2: 9}})')
    mh = o.as_object_map(hereditary=True)
    mn = o.as_object_map(hereditary=False)
    assert mh['1']['3'] == 77
    with pytest.raises(TypeError):
        mn['1']['3']
    for x in mh.values():
        assert x['2'] == 9
    for x in mn.values():
        with pytest.raises(TypeError):
            x['2']
    n = mh.pop('1')
    assert n['3'] == 77

@run_in_pyodide
def test_jsproxy_subtypes(selenium):
    if False:
        print('Hello World!')
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsArray, JsBuffer, JsPromise, JsProxy
    with pytest.raises(TypeError, match='JsProxy'):
        JsProxy()
    with pytest.raises(TypeError, match='JsArray'):
        JsArray()
    nullobj = run_js('Object.create(null)')
    a = run_js('[Promise.resolve()]')
    assert isinstance(a, JsProxy)
    assert isinstance(a, JsArray)
    assert not isinstance(a, JsPromise)
    assert not isinstance(a, JsBuffer)
    assert issubclass(type(a), JsProxy)
    assert issubclass(type(a), JsArray)
    assert not issubclass(JsArray, type(a))
    assert isinstance(a[0], JsPromise)
    assert issubclass(JsPromise, type(a[0]))
    assert not isinstance(a, JsBuffer)
    assert issubclass(type(a), type(nullobj))
    assert issubclass(type(a[0]), type(nullobj))
    assert issubclass(JsProxy, type(nullobj))
    assert issubclass(type(nullobj), JsProxy)

@run_in_pyodide
def test_gen_send(selenium):
    if False:
        for i in range(10):
            print('nop')
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsAsyncGenerator, JsAsyncIterator, JsGenerator, JsIterator
    f = run_js('\n        (function*(){\n            let n = 0;\n            for(let i = 0; i < 3; i++){\n                n = yield n + 2;\n            }\n        });\n        ')
    it = f()
    assert isinstance(it, JsGenerator)
    assert not isinstance(it, JsAsyncGenerator)
    assert isinstance(it, JsIterator)
    assert not isinstance(it, JsAsyncIterator)
    assert it.send(None) == 2
    assert it.send(2) == 4
    assert it.send(3) == 5
    with pytest.raises(StopIteration):
        it.send(4)

@run_in_pyodide
def test_gen_send_type_errors(selenium):
    if False:
        print('Hello World!')
    from re import escape
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsAsyncIterator, JsGenerator, JsIterator
    g = run_js('\n        ({next(){ return 2; }});\n        ')
    assert isinstance(g, JsIterator)
    assert isinstance(g, JsAsyncIterator)
    assert not isinstance(g, JsGenerator)
    with pytest.raises(TypeError, match='Result should have type "object" not "number"'):
        g.send(None)
    g = run_js('\n        ({next(){ return Promise.resolve(2); }});\n        ')
    with pytest.raises(TypeError, match=escape('Result was a promise, use anext() / asend() / athrow() instead.')):
        g.send(None)
    g = run_js('\n        ({next(){ return {}; }});\n        ')
    with pytest.raises(TypeError, match='Result has no "done" field.'):
        g.send(None)

@run_in_pyodide
def test_gen_throw(selenium):
    if False:
        print('Hello World!')
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsGenerator
    f = run_js('\n        (function *() {\n            try {\n                yield 1;\n            } finally {\n                yield 2;\n                console.log("finally");\n            }\n        })\n        ')
    g = f()
    assert isinstance(g, JsGenerator)
    assert next(g) == 1
    assert g.throw(TypeError('hi')) == 2
    with pytest.raises(TypeError, match='hi'):
        next(g)
    g = f()
    assert next(g) == 1
    assert g.throw(TypeError, 'hi') == 2
    with pytest.raises(TypeError, match='hi'):
        next(g)
    f = run_js('\n        (function *() {\n            yield 1;\n            yield 2;\n            yield 3;\n        })\n        ')
    g = f()
    assert next(g) == 1
    g.close()

@run_in_pyodide
def test_gen_close(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsGenerator
    f = run_js('\n        (function *(x) {\n            try {\n                yield 1;\n                yield 2;\n                x.push("this never happens");\n                yield 3;\n            } finally {\n                x.append("finally");\n            }\n        })\n        ')
    from pyodide.ffi import create_proxy
    l: list[str] = []
    p = create_proxy(l)
    g = f(p)
    assert isinstance(g, JsGenerator)
    assert next(g) == 1
    assert next(g) == 2
    assert g.close() is None
    p.destroy()
    assert l == ['finally']
    f = run_js('\n        (function *(x) {\n            try {\n                yield 1;\n            } finally {\n                yield 2;\n            }\n        })\n        ')
    g = f()
    next(g)
    with pytest.raises(RuntimeError, match='JavaScript generator ignored return'):
        g.close()

@run_in_pyodide
async def test_agen_aiter(selenium):
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsAsyncGenerator, JsAsyncIterator, JsGenerator, JsIterator
    f = run_js('\n        (async function *(){\n            yield 2;\n            yield 3;\n            return 7;\n        })\n        ')
    b = f()
    assert isinstance(b, JsAsyncIterator)
    assert not isinstance(b, JsIterator)
    assert isinstance(b, JsAsyncGenerator)
    assert not isinstance(b, JsGenerator)
    assert await anext(b) == 2
    assert await anext(b) == 3
    with pytest.raises(StopAsyncIteration):
        await anext(b)
    g = run_js('\n        (function *(){\n            yield 2;\n            yield 3;\n            return 7;\n        })()\n        ')
    assert not isinstance(g, JsAsyncIterator)
    assert isinstance(g, JsIterator)
    assert not isinstance(g, JsAsyncGenerator)
    assert isinstance(g, JsGenerator)

@run_in_pyodide
async def test_agen_aiter2(selenium):
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsAsyncIterable, JsAsyncIterator, JsIterable, JsIterator
    iterable = run_js('\n        ({\n        [Symbol.asyncIterator]() {\n            return (async function *f(){yield 1; yield 2; yield 3;})();\n        }\n        })\n        ')
    assert not isinstance(iterable, JsIterable)
    assert isinstance(iterable, JsAsyncIterable)
    with pytest.raises(TypeError, match='object is not iterable'):
        iter(iterable)
    it = aiter(iterable)
    assert isinstance(it, JsAsyncIterator)
    assert not isinstance(it, JsIterator)
    assert await anext(it) == 1
    assert await anext(it) == 2
    assert await anext(it) == 3
    with pytest.raises(StopAsyncIteration):
        await anext(it)

@run_in_pyodide
async def test_agen_asend(selenium):
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsAsyncGenerator, JsIterator
    it = run_js('\n        (async function*(){\n            let n = 0;\n            for(let i = 0; i < 3; i++){\n                n = yield n + 2;\n            }\n        })();\n        ')
    assert isinstance(it, JsAsyncGenerator)
    assert not isinstance(it, JsIterator)
    assert await it.asend(None) == 2
    assert await it.asend(2) == 4
    assert await it.asend(3) == 5
    with pytest.raises(StopAsyncIteration):
        await it.asend(4)

@run_in_pyodide
async def test_agen_athrow(selenium):
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsAsyncGenerator, JsException
    f = run_js('\n        (async function *() {\n            try {\n                yield 1;\n            } finally {\n                yield 2;\n                console.log("finally");\n            }\n        })\n        ')
    g = f()
    assert isinstance(g, JsAsyncGenerator)
    assert await anext(g) == 1
    assert await g.athrow(TypeError('hi')) == 2
    with pytest.raises(JsException, match='hi'):
        await anext(g)
    g = f()
    assert isinstance(g, JsAsyncGenerator)
    assert await anext(g) == 1
    assert await g.athrow(TypeError, 'hi') == 2
    with pytest.raises(JsException, match='hi'):
        await anext(g)
    f = run_js('\n        (async function *() {\n            yield 1;\n            yield 2;\n            yield 3;\n        })\n        ')
    g = f()
    assert isinstance(g, JsAsyncGenerator)
    assert await anext(g) == 1
    await g.aclose()

@run_in_pyodide
async def test_agen_aclose(selenium):
    from pyodide.code import run_js
    from pyodide.ffi import JsAsyncGenerator
    f = run_js('\n        (async function *(x) {\n            try {\n                yield 1;\n                yield 2;\n                x.push("this never happens");\n                yield 3;\n            } finally {\n                x.append("finally");\n            }\n        })\n        ')
    from pyodide.ffi import create_proxy
    l: list[str] = []
    p = create_proxy(l)
    g = f(p)
    assert isinstance(g, JsAsyncGenerator)
    assert await anext(g) == 1
    assert await anext(g) == 2
    assert await g.aclose() is None
    assert await g.aclose() is None
    p.destroy()
    assert l == ['finally']

@run_in_pyodide
def test_gen_lifetimes(selenium):
    if False:
        for i in range(10):
            print('nop')
    import sys
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsGenerator
    f = run_js('\n        (function *(x) {\n            let l = [x];\n            l.push(yield);\n            l.push(yield);\n            l.push(yield);\n            return pyodide.toPy(l.map((x) => x.toString()));\n        })\n        ')
    g = f({1})
    assert isinstance(g, JsGenerator)
    g.send(None)
    g.send({2})
    g.send({3})
    with pytest.raises(StopIteration) as exc_info:
        g.send({4})
    v = exc_info.value.value
    del exc_info
    assert v == ['{1}', '{2}', '{3}', '{4}']
    assert sys.getrefcount(v) == 2

@run_in_pyodide
async def test_agen_lifetimes(selenium):
    import sys
    from asyncio import sleep
    from pyodide.code import run_js
    from pyodide.ffi import JsAsyncGenerator
    f = run_js('\n        (async function *(x) {\n            let l = [x];\n            l.push(yield);\n            l.push(yield);\n            l.push(yield);\n            return pyodide.toPy(l.map((x) => x.toString()));\n        })\n        ')
    g = f({1})
    assert isinstance(g, JsAsyncGenerator)
    await g.asend(None)
    await g.asend({2})
    await g.asend({3})
    res = g.asend({4})
    await sleep(0.01)
    v = res.exception().args[0]
    del res
    assert v == ['{1}', '{2}', '{3}', '{4}']
    assert sys.getrefcount(v) == 2

@run_in_pyodide
def test_python_reserved_keywords(selenium):
    if False:
        return 10
    import pytest
    from pyodide.code import run_js
    o = run_js('({\n            async: 1,\n            await: 2,\n            False: 3,\n            nonlocal: 4,\n            yield: 5,\n            try: 6,\n            assert: 7,\n            match: 222,\n        })\n        ')
    assert o.match == 222
    with pytest.raises(AttributeError):
        o.match_
    assert eval('o.match') == 222
    keys = ['async', 'await', 'False', 'nonlocal', 'yield', 'try', 'assert']
    for k in keys:
        with pytest.raises(SyntaxError):
            eval(f'o.{k}')
    assert o.async_ == 1
    assert o.await_ == 2
    assert o.False_ == 3
    assert o.nonlocal_ == 4
    assert o.yield_ == 5
    assert o.try_ == 6
    assert o.assert_ == 7
    expected_set = {k + '_' for k in keys} | {'match'}
    actual_set = set(dir(o)) & expected_set
    assert actual_set == expected_set
    assert set(dir(o)) & set(keys) == set()
    o.async_ = 2
    assert run_js('(o) => o.async')(o) == 2
    del o.async_
    assert run_js('(o) => o.async')(o) is None
    o = run_js('({async: 1, async_: 2, async__: 3})')
    expected_set = {'async_', 'async__', 'async___'}
    actual_set = set(dir(o)) & expected_set
    assert actual_set == expected_set
    assert o.async_ == 1
    assert o.async__ == 2
    assert o.async___ == 3
    assert getattr(o, 'async_') == 1
    assert getattr(o, 'async__') == 2
    assert getattr(o, 'async') == 1
    assert hasattr(o, 'async_')
    assert hasattr(o, 'async')
    setattr(o, 'async', 2)
    assert o.async_ == 2
    delattr(o, 'async')
    assert not hasattr(o, 'async_')
    assert not hasattr(o, 'async')

@run_in_pyodide
def test_revoked_proxy(selenium):
    if False:
        for i in range(10):
            print('nop')
    'I think this is just about the worst thing that it is possible to\n    make.\n\n    A good stress test for our systems...\n    '
    from pyodide.code import run_js
    x = run_js('(p = Proxy.revocable({}, {})); p.revoke(); p.proxy')
    run_js('((x) => x)')(x)

@run_in_pyodide
def test_js_proxy_attribute(selenium):
    if False:
        return 10
    import pytest
    from pyodide.code import run_js
    x = run_js('\n        new Proxy(\n            {},\n            {\n                get(target, val) {\n                    return { a: 3, b: 7, c: undefined, d: undefined }[val];\n                },\n                has(target, val) {\n                    return { a: true, b: false, c: true, d: false }[val];\n                },\n            }\n        );\n        ')
    assert x.a == 3
    assert x.b == 7
    assert x.c is None
    with pytest.raises(AttributeError):
        x.d