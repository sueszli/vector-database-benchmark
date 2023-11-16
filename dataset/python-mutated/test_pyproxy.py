import time
import pytest
from pytest_pyodide.decorator import run_in_pyodide

def test_pyproxy_class(selenium):
    if False:
        return 10
    selenium.run_js('\n        pyodide.runPython(`\n            class Foo:\n                bar = 42\n                def get_value(self, value):\n                    return value * 64\n            f = Foo()\n        `);\n        self.f = pyodide.globals.get(\'f\');\n        assert(() => f.type === "Foo");\n        let f_get_value = f.get_value\n        assert(() => f_get_value(2) === 128);\n        f_get_value.destroy();\n        assert(() => f.bar === 42);\n        assert(() => \'bar\' in f);\n        f.baz = 32;\n        assert(() => f.baz === 32);\n        pyodide.runPython(`assert hasattr(f, \'baz\')`)\n        self.f_props = Object.getOwnPropertyNames(f);\n        delete f.baz\n        pyodide.runPython(`assert not hasattr(f, \'baz\')`)\n        assert(() => f.toString().startsWith("<__main__.Foo"));\n        f.destroy();\n        ')
    assert {'__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'bar', 'baz', 'get_value'}.difference(selenium.run_js('return f_props')) == set()

@run_in_pyodide
def test_pyproxy_tostring(selenium):
    if False:
        print('Hello World!')
    from pathlib import Path
    from pyodide.code import run_js
    from pyodide_js._api import setPyProxyToStringMethod
    pyproxy_to_string = run_js('(e) => e.toString()')
    p = Path('a/b/c')
    assert pyproxy_to_string(p) == str(p)
    setPyProxyToStringMethod(True)
    assert pyproxy_to_string(p) == repr(p)
    setPyProxyToStringMethod(False)
    assert pyproxy_to_string(p) == str(p)

def test_del_builtin(selenium):
    if False:
        i = 10
        return i + 15
    msg = 'NameError'
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run('del open')
    assert selenium.run_js('\n        let open = pyodide.globals.get("open");\n        let result = !!open;\n        open.destroy();\n        return result;\n        ')
    assert selenium.run_js("return pyodide.globals.get('__name__');") == '__main__'

def test_in_globals(selenium):
    if False:
        return 10
    selenium.run('yyyyy = 7')
    assert selenium.run_js('\n            let result = [];\n            result.push(pyodide.globals.has("xxxxx"));\n            result.push(pyodide.globals.has("yyyyy"));\n            result.push(pyodide.globals.has("globals"));\n            result.push(pyodide.globals.has("open"));\n            return result;\n            ') == [False, True, True, True]

def test_pyproxy_copy(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        let d = pyodide.runPython("list(range(10))")\n        e = d.copy();\n        d.destroy();\n        assert(() => e.length === 10);\n        e.destroy();\n        ')

def test_pyproxy_refcount(selenium):
    if False:
        return 10
    selenium.run_js('\n        function getRefCount(){\n            return pyodide.runPython("sys.getrefcount(pyfunc)");\n        }\n        self.jsfunc = function (f) { f(); };\n        pyodide.runPython(`\n            import sys\n            from js import jsfunc\n\n            def pyfunc(*args, **kwargs):\n                print(*args, **kwargs)\n        `);\n\n        // the refcount should be 2 because:\n        // 1. pyfunc exists\n        // 2. pyfunc is referenced from the sys.getrefcount()-test below\n        //\n        // Each time jsfunc is called a new PyProxy to pyfunc is created. That\n        // PyProxy is destroyed when the call finishes, so the calls to\n        // jsfunc(pyfunc) do not change the reference count.\n\n        assert(() => getRefCount() === 2);\n\n        pyodide.runPython(`\n            jsfunc(pyfunc)\n        `);\n\n        assert(() => getRefCount() === 2);\n\n        pyodide.runPython(`\n            jsfunc(pyfunc)\n            jsfunc(pyfunc)\n        `)\n        assert(() => getRefCount() === 2);\n        pyodide.runPython(`del jsfunc`)\n        ')

def test_pyproxy_destroy(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        pyodide.runPython(`\n            class Foo:\n                def get_value(self, value):\n                    return value * 64\n            f = Foo()\n        `);\n        let f = pyodide.globals.get(\'f\');\n        assert(()=> f.get_value(1) === 64);\n        f.destroy();\n        assertThrows(() => f.get_value(1), "Error", "already been destroyed");\n        ')

def test_pyproxy_iter(selenium):
    if False:
        print('Hello World!')
    [ty, l] = selenium.run_js('\n        let c = pyodide.runPython(`\n            def test():\n                for i in range(10):\n                    yield i\n            test()\n        `);\n        let result = [c.type, [...c]];\n        c.destroy();\n        return result;\n        ')
    assert ty == 'generator'
    assert l == list(range(10))
    [ty, l] = selenium.run_js('\n        let c = pyodide.runPython(`\n            from collections import ChainMap\n            ChainMap({"a" : 2, "b" : 3})\n        `);\n        let result = [c.type, [...c]];\n        c.destroy();\n        return result;\n        ')
    assert ty == 'ChainMap'
    assert set(l) == {'a', 'b'}
    [result, result2] = selenium.run_js('\n        let c = pyodide.runPython(`\n            def test():\n                acc = 0\n                for i in range(10):\n                    r = yield acc\n                    acc += i * r\n            test()\n        `)\n        let {done, value} = c.next();\n        let result = [];\n        while(!done){\n            result.push(value);\n            ({done, value} = c.next(value + 1));\n        }\n        c.destroy();\n\n        function* test(){\n            let acc = 0;\n            for(let i=0; i < 10; i++){\n                let r = yield acc;\n                acc += i * r;\n            }\n        }\n        c = test();\n        ({done, value} = c.next());\n        let result2 = [];\n        while(!done){\n            result2.push(value);\n            ({done, value} = c.next(value + 1));\n        }\n        return [result, result2];\n        ')
    assert result == result2

def test_pyproxy_iter_error(selenium):
    if False:
        return 10
    selenium.run_js('\n        let t = pyodide.runPython(`\n            class T:\n                def __iter__(self):\n                    raise Exception(\'hi\')\n            T()\n        `);\n        assertThrows(() => t[Symbol.iterator](), "PythonError", "hi");\n        t.destroy();\n        ')

def test_pyproxy_iter_error2(selenium):
    if False:
        return 10
    selenium.run_js('\n        let gen = pyodide.runPython(`\n            def g():\n                yield 1\n                yield 2\n                raise Exception(\'hi\')\n                yield 3\n            g()\n        `);\n        assert(() => gen.next().value === 1);\n        assert(() => gen.next().value === 2);\n        assertThrows(() => gen.next(), "PythonError", "hi");\n        gen.destroy();\n        ')

def test_pyproxy_get_buffer(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        pyodide.runPython(`\n            from sys import getrefcount\n            z1 = memoryview(bytes(range(24))).cast("b", [8,3])\n            z2 = z1[-1::-1]\n        `);\n        for(let x of ["z1", "z2"]){\n            pyodide.runPython(`assert getrefcount(${x}) == 2`);\n            let proxy = pyodide.globals.get(x);\n            pyodide.runPython(`assert getrefcount(${x}) == 3`);\n            let z = proxy.getBuffer();\n            pyodide.runPython(`assert getrefcount(${x}) == 4`);\n            proxy.destroy();\n            pyodide.runPython(`assert getrefcount(${x}) == 3`);\n            for(let idx1 = 0; idx1 < 8; idx1++) {\n                for(let idx2 = 0; idx2 < 3; idx2++){\n                    let v1 = z.data[z.offset + z.strides[0] * idx1 + z.strides[1] * idx2];\n                    let v2 = pyodide.runPython(`repr(${x}[${idx1}, ${idx2}])`);\n                    if(v1.toString() !== v2){\n                        throw new Error(`Discrepancy ${x}[${idx1}, ${idx2}]: ${v1} != ${v2}`);\n                    }\n                }\n            }\n            z.release();\n            pyodide.runPython(`print("${x}", getrefcount(${x}))`);\n            pyodide.runPython(`assert getrefcount(${x}) == 2`);\n        }\n        ')

def test_get_empty_buffer(selenium):
    if False:
        for i in range(10):
            print('nop')
    "Previously empty buffers would raise alignment errors\n\n    This is because when Python makes an empty buffer, apparently the pointer\n    field is allowed to contain random garbage, which in particular won't be aligned.\n    "
    selenium.run_js('\n        let a = pyodide.runPython(`\n            from array import array\n            array("Q")\n        `);\n        let b = a.getBuffer();\n        b.release();\n        a.destroy();\n        ')

@pytest.mark.parametrize('array_type', [['i8', 'Int8Array', 'b'], ['u8', 'Uint8Array', 'B'], ['u8clamped', 'Uint8ClampedArray', 'B'], ['i16', 'Int16Array', 'h'], ['u16', 'Uint16Array', 'H'], ['i32', 'Int32Array', 'i'], ['u32', 'Uint32Array', 'I'], ['i64', 'BigInt64Array', 'q'], ['u64', 'BigUint64Array', 'Q'], ['f32', 'Float32Array', 'f'], ['f64', 'Float64Array', 'd']])
def test_pyproxy_get_buffer_type_argument(selenium, array_type):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        self.a = pyodide.runPython("bytes(range(256))");\n        assert(() => a instanceof pyodide.ffi.PyBuffer);\n        ')
    try:
        mv = memoryview(bytes(range(256)))
        (ty, array_ty, fmt) = array_type
        [check, result] = selenium.run_js(f'\n            let buf = a.getBuffer({ty!r});\n            assert(() => buf instanceof pyodide.ffi.PyBufferView);\n            let check = (buf.data.constructor.name === {array_ty!r});\n            let result = Array.from(buf.data);\n            if(typeof result[0] === "bigint"){{\n                result = result.map(x => x.toString(16));\n            }}\n            buf.release();\n            return [check, result];\n            ')
        assert check
        if fmt.lower() == 'q':
            assert result == [hex(x).replace('0x', '') for x in list(mv.cast(fmt))]
        elif fmt == 'f' or fmt == 'd':
            from math import isclose, isnan
            for (a, b) in zip(result, list(mv.cast(fmt)), strict=False):
                if a and b and (not (isnan(a) or isnan(b))):
                    assert isclose(a, b)
        else:
            assert result == list(mv.cast(fmt))
    finally:
        selenium.run_js('a.destroy(); self.a = undefined;')

def test_pyproxy_mixins1(selenium):
    if False:
        i = 10
        return i + 15
    result = selenium.run_js('\n        let [noimpls, awaitable, iterable, iterator, awaititerable, awaititerator] = pyodide.runPython(`\n            class NoImpls: pass\n\n            class Await:\n                def __await__(self):\n                    return iter([])\n\n            class Iter:\n                def __iter__(self):\n                    return iter([])\n\n            class Next:\n                def __next__(self):\n                    pass\n\n            class AwaitIter(Await, Iter): pass\n\n            class AwaitNext(Await, Next): pass\n            from pyodide.ffi import to_js\n            to_js([NoImpls(), Await(), Iter(), Next(), AwaitIter(), AwaitNext()])\n        `);\n        let name_proxy = {noimpls, awaitable, iterable, iterator, awaititerable, awaititerator};\n        let result = {};\n        for(let [name, x] of Object.entries(name_proxy)){\n            let impls = {};\n            for(let [name, key] of [\n                ["then", "then"],\n                ["catch", "catch"],\n                ["finally_", "finally"],\n                ["iterable", Symbol.iterator],\n                ["iterator", "next"],\n            ]){\n                impls[name] = key in x;\n            }\n            for(let name of ["PyAwaitable", "PyIterable", "PyIterator"]){\n                impls[name] = x instanceof pyodide.ffi[name];\n            }\n            result[name] = impls;\n            x.destroy();\n        }\n        return result;\n        ')
    assert result == dict(noimpls=dict(then=False, catch=False, finally_=False, iterable=False, iterator=False) | dict(PyAwaitable=False, PyIterable=False, PyIterator=False), awaitable=dict(then=True, catch=True, finally_=True, iterable=False, iterator=False) | dict(PyAwaitable=True, PyIterable=False, PyIterator=False), iterable=dict(then=False, catch=False, finally_=False, iterable=True, iterator=False) | dict(PyAwaitable=False, PyIterable=True, PyIterator=False), iterator=dict(then=False, catch=False, finally_=False, iterable=True, iterator=True) | dict(PyAwaitable=False, PyIterable=True, PyIterator=True), awaititerable=dict(then=True, catch=True, finally_=True, iterable=True, iterator=False) | dict(PyAwaitable=True, PyIterable=True, PyIterator=False), awaititerator=dict(then=True, catch=True, finally_=True, iterable=True, iterator=True) | dict(PyAwaitable=True, PyIterable=True, PyIterator=True))

def test_pyproxy_mixins2(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        let d = pyodide.runPython("{}");\n\n        assert(() => !("prototype" in d));\n        assert(() => !("caller" in d));\n        assert(() => !("name" in d));\n        assert(() => "length" in d);\n        assert(() => d instanceof pyodide.ffi.PyDict);\n        assert(() => d instanceof pyodide.ffi.PyProxyWithLength);\n        assert(() => d instanceof pyodide.ffi.PyProxyWithHas);\n        assert(() => d instanceof pyodide.ffi.PyProxyWithGet);\n        assert(() => d instanceof pyodide.ffi.PyProxyWithSet);\n\n        assert(() => "prototype" in d.__getitem__);\n        assert(() => d.__getitem__.prototype === undefined);\n        assert(() => !("length" in d.__getitem__));\n        assert(() => !("name" in d.__getitem__));\n\n        assert(() => d.$get.type === "builtin_function_or_method");\n        assert(() => d.get.type === undefined);\n        assert(() => d.set.type === undefined);\n        d.destroy();\n        ')

def test_pyproxy_mixins31(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        "use strict";\n        let [Test, t] = pyodide.runPython(`\n            class Test: pass\n            from pyodide.ffi import to_js\n            to_js([Test, Test()])\n        `);\n        assert(() => Test.prototype === undefined);\n        assert(() => !("name" in Test));\n        assert(() => !("length" in Test));\n\n        assert(() => !("prototype" in t));\n        assert(() => !("caller" in t));\n        assert(() => !("name" in t));\n        assert(() => !("length" in t));\n\n        Test.prototype = 7;\n        Test.name = 7;\n        Test.length = 7;\n        pyodide.runPython("assert Test.prototype == 7");\n        pyodide.runPython("assert Test.name == 7");\n        pyodide.runPython("assert Test.length == 7");\n        // prototype cannot be removed once added because it is nonconfigurable...\n        assertThrows(() => delete Test.prototype, "TypeError", "");\n        delete Test.name;\n        delete Test.length;\n        pyodide.runPython(`assert Test.prototype == 7`);\n        pyodide.runPython(`assert not hasattr(Test, "name")`);\n        pyodide.runPython(`assert not hasattr(Test, "length")`);\n\n        Test.$a = 7;\n        Object.defineProperty(Test, "a", {\n            get(){ return Test.$a + 1; },\n            set(v) {\n                Test.$a = v;\n            }\n        });\n\n        pyodide.runPython("assert Test.a == 7")\n        assert(() => Test.a === 8);\n        Test.a = 9;\n        assert(() => Test.a === 10);\n        pyodide.runPython("assert Test.a == 9")\n        assertThrows(() => delete Test.a, "TypeError", "");\n\n        Object.defineProperty(Test, "b", {\n            get(){ return Test.$a + 2; },\n        });\n        assert(() => Test.b === 11);\n        assertThrows(() => Test.b = 7,"TypeError", "");\n        assertThrows(() => delete Test.b, "TypeError", "");\n        Test.destroy();\n        t.destroy();\n        ')

@pytest.mark.parametrize('configurable', [False, True])
@pytest.mark.parametrize('writable', [False, True])
def test_pyproxy_mixins32(selenium, configurable, writable):
    if False:
        for i in range(10):
            print('nop')
    match selenium.browser:
        case 'node' | 'chrome':
            template = "'{}' on proxy: trap returned falsish for property 'x'"
            setText = template.format('set')
            deleteText = template.format('deleteProperty')
        case 'firefox':
            template = 'proxy {} handler returned false'
            setText = template.format('set')
            deleteText = template.format('deleteProperty')
        case 'safari':
            setText = "Proxy object's 'set' trap returned falsy value for property 'x'"
            deleteText = 'Unable to delete property.'
    selenium.run_js(f'\n        "use strict";\n        const configurable = !!{int(configurable)};\n        const writable = !!{int(writable)};\n        \n        const d = pyodide.runPython("{{}}");\n        Object.defineProperty(d, "x", {{\n            value: 9,\n            configurable,\n            writable,\n        }});\n        assert(() => d.x === 9);\n        if(writable) {{\n            d.x = 10;\n            assert(() => d.x === 10);\n        }} else {{\n            assertThrows(() => d.x = 10, "TypeError", "%s");\n        }}\n        if(configurable) {{\n            delete d.x;\n            assert(() => d.x === undefined);\n        }} else {{\n            assertThrows(() => delete d.x, "TypeError", "%s");\n        }}\n        d.destroy();\n        ' % (setText, deleteText))

def test_pyproxy_mixins41(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        [Test, t] = pyodide.runPython(`\n            class Test:\n                caller="fifty"\n                prototype="prototype"\n                name="me"\n                length=7\n                def __call__(self, x):\n                    return x + 1\n\n            from pyodide.ffi import to_js\n            to_js([Test, Test()])\n        `);\n        assert(() => Test.$prototype === "prototype");\n        assert(() => Test.prototype === "prototype");\n        assert(() => Test.name==="me");\n        assert(() => Test.length === 7);\n\n        assert(() => t.caller === "fifty");\n        assert(() => "prototype" in t);\n        assert(() => t.prototype === "prototype");\n        assert(() => t.name==="me");\n        assert(() => t.length === 7);\n        assert(() => t(7) === 8);\n        Test.destroy();\n        t.destroy();\n        ')

def test_pyproxy_mixins42(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        let t = pyodide.runPython(`\n            class Test:\n                def __call__(self, x):\n                    return x + 1\n\n            from pyodide.ffi import to_js\n            Test()\n        `);\n        assert(() => "prototype" in t);\n        assert(() => t.prototype === undefined);\n        t.destroy();\n        ')

def test_pyproxy_mixins5(selenium):
    if False:
        for i in range(10):
            print('nop')
    try:
        r = selenium.run_js('\n            "use strict";\n            const [Test, t] = pyodide.runPython(`\n                class Test:\n                    def __len__(self):\n                        return 9\n                from pyodide.ffi import to_js\n                to_js([Test, Test()])\n            `);\n            assert(() => !("length" in Test));\n            assert(() => t.length === 9);\n            assert(() => t instanceof pyodide.ffi.PyProxyWithLength);\n            assertThrows(() => {t.length = 10}, "TypeError", "");\n            assert(() => t.length === 9);\n\n            // For some reason, this is the normal behavior for a JS getter:\n            // delete just does nothing...\n            delete t.length;\n            assert(() => t.length === 9);\n\n            Test.destroy();\n            t.destroy();\n            ')
        print(r)
    finally:
        print(selenium.logs)

def test_pyproxy_mixins6(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        let l = pyodide.runPython(`\n            l = [5, 6, 7] ; l\n        `);\n        assert(() => l.get.type === undefined);\n        assert(() => l.get(1) === 6);\n        assert(() => l.length === 3);\n        assert(() => l instanceof pyodide.ffi.PyProxyWithLength);\n        assert(() => l instanceof pyodide.ffi.PyProxyWithHas);\n        assert(() => l instanceof pyodide.ffi.PyProxyWithGet);\n        assert(() => l instanceof pyodide.ffi.PyProxyWithSet);\n        l.set(0, 80);\n        pyodide.runPython(`\n            assert l[0] == 80\n        `);\n        l.delete(1);\n        pyodide.runPython(`\n            assert len(l) == 2 and l[1] == 7\n        `);\n        assert(() => l.length === 2 && l.get(1) === 7);\n        l.destroy();\n        ')

@pytest.mark.skip_pyproxy_check
def test_pyproxy_gc(selenium):
    if False:
        while True:
            i = 10
    if not hasattr(selenium, 'collect_garbage'):
        pytest.skip('No gc exposed')
    selenium.run_js('\n        self.x = new FinalizationRegistry((val) => { self.val = val; });\n        x.register({}, 77);\n        gc();\n        ')
    time.sleep(0.1)
    selenium.run_js('\n        gc();\n        ')
    assert selenium.run_js('return self.val;') == 77
    selenium.run_js('\n        self.res = new Map();\n\n        let d = pyodide.runPython(`\n            from js import res\n            def get_ref_count(x):\n                res[x] = sys.getrefcount(d)\n                return res[x]\n\n            import sys\n            class Test:\n                def __del__(self):\n                    res["destructor_ran"] = True\n\n                def get(self):\n                    return 7\n\n            d = Test()\n            get_ref_count(0)\n            d\n        `);\n        let get_ref_count = pyodide.globals.get("get_ref_count");\n        get_ref_count(1);\n        d.get();\n        get_ref_count(2);\n        d.get();\n        d.destroy()\n        ')
    selenium.collect_garbage()
    selenium.run('\n        get_ref_count(3)\n        del d\n        ')
    selenium.collect_garbage()
    a = selenium.run_js('return Array.from(res.entries());')
    assert dict(a) == {0: 2, 1: 3, 2: 4, 3: 2, 'destructor_ran': True}

@pytest.mark.skip_pyproxy_check
def test_pyproxy_gc_destroy(selenium):
    if False:
        return 10
    if not hasattr(selenium, 'collect_garbage'):
        pytest.skip('No gc exposed')
    selenium.run_js('\n        self.res = new Map();\n        let d = pyodide.runPython(`\n            from js import res\n            def get_ref_count(x):\n                res[x] = sys.getrefcount(d)\n                return res[x]\n            import sys\n            class Test:\n                def __del__(self):\n                    res["destructor_ran"] = True\n\n                def get(self):\n                    return 7\n\n            d = Test()\n            get_ref_count(0)\n            d\n        `);\n        let get_ref_count = pyodide.globals.get("get_ref_count");\n        get_ref_count(1);\n        d.get();\n        get_ref_count(2);\n        d.get();\n        get_ref_count(3);\n        delete d;\n        get_ref_count.destroy();\n        ')
    selenium.collect_garbage()
    selenium.collect_garbage()
    selenium.run('\n        get_ref_count(4)\n        del d\n        ')
    a = selenium.run_js('return Array.from(res.entries());')
    assert dict(a) == {0: 2, 1: 3, 2: 4, 3: 4, 4: 2, 'destructor_ran': True}

def test_pyproxy_implicit_copy(selenium):
    if False:
        print('Hello World!')
    result = selenium.run_js('\n        let result = [];\n        let a = pyodide.runPython(`d = { 1 : 2}; d`);\n        let b = pyodide.runPython(`d`);\n        result.push(a.get(1));\n        result.push(b.get(1));\n        a.destroy();\n        b.destroy();\n        return result;\n        ')
    assert result[0] == 2
    assert result[1] == 2

@pytest.mark.skip_pyproxy_check
def test_errors(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        const origDebug = pyodide.setDebug(true);\n        try {\n            const t = pyodide.runPython(`\n                from pyodide.ffi import to_js\n                def te(self, *args, **kwargs):\n                    raise Exception(repr(args))\n                class Temp:\n                    __getattr__ = te\n                    __setattr__ = te\n                    __delattr__ = te\n                    __dir__ = te\n                    __call__ = te\n                    __getitem__ = te\n                    __setitem__ = te\n                    __delitem__ = te\n                    __iter__ = te\n                    __len__ = te\n                    __contains__ = te\n                    __await__ = te\n                    __repr__ = te\n                to_js(Temp())\n                Temp()\n            `);\n            assertThrows(() => t.x, "PythonError", "");\n            try {\n                t.x;\n            } catch(e){\n                assert(() => e instanceof pyodide.ffi.PythonError);\n            }\n            assertThrows(() => t.x = 2, "PythonError", "");\n            assertThrows(() => delete t.x, "PythonError", "");\n            assertThrows(() => Object.getOwnPropertyNames(t), "PythonError", "");\n            assertThrows(() => t(), "PythonError", "");\n            assertThrows(() => t.get(1), "PythonError", "");\n            assertThrows(() => t.set(1, 2), "PythonError", "");\n            assertThrows(() => t.delete(1), "PythonError", "");\n            assertThrows(() => t.has(1), "PythonError", "");\n            assertThrows(() => t.length, "PythonError", "");\n            assertThrows(() => t.toString(), "PythonError", "");\n            assertThrows(() => Array.from(t), "PythonError", "");\n            await assertThrowsAsync(async () => await t, "PythonError", "");\n            t.destroy();\n            assertThrows(() => t.type, "Error",\n                "Object has already been destroyed\\n" +\n                \'The object was of type "Temp" and an error was raised when trying to generate its repr\'\n            );\n        } finally {\n            pyodide.setDebug(origDebug);\n        }\n        ')

@pytest.mark.skip_pyproxy_check
def test_nogil(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        let t = pyodide.runPython(`\n            def te(self, *args, **kwargs):\n                raise Exception(repr(args))\n            class Temp:\n                __getattr__ = te\n                __setattr__ = te\n                __delattr__ = te\n                __dir__ = te\n                __call__ = te\n                __getitem__ = te\n                __setitem__ = te\n                __delitem__ = te\n                __iter__ = te\n                __len__ = te\n                __contains__ = te\n                __await__ = te\n                __repr__ = te\n            Temp()\n        `);\n        // release GIL\n        const tstate = pyodide._module._PyEval_SaveThread();\n\n        try {\n            assertThrows(() => t.x, "NoGilError", "");\n            try {\n                t.x;\n            } catch(e){\n                assert(() => e instanceof pyodide._api.NoGilError);\n            }\n            assertThrows(() => t.x = 2, "NoGilError", "");\n            assertThrows(() => delete t.x, "NoGilError", "");\n            assertThrows(() => Object.getOwnPropertyNames(t), "NoGilError", "");\n            assertThrows(() => t(), "NoGilError", "");\n            assertThrows(() => t.get(1), "NoGilError", "");\n            assertThrows(() => t.set(1, 2), "NoGilError", "");\n            assertThrows(() => t.delete(1), "NoGilError", "");\n            assertThrows(() => t.has(1), "NoGilError", "");\n            assertThrows(() => t.length, "NoGilError", "");\n            assertThrows(() => t.toString(), "NoGilError", "");\n            assertThrows(() => Array.from(t), "NoGilError", "");\n            await assertThrowsAsync(async () => await t, "NoGilError", "");\n            assertThrows(() => t.destroy(), "NoGilError", "");\n        } finally {\n            // acquire GIL\n            pyodide._module._PyEval_RestoreThread(tstate)\n\n        }\n        ')

@pytest.mark.skip_pyproxy_check
def test_fatal_error(selenium_standalone):
    if False:
        return 10
    'Inject fatal errors in all the reasonable entrypoints'
    selenium_standalone.run_js('\n        let fatal_error = false;\n        let old_fatal_error = pyodide._api.fatal_error;\n        pyodide._api.fatal_error = (e) => {\n            fatal_error = true;\n            throw e;\n        }\n        try {\n            function expect_fatal(func){\n                fatal_error = false;\n                try {\n                    func();\n                } catch(e) {\n                    // pass\n                } finally {\n                    if(!fatal_error){\n                        throw new Error(`No fatal error occurred: ${func.toString().slice(6)}`);\n                    }\n                }\n            }\n            let t = pyodide.runPython(`\n                from _pyodide_core import trigger_fatal_error\n                def tfe(*args, **kwargs):\n                    trigger_fatal_error()\n                class Temp:\n                    __getattr__ = tfe\n                    __setattr__ = tfe\n                    __delattr__ = tfe\n                    __dir__ = tfe\n                    __call__ = tfe\n                    __getitem__ = tfe\n                    __setitem__ = tfe\n                    __delitem__ = tfe\n                    __iter__ = tfe\n                    __len__ = tfe\n                    __contains__ = tfe\n                    __await__ = tfe\n                    __repr__ = tfe\n                    __del__ = tfe\n                Temp()\n            `);\n            expect_fatal(() => "x" in t);\n            expect_fatal(() => t.x);\n            expect_fatal(() => t.x = 2);\n            expect_fatal(() => delete t.x);\n            expect_fatal(() => Object.getOwnPropertyNames(t));\n            expect_fatal(() => t());\n            expect_fatal(() => t.get(1));\n            expect_fatal(() => t.set(1, 2));\n            expect_fatal(() => t.delete(1));\n            expect_fatal(() => t.has(1));\n            expect_fatal(() => t.length);\n            expect_fatal(() => t.toString());\n            expect_fatal(() => Array.from(t));\n            t.destroy();\n            /*\n            // FIXME: Test `memory access out of bounds` error.\n            //        Testing this causes trouble on Chrome 97.0.4692.99 / ChromeDriver 97.0.4692.71.\n            //        (See: https://github.com/pyodide/pyodide/pull/2152)\n            a = pyodide.runPython(`\n                from array import array\n                array("I", [1,2,3,4])\n            `);\n            b = a.getBuffer();\n            b._view_ptr = 1e10;\n            expect_fatal(() => b.release());\n            */\n        } finally {\n            pyodide._api.fatal_error = old_fatal_error;\n        }\n        ')

def test_pyproxy_call(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        pyodide.runPython(`\n            from pyodide.ffi import to_js\n            def f(x=2, y=3):\n                return to_js([x, y])\n        `);\n        self.f = pyodide.globals.get("f");\n        ')

    def assert_call(s, val):
        if False:
            return 10
        res = selenium.run_js(f'return {s};')
        assert res == val
    assert_call('f()', [2, 3])
    assert_call('f(7)', [7, 3])
    assert_call('f(7, -1)', [7, -1])
    assert_call('f.callKwargs({})', [2, 3])
    assert_call('f.callKwargs(7, {})', [7, 3])
    assert_call('f.callKwargs(7, -1, {})', [7, -1])
    assert_call('f.callKwargs({ y : 4 })', [2, 4])
    assert_call('f.callKwargs({ y : 4, x : 9 })', [9, 4])
    assert_call('f.callKwargs(8, { y : 4 })', [8, 4])
    msg = 'TypeError: callKwargs requires at least one argument'
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run_js('f.callKwargs()')
    msg = 'TypeError: callKwargs requires at least one argument'
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run_js('f.callKwargs()')
    msg = "TypeError: f\\(\\) got an unexpected keyword argument 'z'"
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run_js('f.callKwargs({z : 6})')
    msg = "TypeError: f\\(\\) got multiple values for argument 'x'"
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run_js('f.callKwargs(76, {x : 6})')
    selenium.run_js('f.destroy()')

def test_pyproxy_borrow(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        let t = pyodide.runPython(`\n            class Tinner:\n                def f(self):\n                    return 7\n            class Touter:\n                T = Tinner()\n            Touter\n        `);\n        assert(() => t.T.f() === 7);\n        let T = t.T;\n        let Tcopy = T.copy();\n        assert(() => T.f() === 7);\n        assert(() => Tcopy.f() === 7);\n        t.destroy();\n        assert(() => Tcopy.f() === 7);\n        assertThrows(() => T.f(), "Error", "automatically destroyed in the process of destroying the proxy it was borrowed from");\n        Tcopy.destroy();\n        ')

def test_coroutine_scheduling(selenium):
    if False:
        print('Hello World!')
    selenium.run_js("\n        let f = pyodide.runPython(`\n            x = 0\n            async def f():\n                global x\n                print('hi!')\n                x += 1\n            f\n        `);\n        setTimeout(f, 100);\n        await sleep(200);\n        assert(() => pyodide.globals.get('x') === 1);\n        f.destroy();\n        ")

def test_pyproxy_apply(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        pyodide.runPython(`\n            from pyodide.ffi import to_js\n            def f(*args):\n                return to_js(args)\n        `);\n        let fpy = pyodide.globals.get("f");\n        let fjs = function(...args){ return args; };\n        let examples = [\n            undefined,\n            null,\n            {},\n            {0:1, 1:7, 2: -3},\n            { *[Symbol.iterator](){yield 3; yield 5; yield 7;} },\n            {0:1, 1:7, 2: -3, length: 2},\n            [1,7,9,5],\n            function(a,b,c){},\n        ];\n        for(let input of examples){\n            assert(() => JSON.stringify(fpy.apply(undefined, input)) === JSON.stringify(fjs.apply(undefined, input)));\n        }\n\n        for(let error_input of [1, "abc", 1n, Symbol.iterator, true]) {\n            assertThrows(() => fjs.apply(undefined, error_input), "TypeError", "");\n            assertThrows(() => fpy.apply(undefined, error_input), "TypeError", "");\n        }\n\n        fpy.destroy();\n        ')

def test_pyproxy_this1(selenium):
    if False:
        return 10
    selenium.run_js('\n        let f = pyodide.runPython(`\n            x = 0\n            def f(self, x):\n                return getattr(self, x)\n            f\n        `);\n\n        let x = {};\n        x.f = f.captureThis();\n        x.a = 7;\n        assert(() => x.f("a") === 7 );\n        f.destroy();\n        ')

def test_pyproxy_this2(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        let g = pyodide.runPython(`\n            x = 0\n            from pyodide.ffi import to_js\n            def f(*args):\n                return to_js(args)\n            f\n        `);\n\n        let f = g.captureThis();\n        let fjs = function(...args){return [this, ...args];};\n\n        let f1 = f.bind(1);\n        let fjs1 = fjs.bind(1);\n        assert(() => JSON.stringify(f1(2, 3, 4)) === JSON.stringify(fjs1(2, 3, 4)));\n\n        let f2 = f1.bind(2);\n        let fjs2 = fjs1.bind(2);\n        assert(() => JSON.stringify(f2(2, 3, 4)) === JSON.stringify(fjs2(2, 3, 4)));\n        let f3 = f.bind(2);\n        let fjs3 = fjs.bind(2);\n        assert(() => JSON.stringify(f3(2, 3, 4)) === JSON.stringify(fjs3(2, 3, 4)));\n\n        let gjs = function(...args){return [...args];};\n\n        let g1 = g.bind(1, 2, 3, 4);\n        let gjs1 = gjs.bind(1, 2, 3, 4);\n\n        let g2 = g1.bind(5, 6, 7, 8);\n        let gjs2 = gjs1.bind(5, 6, 7, 8);\n\n        let g3 = g2.captureThis();\n\n        assert(() => JSON.stringify(g1(-1, -2, -3, -4)) === JSON.stringify(gjs1(-1, -2, -3, -4)));\n        assert(() => JSON.stringify(g2(-1, -2, -3, -4)) === JSON.stringify(gjs2(-1, -2, -3, -4)));\n        assert(() => JSON.stringify(g3(-1, -2, -3, -4)) === JSON.stringify([1, 2, 3, 4, 6, 7, 8, -1, -2, -3, -4]));\n        g.destroy();\n        ')

@run_in_pyodide
async def test_async_iter1(selenium):
    from pyodide.code import run_js

    class Gen:

        async def __aiter__(self):
            yield 1
            yield 2
    g = Gen()
    p = run_js('\n        async (g) => {\n            assert(() => g instanceof pyodide.ffi.PyAsyncIterable);\n            let r = [];\n            for await (let a of g) {\n                r.push(a);\n            }\n            return r;\n        }\n    ')(g)
    assert (await p).to_py() == [1, 2]

@run_in_pyodide
async def test_async_iter2(selenium):
    from pyodide.code import run_js

    class Gen:

        def __init__(self):
            if False:
                return 10
            self.i = 0

        def __aiter__(self):
            if False:
                print('Hello World!')
            return self

        async def __anext__(self):
            self.i += 1
            if self.i > 2:
                raise StopAsyncIteration
            return self.i
    g = Gen()
    p = run_js('\n        async (g) => {\n            assert(() => g instanceof pyodide.ffi.PyAsyncIterable);\n            let r = [];\n            for await (let a of g) {\n                r.push(a);\n            }\n            return r;\n        }\n    ')(g)
    assert (await p).to_py() == [1, 2]

@run_in_pyodide
def test_gen(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js

    def g():
        if False:
            print('Hello World!')
        n = 0
        for _ in range(3):
            n = (yield (n + 2))
    p = run_js('\n        (g) => {\n            assert(() => g instanceof pyodide.ffi.PyGenerator);\n            assert(() => g instanceof pyodide.ffi.PyIterable);\n            assert(() => g instanceof pyodide.ffi.PyIterator);\n            assert(() => !(g instanceof pyodide.ffi.PyAsyncGenerator));\n            let r = [];\n            r.push(g.next());\n            r.push(g.next(3));\n            r.push(g.next(4));\n            r.push(g.next(5));\n            return r;\n        }\n    ')(g())
    assert p.to_py() == [{'done': False, 'value': 2}, {'done': False, 'value': 5}, {'done': False, 'value': 6}, {'done': True, 'value': None}]

@run_in_pyodide
def test_gen_return(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js

    def g1():
        if False:
            while True:
                i = 10
        yield 1
        yield 2
    p = run_js('\n        (g) => {\n            let r = [];\n            r.push(g.next());\n            r.push(g.return(5));\n            return r;\n        }\n    ')(g1())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': True, 'value': 5}]

    def g2():
        if False:
            for i in range(10):
                print('nop')
        try:
            yield 1
            yield 2
        finally:
            yield 3
            return 5
    p = run_js('\n        (g) => {\n            let r = [];\n            r.push(g.next());\n            r.push(g.return(5));\n            r.push(g.next());\n            return r;\n        }\n    ')(g2())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': False, 'value': 3}, {'done': True, 'value': 5}]

    def g3():
        if False:
            print('Hello World!')
        try:
            yield 1
            yield 2
        finally:
            return 3
    p = run_js('\n        (g) => {\n            let r = [];\n            r.push(g.next());\n            r.push(g.return(5));\n            return r;\n        }\n    ')(g3())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': True, 'value': 3}]

@run_in_pyodide
def test_gen_throw(selenium):
    if False:
        for i in range(10):
            print('nop')
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsException

    def g1():
        if False:
            i = 10
            return i + 15
        yield 1
        yield 2
    p = run_js("\n        (g) => {\n            g.next();\n            g.throw(new TypeError('hi'));\n        }\n    ")
    with pytest.raises(JsException, match='hi'):
        p(g1())

    def g2():
        if False:
            print('Hello World!')
        try:
            yield 1
            yield 2
        finally:
            yield 3
            return 5
    p = run_js("\n        (g) => {\n            let r = [];\n            r.push(g.next());\n            r.push(g.throw(new TypeError('hi')));\n            r.push(g.next());\n            return r;\n        }\n    ")(g2())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': False, 'value': 3}, {'done': True, 'value': 5}]

    def g3():
        if False:
            for i in range(10):
                print('nop')
        try:
            yield 1
            yield 2
        finally:
            return 3
    p = run_js("\n        (g) => {\n            let r = [];\n            r.push(g.next());\n            r.push(g.throw(new TypeError('hi')));\n            return r;\n        }\n    ")(g3())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': True, 'value': 3}]

@run_in_pyodide
async def test_async_gen1(selenium):
    from pyodide.code import run_js

    async def g():
        n = 0
        for _ in range(3):
            n = (yield (n + 2))
    p = run_js('\n        async (g) => {\n            assert(() => g instanceof pyodide.ffi.PyAsyncGenerator);\n            assert(() => g instanceof pyodide.ffi.PyAsyncIterable);\n            assert(() => g instanceof pyodide.ffi.PyAsyncIterator);\n            assert(() => !(g instanceof pyodide.ffi.PyGenerator));\n            let r = [];\n            r.push(await g.next());\n            r.push(await g.next(3));\n            r.push(await g.next(4));\n            r.push(await g.next(5));\n            return r;\n        }\n    ')(g())
    assert (await p).to_py() == [{'done': False, 'value': 2}, {'done': False, 'value': 5}, {'done': False, 'value': 6}, {'done': True, 'value': None}]

@run_in_pyodide
async def test_async_gen2(selenium):
    from pyodide.code import run_js

    async def g():
        for n in range(3):
            yield n
    p = run_js('\n        async (g) => {\n            let result = [];\n            for await (let x of g){\n                result.push(x);\n            }\n            return result;\n        }\n    ')(g())
    assert (await p).to_py() == [0, 1, 2]

@run_in_pyodide
async def test_async_gen_return(selenium):
    from pyodide.code import run_js

    async def g1():
        yield 1
        yield 2
    p = await run_js('\n        async (g) => {\n            let r = [];\n            r.push(await g.next());\n            r.push(await g.return(5));\n            return r;\n        }\n    ')(g1())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': True, 'value': 5}]

    async def g2():
        try:
            yield 1
            yield 2
        finally:
            yield 3
            return
    p = await run_js('\n        async (g) => {\n            let r = [];\n            r.push(await g.next());\n            r.push(await g.return(5));\n            r.push(await g.next());\n            return r;\n        }\n    ')(g2())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': False, 'value': 3}, {'done': True, 'value': None}]

    async def g3():
        try:
            yield 1
            yield 2
        finally:
            return
    p = await run_js('\n        async (g) => {\n            let r = [];\n            r.push(await g.next());\n            r.push(await g.return(5));\n            return r;\n        }\n    ')(g3())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': True, 'value': None}]

@run_in_pyodide
async def test_async_gen_throw(selenium):
    import pytest
    from pyodide.code import run_js
    from pyodide.ffi import JsException

    async def g1():
        yield 1
        yield 2
    p = run_js("\n        async (g) => {\n            await g.next();\n            await g.throw(new TypeError('hi'));\n        }\n    ")
    with pytest.raises(JsException, match='hi'):
        await p(g1())

    async def g2():
        try:
            yield 1
            yield 2
        finally:
            yield 3
            return
    p = await run_js("\n        async (g) => {\n            let r = [];\n            r.push(await g.next());\n            r.push(await g.throw(new TypeError('hi')));\n            r.push(await g.next());\n            return r;\n        }\n    ")(g2())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': False, 'value': 3}, {'done': True, 'value': None}]

    async def g3():
        try:
            yield 1
            yield 2
        finally:
            return
    p = await run_js("\n        async (g) => {\n            let r = [];\n            r.push(await g.next());\n            r.push(await g.throw(new TypeError('hi')));\n            return r;\n        }\n    ")(g3())
    assert p.to_py() == [{'done': False, 'value': 1}, {'done': True, 'value': None}]

@run_in_pyodide
def test_roundtrip_no_destroy(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    from pyodide.ffi import create_proxy
    from pyodide_js._api import pyproxyIsAlive as isalive
    p = create_proxy({1: 2})
    run_js('(x) => x')(p)
    assert isalive(p)
    run_js('\n        (p) => {\n            p.destroy({destroyRoundtrip : false});\n        }\n        ')(p)
    assert isalive(p)
    run_js('\n        (p) => {\n            p.destroy({destroyRoundtrip : true});\n        }\n        ')(p)
    assert not isalive(p)
    p = create_proxy({1: 2})
    run_js('\n        (p) => {\n            p.destroy();\n        }\n        ')(p)
    assert not isalive(p)

@run_in_pyodide
async def test_multiple_interpreters(selenium):
    from js import loadPyodide
    py2 = await loadPyodide()
    d1 = {'a': 2}
    d2 = py2.runPython(str(d1))
    assert d2.toJs().to_py() == d1

@run_in_pyodide
def test_pyproxy_of_list_index(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    pylist = [9, 8, 7]
    jslist = run_js('\n        (p) => {\n            return [p[0], p[1], p[2]]\n        }\n        ')(pylist)
    assert jslist.to_py() == pylist

@run_in_pyodide
def test_pyproxy_of_list_join(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['Wind', 'Water', 'Fire']
    ajs = to_js(a)
    func = run_js('((a, k) => a.join(k))')
    assert func(a, None) == func(ajs, None)
    assert func(a, ', ') == func(ajs, ', ')
    assert func(a, ' ') == func(ajs, ' ')

@run_in_pyodide
def test_pyproxy_of_list_slice(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['ant', 'bison', 'camel', 'duck', 'elephant']
    ajs = to_js(a)
    func_strs = ['a.slice(2)', 'a.slice(2, 4)', 'a.slice(1, 5)', 'a.slice(-2)', 'a.slice(2, -1)', 'a.slice()']
    for func_str in func_strs:
        func = run_js(f'(a) => {func_str}')
        assert func(a).to_py() == func(ajs).to_py()

@run_in_pyodide
def test_pyproxy_of_list_indexOf(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['ant', 'bison', 'camel', 'duck', 'bison']
    ajs = to_js(a)
    func_strs = ["beasts.indexOf('bison')", "beasts.indexOf('bison', 2)", "beasts.indexOf('bison', -4)", "beasts.indexOf('bison', 3)", "beasts.indexOf('giraffe')"]
    for func_str in func_strs:
        func = run_js(f'(beasts) => {func_str}')
        assert func(a) == func(ajs)

@run_in_pyodide
def test_pyproxy_of_list_lastIndexOf(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['ant', 'bison', 'camel', 'duck', 'bison']
    ajs = to_js(a)
    func_strs = ["beasts.lastIndexOf('bison')", "beasts.lastIndexOf('bison', 2)", "beasts.lastIndexOf('bison', -4)", "beasts.lastIndexOf('bison', 3)", "beasts.lastIndexOf('giraffe')"]
    for func_str in func_strs:
        func = run_js(f'(beasts) => {func_str}')
        assert func(a) == func(ajs)

@run_in_pyodide
def test_pyproxy_of_list_forEach(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['a', 'b', 'c']
    ajs = to_js(a)
    func = run_js('\n        ((a) => {\n            let s = "";\n            a.forEach((elt, idx, list) => {\n                s += "::";\n                s += idx;\n                s += elt;\n                s += this[elt];\n            },\n                {a: 6, b: 9, c: 22}\n            );\n            return s;\n        })\n        ')
    assert func(a) == func(ajs)

@run_in_pyodide
def test_pyproxy_of_list_map(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['a', 'b', 'c']
    ajs = to_js(a)
    func = run_js('\n        (a) => a.map(\n            function (elt, idx, list){\n                return [elt, idx, this[elt]]\n            },\n            {a: 6, b: 9, c: 22}\n        )\n        ')
    assert func(a).to_py() == func(ajs).to_py()

@run_in_pyodide
def test_pyproxy_of_list_filter(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = list(range(20, 0, -2))
    ajs = to_js(a)
    func = run_js('\n        (a) => a.filter(\n            function (elt, idx){\n                return elt + idx > 12\n            }\n        )\n        ')
    assert func(a).to_py() == func(ajs).to_py()

@run_in_pyodide
def test_pyproxy_of_list_reduce(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = list(range(20, 0, -2))
    ajs = to_js(a)
    func = run_js('\n        (a) => a.reduce((l, r) => l + 2*r)\n        ')
    assert func(a) == func(ajs)

@run_in_pyodide
def test_pyproxy_of_list_reduceRight(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = list(range(20, 0, -2))
    ajs = to_js(a)
    func = run_js('\n        (a) => a.reduceRight((l, r) => l + 2*r)\n        ')
    assert func(a) == func(ajs)

@run_in_pyodide
def test_pyproxy_of_list_some(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    func = run_js('(a) => a.some((element, idx) => (element + idx) % 2 === 0)')
    for a in [[1, 2, 3, 4, 5], [2, 3, 4, 5], [1, 3, 5], [1, 4, 5], [4, 5]]:
        assert func(a) == func(to_js(a))

@run_in_pyodide
def test_pyproxy_of_list_every(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    func = run_js('(a) => a.every((element, idx) => (element + idx) % 2 === 0)')
    for a in [[1, 2, 3, 4, 5], [2, 3, 4, 5], [1, 3, 5], [1, 4, 5], [4, 5]]:
        assert func(a) == func(to_js(a))

@run_in_pyodide
def test_pyproxy_of_list_at(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = [5, 12, 8, 130, 44]
    ajs = to_js(a)
    func = run_js('(a, idx) => a.at(idx)')
    for idx in [2, 3, 4, -2, -3, -4, 5, 7, -7]:
        assert func(a, idx) == func(ajs, idx)

@run_in_pyodide
def test_pyproxy_of_list_concat(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = [[5, 12, 8], [130, 44], [6, 7, 7]]
    ajs = to_js(a)
    func = run_js('(a, b, c) => a.concat(b, c)')
    assert func(*a).to_py() == func(*ajs).to_py()

@run_in_pyodide
def test_pyproxy_of_list_includes(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = [5, 12, 8, 130, 44, 6, 7, 7]
    ajs = to_js(a)
    func = run_js('(a, n) => a.includes(n)')
    for n in range(4, 10):
        assert func(a, n) == func(ajs, n)

@run_in_pyodide
def test_pyproxy_of_list_entries(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = [5, 12, 8, 130, 44, 6, 7, 7]
    ajs = to_js(a)
    func = run_js('(a, k) => Array.from(a[k]())')
    for k in ['entries', 'keys', 'values']:
        assert func(a, k).to_py() == func(ajs, k).to_py()

@run_in_pyodide
def test_pyproxy_of_list_find(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = [5, 12, 8, 130, 44, 6, 7, 7]
    ajs = to_js(a)
    func = run_js('(a, k) => a[k](element => element > 10)')
    for k in ['find', 'findIndex']:
        assert func(a, k) == func(ajs, k)

@run_in_pyodide
def test_pyproxy_of_list_sort(selenium):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    stringArray = ['Blue', 'Humpback', 'Beluga']
    numberArray = [40, None, 1, 5, 200]
    numericStringArray = ['80', '9', '700']
    mixedNumericArray = ['80', '9', '700', 40, 1, 5, 200]
    run_js('globalThis.compareNumbers = (a, b) => a - b')
    assert run_js('((a) => a.join())')(stringArray) == 'Blue,Humpback,Beluga'
    assert run_js('((a) => a.sort())')(stringArray) is stringArray
    assert stringArray == ['Beluga', 'Blue', 'Humpback']
    assert run_js('((a) => a.join())')(numberArray) == '40,,1,5,200'
    assert run_js('((a) => a.sort())')(numberArray) == [1, 200, 40, 5, None]
    assert run_js('((a) => a.sort(compareNumbers))')(numberArray) == [1, 5, 40, 200, None]
    assert run_js('((a) => a.join())')(numericStringArray) == '80,9,700'
    assert run_js('((a) => a.sort())')(numericStringArray) == ['700', '80', '9']
    assert run_js('((a) => a.sort(compareNumbers))')(numericStringArray) == ['9', '80', '700']
    assert run_js('((a) => a.join())')(mixedNumericArray) == '80,9,700,40,1,5,200'
    assert run_js('((a) => a.sort())')(mixedNumericArray) == [1, 200, 40, 5, '700', '80', '9']
    assert run_js('((a) => a.sort(compareNumbers))')(mixedNumericArray) == [1, 5, '9', 40, '80', 200, '700']

@run_in_pyodide
def test_pyproxy_of_list_reverse(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = [3, 2, 4, 1, 5]
    ajs = to_js(a)
    func = run_js('((a) => a.reverse())')
    assert func(a) is a
    func(ajs)
    assert ajs.to_py() == a

@pytest.mark.parametrize('func', ['splice(2, 0, "drum")', 'splice(2, 0, "drum", "guitar")', 'splice(3, 1)', 'splice(2, 1, "trumpet")', 'splice(0, 2, "parrot", "anemone", "blue")', 'splice(2, 2)', 'splice(-2, 1)', 'splice(2)', 'splice()'])
@run_in_pyodide
def test_pyproxy_of_list_splice(selenium, func):
    if False:
        return 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['angel', 'clown', 'mandarin', 'sturgeon']
    ajs = to_js(a)
    func = run_js(f'((a) => a.{func})')
    assert func(a).to_py() == func(ajs).to_py()
    assert a == ajs.to_py()

@run_in_pyodide
def test_pyproxy_of_list_push(selenium):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = [4, 5, 6]
    ajs = to_js(a)
    func = run_js('(a) => a.push(1, 2, 3)')
    assert func(a) == func(ajs)
    assert ajs.to_py() == a
    a = [4, 5, 6]
    ajs = to_js(a)
    func = run_js('\n        (a) => {\n            a.push(1);\n            a.push(2);\n            return a.push(3);\n        }\n        ')
    assert func(a) == func(ajs)
    assert ajs.to_py() == a

@run_in_pyodide
def test_pyproxy_of_list_pop(selenium):
    if False:
        i = 10
        return i + 15
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    func = run_js('((a) => a.pop())')
    for a in [[], ['broccoli', 'cauliflower', 'cabbage', 'kale', 'tomato']]:
        ajs = to_js(a)
        assert func(a) == func(ajs)
        assert ajs.to_py() == a

@run_in_pyodide
def test_pyproxy_of_list_shift(selenium):
    if False:
        while True:
            i = 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['Andrew', 'Tyrone', 'Paul', 'Maria', 'Gayatri']
    ajs = to_js(a)
    func = run_js('\n        (a) => {\n            let result = [];\n            while (typeof (i = a.shift()) !== "undefined") {\n                result.push(i);\n            }\n            return result;\n        }\n        ')
    assert func(a).to_py() == func(ajs).to_py()
    assert a == []
    assert ajs.to_py() == []

@run_in_pyodide
def test_pyproxy_of_list_unshift(selenium):
    if False:
        return 10
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = [4, 5, 6]
    ajs = to_js(a)
    func = run_js('(a) => a.unshift(1, 2, 3)')
    assert func(a) == func(ajs)
    assert ajs.to_py() == a
    a = [4, 5, 6]
    ajs = to_js(a)
    func = run_js('\n        (a) => {\n            a.unshift(1);\n            a.unshift(2);\n            return a.unshift(3);\n        }\n        ')
    assert func(a) == func(ajs)
    assert ajs.to_py() == a

@pytest.mark.parametrize('func', ['copyWithin(-2)', 'copyWithin(0, 3)', 'copyWithin(0, 3, 4)', 'copyWithin(-2, -3, -1)'])
@run_in_pyodide
def test_pyproxy_of_list_copyWithin(selenium, func):
    if False:
        for i in range(10):
            print('nop')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['a', 'b', 'c', 'd', 'e']
    ajs = to_js(a)
    func = run_js(f'(a) => a.{func}')
    assert func(a) is a
    func(ajs)
    assert a == ajs.to_py()

@pytest.mark.parametrize('func', ['fill(0, 2, 4)', 'fill(5, 1)', 'fill(6)'])
@run_in_pyodide
def test_pyproxy_of_list_fill(selenium, func):
    if False:
        print('Hello World!')
    from pyodide.code import run_js
    from pyodide.ffi import to_js
    a = ['a', 'b', 'c', 'd', 'e']
    ajs = to_js(a)
    func = run_js(f'(a) => a.{func}')
    assert func(a) is a
    func(ajs)
    assert a == ajs.to_py()

def test_pyproxy_instanceof_function(selenium):
    if False:
        return 10
    weird_function_shim = ''
    if selenium.browser in ['firefox', 'node']:
        weird_function_shim = 'let Function = pyodide._api.tests.Function;'
    selenium.run_js(f'\n        {weird_function_shim}\n        \n        const pyFunc_0 = pyodide.runPython(`\n            lambda: print("zero")\n        `);\n\n        const pyFunc_1 = pyodide.runPython(`\n            def foo():\n                print("two")\n            foo\n        `);\n\n        const pyFunc_2 = pyodide.runPython(`\n            class A():\n                def a(self):\n                    print("three") # method from class\n            A.a\n        `);\n\n        const pyFunc_3 = pyodide.runPython(`\n            class B():\n                def __call__(self):\n                    print("five (B as a callable instance)")\n\n            b = B()\n            b\n        `);\n\n        assert(() => pyFunc_0 instanceof Function);\n        assert(() => pyFunc_0 instanceof pyodide.ffi.PyProxy);\n        assert(() => pyFunc_0 instanceof pyodide.ffi.PyCallable);\n\n        assert(() => pyFunc_1 instanceof Function);\n        assert(() => pyFunc_1 instanceof pyodide.ffi.PyProxy);\n        assert(() => pyFunc_1 instanceof pyodide.ffi.PyCallable);\n\n        assert(() => pyFunc_2 instanceof Function);\n        assert(() => pyFunc_2 instanceof pyodide.ffi.PyProxy);\n        assert(() => pyFunc_2 instanceof pyodide.ffi.PyCallable);\n\n        assert(() => pyFunc_3 instanceof Function);\n        assert(() => pyFunc_3 instanceof pyodide.ffi.PyProxy);\n        assert(() => pyFunc_3 instanceof pyodide.ffi.PyCallable);\n\n        d = pyodide.runPython("{{}}");\n        assert(() => !(d instanceof Function));\n        assert(() => !(d instanceof pyodide.ffi.PyCallable));\n        assert(() => d instanceof pyodide.ffi.PyProxy);\n        assert(() => d instanceof pyFunc_0.constructor);\n        assert(() => pyFunc_0 instanceof d.constructor);\n\n        for(const p of [pyFunc_0, pyFunc_1, pyFunc_2, pyFunc_3, d])  {{\n            p.destroy();\n        }}\n        ')

def test_pyproxy_callable_prototype(selenium):
    if False:
        print('Hello World!')
    result = selenium.run_js('\n        const o = pyodide.runPython("lambda:None");\n        const res = Object.fromEntries(Reflect.ownKeys(Function.prototype).map(k => [k.toString(), k in o]));\n        o.destroy();\n        return res;\n        ')
    subdict = {'length': False, 'name': False, 'arguments': False, 'caller': False, 'apply': True, 'bind': True, 'call': True, 'Symbol(Symbol.hasInstance)': True}
    filtered_result = {k: v for (k, v) in result.items() if k in subdict}
    assert filtered_result == subdict

@pytest.mark.skip_pyproxy_check
def test_automatic_coroutine_scheduling(selenium):
    if False:
        i = 10
        return i + 15
    res = selenium.run_js('\n        function d(x) {\n            if(x && x.destroy) {\n                x.destroy();\n            }\n        }\n\n        d(pyodide.runPython(`\n            l = []\n            async def f(n):\n                l.append(n)\n\n            def g(n):\n                return f(n)\n\n            async def h(n):\n                return f(n)\n\n            f(1)\n        `));\n        const f = pyodide.globals.get("f");\n        const g = pyodide.globals.get("g");\n        const h = pyodide.globals.get("h");\n        f(3);\n        d(pyodide.runPython("f(2)"));\n        pyodide.runPythonAsync("f(4)");\n        d(g(5));\n        h(6);\n        await sleep(0);\n        await sleep(0);\n        await sleep(0);\n        const l = pyodide.globals.get("l");\n        const res = l.toJs();\n        for(let p of [f, g, l]) {\n            p.destroy();\n        }\n        return res;\n        ')
    assert res == [3, 4, 6]