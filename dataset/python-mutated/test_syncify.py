import pytest

@pytest.mark.xfail_browsers(node="Scopes don't work as needed")
def test_syncify_not_supported(selenium_standalone_noload):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_standalone_noload
    selenium.run_js('\n        // Ensure that it\'s not supported by deleting WebAssembly.Suspender\n        delete WebAssembly.Suspender;\n        let pyodide = await loadPyodide({});\n        await assertThrowsAsync(\n          async () => await pyodide.runPythonSyncifying("1+1"),\n          "Error",\n          "WebAssembly stack switching not supported in this JavaScript runtime"\n        );\n        await assertThrows(\n          () => pyodide.runPython("from js import sleep; sleep().syncify()"),\n          "PythonError",\n          "RuntimeError: WebAssembly stack switching not supported in this JavaScript runtime"\n        );\n        ')

@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_syncify1(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js("\n        await pyodide.runPythonSyncifying(`\n            from pyodide.code import run_js\n\n            test = run_js(\n                '''\n                (async function test() {\n                    await sleep(1000);\n                    return 7;\n                })\n                '''\n            )\n            assert test().syncify() == 7\n            del test\n        `);\n        ")

@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_syncify2(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        await pyodide.runPythonSyncifying(`\n            from pyodide_js import loadPackage\n            loadPackage("pytest").syncify()\n            import pytest\n            import importlib.metadata\n            with pytest.raises(ModuleNotFoundError):\n                importlib.metadata.version("micropip")\n\n            from pyodide_js import loadPackage\n            loadPackage("micropip").syncify()\n\n            assert importlib.metadata.version("micropip")\n        `);\n        ')

@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_syncify_error(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        await pyodide.loadPackage("pytest");\n        await pyodide.runPythonSyncifying(`\n            def temp():\n                from pyodide.code import run_js\n\n                asyncThrow = run_js(\n                    \'\'\'\n                    (async function asyncThrow(){\n                        throw new Error("hi");\n                    })\n                    \'\'\'\n                )\n                from pyodide.ffi import JsException\n                import pytest\n                with pytest.raises(JsException, match="hi"):\n                    asyncThrow().syncify()\n            temp()\n        `);\n        ')

@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_syncify_null(selenium):
    if False:
        i = 10
        return i + 15
    selenium.run_js('\n        await pyodide.loadPackage("pytest");\n        await pyodide.runPythonSyncifying(`\n            def temp():\n                from pyodide.code import run_js\n\n                asyncNull = run_js(\n                    \'\'\'\n                    (async function asyncThrow(){\n                        await sleep(50);\n                        return null;\n                    })\n                    \'\'\'\n                )\n                assert asyncNull().syncify() is None\n            temp()\n        `);\n        ')

@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_syncify_no_suspender(selenium):
    if False:
        while True:
            i = 10
    selenium.run_js('\n        await pyodide.loadPackage("pytest");\n        await pyodide.runPython(`\n            from pyodide.code import run_js\n            import pytest\n\n            test = run_js(\n                \'\'\'\n                (async function test() {\n                    await sleep(1000);\n                    return 7;\n                })\n                \'\'\'\n            )\n            with pytest.raises(RuntimeError, match="No suspender"):\n                test().syncify()\n            del test\n        `);\n        ')

@pytest.mark.requires_dynamic_linking
@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_syncify_getset(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        await pyodide.loadPackage("fpcast-test")\n        await pyodide.runPythonSyncifying(`\n            def temp():\n                from pyodide.code import run_js\n\n                test = run_js(\n                    \'\'\'\n                    (async function test() {\n                        await sleep(1000);\n                        return 7;\n                    })\n                    \'\'\'\n                )\n                x = []\n                def wrapper():\n                    x.append(test().syncify())\n\n                import fpcast_test\n                t = fpcast_test.TestType()\n                t.getset_jspi_test = wrapper\n                t.getset_jspi_test\n                t.getset_jspi_test = None\n                assert x == [7, 7]\n            temp()\n        `);\n        ')

@pytest.mark.requires_dynamic_linking
@pytest.mark.xfail(reason='Will fix in a followup')
def test_syncify_ctypes():
    if False:
        return 10
    selenium.run_js("\n        await pyodide.runPythonSyncifying(`\n            from pyodide.code import run_js\n\n            test = run_js(\n                '''\n                (async function test() {\n                    await sleep(1000);\n                    return 7;\n                })\n                '''\n            )\n\n            def wrapper():\n                return test().syncify()\n            from ctypes import pythonapi, py_object\n            pythonapi.PyObject_CallNoArgs.argtypes = [py_object]\n            pythonapi.PyObject_CallNoArgs.restype = py_object\n            assert pythonapi.PyObject_CallNoArgs(wrapper) == 7\n        `);\n        ")

@pytest.mark.requires_dynamic_linking
@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_cpp_exceptions_and_syncify(selenium):
    if False:
        return 10
    assert selenium.run_js('\n            ptr = pyodide.runPython(`\n                from pyodide.code import run_js\n                temp = run_js(\n                    \'\'\'\n                    (async function temp() {\n                        await sleep(100);\n                        return 9;\n                    })\n                    \'\'\'\n                )\n\n                def f():\n                    try:\n                        return temp().syncify()\n                    except Exception as e:\n                        print(e)\n                        return -1\n                id(f)\n            `);\n\n            await pyodide.loadPackage("cpp-exceptions-test")\n            const Module = pyodide._module;\n            const catchlib = pyodide._module.LDSO.loadedLibsByName["/usr/lib/cpp-exceptions-test-catch.so"].exports;\n            async function t(x){\n                Module.validSuspender.value = true;\n                const ptr = await Module.createPromising(catchlib.catch_call_pyobj)(x);\n                Module.validSuspender.value = false;\n                const res = Module.UTF8ToString(ptr);\n                Module._free(ptr);\n                return res;\n            }\n            return await t(ptr)\n            ') == 'result was: 9'

@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_two_way_transfer(selenium):
    if False:
        return 10
    res = selenium.run_js('\n        pyodide.runPython(`\n            l = []\n            def f(n, t):\n                from js import sleep\n                for i in range(5):\n                    sleep(t).syncify()\n                    l.append([n, i])\n        `);\n        f = pyodide.globals.get("f");\n        await Promise.all([f.callSyncifying("a", 15), f.callSyncifying("b", 25)])\n        f.destroy();\n        const l = pyodide.globals.get("l");\n        const res = l.toJs();\n        l.destroy();\n        return res;\n        ')
    assert res == [['a', 0], ['b', 0], ['a', 1], ['a', 2], ['b', 1], ['a', 3], ['b', 2], ['a', 4], ['b', 3], ['b', 4]]

@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_sync_async_mix(selenium):
    if False:
        return 10
    res = selenium.run_js('\n        pyodide.runPython(\n        `\n            from js import sleep\n            l = [];\n            async def a(t):\n                await sleep(t)\n                l.append(["a", t])\n\n            def b(t):\n                sleep(t).syncify()\n                l.append(["b", t])\n        `);\n        const a = pyodide.globals.get("a");\n        const b = pyodide.globals.get("b");\n        const l = pyodide.globals.get("l");\n\n        await Promise.all([\n            b.callSyncifying(300),\n            b.callSyncifying(200),\n            b.callSyncifying(250),\n            a(220),\n            a(150),\n            a(270)\n        ]);\n        const res = l.toJs();\n        for(let p of [a, b, l]) {\n            p.destroy();\n        }\n        return res;\n        ')
    assert res == [['a', 150], ['b', 200], ['a', 220], ['b', 250], ['a', 270], ['b', 300]]

@pytest.mark.xfail_browsers(safari='No JSPI on Safari', firefox='No JSPI on firefox')
def test_nested_syncify(selenium):
    if False:
        while True:
            i = 10
    res = selenium.run_js('\n        async function f1() {\n            await sleep(30);\n            return await g1.callSyncifying();\n        }\n        async function f2() {\n            await sleep(30);\n            return await g2.callSyncifying();\n        }\n        async function getStuff() {\n            await sleep(30);\n            return "gotStuff";\n        }\n        pyodide.globals.set("f1", f1);\n        pyodide.globals.set("f2", f2);\n        pyodide.globals.set("getStuff", getStuff);\n\n        pyodide.runPython(`\n            from js import sleep\n            def g():\n                sleep(25).syncify()\n                return f1().syncify()\n\n            def g1():\n                sleep(25).syncify()\n                return f2().syncify()\n\n            def g2():\n                sleep(25).syncify()\n                return getStuff().syncify()\n        `);\n        const l = pyodide.runPython("l = []; l")\n        const g = pyodide.globals.get("g");\n        const g1 = pyodide.globals.get("g1");\n        const g2 = pyodide.globals.get("g2");\n        const p = [];\n        p.push(g.callSyncifying().then((res) => l.append(res)));\n        p.push(pyodide.runPythonSyncifying(`\n            from js import sleep\n            for i in range(20):\n                sleep(9).syncify()\n                l.append(i)\n        `));\n        await Promise.all(p);\n        const res = l.toJs();\n        for(let p of [l, g, g1, g2]) {\n            p.destroy()\n        }\n        return res;\n        ')
    assert 'gotStuff' in res
    del res[res.index('gotStuff')]
    assert res == list(range(20))