import pytest

@pytest.fixture(scope='session')
def print_info():
    if False:
        while True:
            i = 10
    headings = ['browser', 'py_limit', 'py_usage', 'js_depth', 'py_depth', 'js_depth/py_usage', 'js_depth/py_depth']
    fmt = '## {{:{:d}s}}  {{:{:d}g}}  {{:{:d}.2f}}  {{:{:d}g}}  {{:{:d}g}}  {{:{:d}g}}  {{:{:d}g}}'.format(*map(len, headings))
    printed_heading = False

    def print_info(*args):
        if False:
            print('Hello World!')
        nonlocal printed_heading
        if not printed_heading:
            printed_heading = True
            print('## ' + '  '.join(headings))
        print(fmt.format(*args))
    yield print_info

@pytest.mark.skip_refcount_check
@pytest.mark.skip_pyproxy_check
def test_stack_usage(selenium, print_info):
    if False:
        i = 10
        return i + 15
    res = selenium.run_js('\n        self.measure_available_js_stack_depth = () => {\n            let depth = 0;\n            function recurse() { depth += 1; recurse(); }\n            try { recurse(); } catch (err) { }\n            return depth;\n        };\n        let py_limit = pyodide.runPython("import sys; sys.getrecursionlimit()");\n        self.jsrecurse = function(f, g) {\n            return (n) => (n > 0) ? f(n-1) : g();\n        }\n        let py_usage = pyodide.runPython(`\n            from js import measure_available_js_stack_depth, jsrecurse\n            from pyodide.ffi import create_proxy\n            recurse_proxy = None\n            def recurse(n):\n                return jsrecurse(recurse_proxy, measure_available_js_stack_depth)(n)\n            recurse_proxy = create_proxy(recurse)\n            (recurse(0)-recurse(100))/100\n        `);\n        let js_depth = measure_available_js_stack_depth();\n        self.py_depth = [0];\n        try {\n        pyodide.runPython(`\n        import sys\n        from js import py_depth\n        sys.setrecursionlimit(2000)\n        def infiniterecurse():\n            py_depth[0] += 1\n            infiniterecurse()\n        infiniterecurse()\n        `);\n        } catch {}\n\n        py_depth = py_depth[0];\n        return [\n            py_limit,\n            py_usage,\n            js_depth,\n            py_depth,\n            Math.floor(js_depth/py_usage),\n            Math.floor(js_depth/py_depth),\n        ]\n        ')
    print_info(selenium.browser, *res)
    selenium.clean_logs()