import pytest

@pytest.mark.requires_dynamic_linking
def test_uncaught_cpp_exceptions(selenium):
    if False:
        print('Hello World!')
    assert selenium.run_js('\n            await pyodide.loadPackage("cpp-exceptions-test");\n            const Tests = pyodide._api.tests;\n            const throwlib = pyodide._module.LDSO.loadedLibsByName["/usr/lib/cpp-exceptions-test-throw.so"].exports;\n            function t(x){\n                try {\n                    throwlib.throw_exc(x);\n                } catch(e){\n                    let errString = Tests.convertCppException(e).toString();\n                    errString = errString.replace(/[0-9]+/, "xxx");\n                    return errString;\n                }\n            }\n            return [t(1), t(2), t(3), t(4), t(5)];\n            ') == ['CppException int: The exception is an object of type int at address xxx which does not inherit from std::exception', 'CppException char: The exception is an object of type char at address xxx which does not inherit from std::exception', 'CppException std::runtime_error: abc', 'CppException myexception: My exception happened', 'CppException char const*: The exception is an object of type char const* at address xxx which does not inherit from std::exception']

@pytest.mark.requires_dynamic_linking
def test_cpp_exception_catching(selenium):
    if False:
        print('Hello World!')
    assert selenium.run_js('\n            await pyodide.loadPackage("cpp-exceptions-test");\n            const Module = pyodide._module;\n            const catchlib = pyodide._module.LDSO.loadedLibsByName["/usr/lib/cpp-exceptions-test-catch.so"].exports;\n            function t(x){\n                const ptr = catchlib.catch_exc(x);\n                const res = Module.UTF8ToString(ptr);\n                Module._free(ptr);\n                return res;\n            }\n\n            return [t(1), t(2), t(3), t(5)];\n            ') == ['caught int 1000', 'caught char 99', 'caught runtime_error abc', 'caught ????']

@pytest.mark.requires_dynamic_linking
def test_sjlj(selenium):
    if False:
        print('Hello World!')
    assert selenium.run_js('\n                await pyodide.loadPackage("cpp-exceptions-test");\n                const Module = pyodide._module;\n                const catchlib = pyodide._module.LDSO.loadedLibsByName["/usr/lib/cpp-exceptions-test-catch.so"].exports;\n                return catchlib.set_jmp_func();\n                ') == 5