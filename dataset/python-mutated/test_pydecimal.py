from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['test', 'pydecimal'], pytest_assert_rewrites=False)
def test_pydecimal(selenium):
    if False:
        while True:
            i = 10
    from test import libregrtest
    name = 'test_decimal'
    ignore_tests = ['test_context_subclassing', 'test_none_args', 'test_threading']
    try:
        libregrtest.main([name], ignore_tests=ignore_tests, verbose=True, verbose3=True)
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f'Failed with code: {e.code}') from None