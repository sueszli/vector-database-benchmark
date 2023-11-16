from pytest_pyodide.decorator import run_in_pyodide

@run_in_pyodide(packages=['xyzservices'])
def test_xyzservices(selenium):
    if False:
        i = 10
        return i + 15
    '\n    Check whether any errors occur by testing basic functionality.\n    Intended to function as a regression test.\n    Might fail if xyzservices is upgraded and the data\n    or API changes.\n    '
    import xyzservices.providers
    assert xyzservices.providers.CartoDB.Positron.url