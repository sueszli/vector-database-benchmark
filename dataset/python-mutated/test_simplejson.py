from pytest_pyodide.decorator import run_in_pyodide

@run_in_pyodide(packages=['simplejson'])
def test_simplejson(selenium):
    if False:
        return 10
    from decimal import Decimal
    import simplejson
    import simplejson._speedups
    dumped = simplejson.dumps({'c': 0, 'b': 0, 'a': 0}, sort_keys=True)
    expected = '{"a": 0, "b": 0, "c": 0}'
    assert dumped == expected
    assert simplejson.loads('1.1', use_decimal=True) == Decimal('1.1')