pytest_plugins = 'pytester'
func_with_doctest = "\ndef hi():\n    '''\n    >>> i = 5\n    >>> i-1\n    4\n    '''\n"

def test_can_run_doctests(testdir):
    if False:
        print('Hello World!')
    script = testdir.makepyfile(func_with_doctest)
    result = testdir.runpytest(script, '--doctest-modules')
    assert result.ret == 0