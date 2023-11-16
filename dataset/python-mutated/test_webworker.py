import pytest

def test_runwebworker_different_package_name(selenium_webworker_standalone, script_type):
    if False:
        return 10
    selenium = selenium_webworker_standalone
    output = selenium.run_webworker('\n        import pyparsing\n        pyparsing.__version__\n        ')
    assert isinstance(output, str)

def test_runwebworker_no_imports(selenium_webworker_standalone, script_type):
    if False:
        print('Hello World!')
    selenium = selenium_webworker_standalone
    output = selenium.run_webworker('\n        42\n        ')
    assert output == 42

def test_runwebworker_missing_import(selenium_webworker_standalone, script_type):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_webworker_standalone
    msg = 'ModuleNotFoundError'
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run_webworker('\n            import foo\n            ')

def test_runwebworker_exception(selenium_webworker_standalone, script_type):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_webworker_standalone
    msg = 'ZeroDivisionError'
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run_webworker('\n            42 / 0\n            ')

def test_runwebworker_exception_after_import(selenium_webworker_standalone, script_type):
    if False:
        i = 10
        return i + 15
    selenium = selenium_webworker_standalone
    msg = 'ZeroDivisionError'
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run_webworker('\n            import pyparsing\n            42 / 0\n            ')

def test_runwebworker_micropip(selenium_webworker_standalone, script_type):
    if False:
        print('Hello World!')
    selenium = selenium_webworker_standalone
    output = selenium.run_webworker("\n        import micropip\n        await micropip.install('snowballstemmer')\n        import snowballstemmer\n        stemmer = snowballstemmer.stemmer('english')\n        stemmer.stemWords('go goes going gone'.split())[0]\n        ")
    assert output == 'go'