import pytest

def test_print(selenium_esm):
    if False:
        print('Hello World!')
    selenium_esm.run("print('This should be logged')")
    assert 'This should be logged' in selenium_esm.logs.splitlines()

@pytest.mark.xfail_browsers(node='No window in node')
def test_import_js(selenium_esm):
    if False:
        while True:
            i = 10
    result = selenium_esm.run("\n        import js\n        js.window.title = 'Foo'\n        js.window.title\n        ")
    assert result == 'Foo'
    result = selenium_esm.run('\n        dir(js)\n        ')
    assert len(result) > 100
    assert 'document' in result
    assert 'window' in result