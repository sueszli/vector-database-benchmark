import pytest
from pipenv.cmdparse import Script, ScriptEmptyError

@pytest.mark.run
@pytest.mark.script
def test_parse():
    if False:
        return 10
    script = Script.parse(['python', '-c', "print('hello')"])
    assert script.command == 'python'
    assert script.args == ['-c', "print('hello')"], script

@pytest.mark.run
@pytest.mark.script
def test_parse_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ScriptEmptyError) as e:
        Script.parse('')
    assert str(e.value) == '[]'

@pytest.mark.run
def test_extend():
    if False:
        i = 10
        return i + 15
    script = Script('python', ['-c', "print('hello')"])
    script.extend(['--verbose'])
    assert script.command == 'python'
    assert script.args == ['-c', "print('hello')", '--verbose'], script

@pytest.mark.run
@pytest.mark.script
def test_cmdify():
    if False:
        while True:
            i = 10
    script = Script('python', ['-c', "print('hello world')"])
    cmd = script.cmdify()
    assert cmd == 'python -c "print(\'hello world\')"', script

@pytest.mark.run
@pytest.mark.script
def test_cmdify_complex():
    if False:
        for i in range(10):
            print('nop')
    script = Script.parse(' '.join(['"C:\\Program Files\\Python36\\python.exe" -c', ' "print(\'Double quote: \\"\')" '.strip()]))
    assert script.cmdify() == ' '.join(['"C:\\Program Files\\Python36\\python.exe"', '-c', ' "print(\'Double quote: \\"\')" '.strip()]), script

@pytest.mark.run
@pytest.mark.script
def test_cmdify_quote_if_paren_in_command():
    if False:
        return 10
    'Ensure ONLY the command is quoted if it contains parentheses.\n    '
    script = Script.parse('"C:\\Python36(x86)\\python.exe" -c print(123)')
    assert script.cmdify() == '"C:\\Python36(x86)\\python.exe" -c print(123)', script

@pytest.mark.run
@pytest.mark.script
def test_cmdify_quote_if_carets():
    if False:
        return 10
    'Ensure arguments are quoted if they contain carets.\n    '
    script = Script('foo^bar', ['baz^rex'])
    assert script.cmdify() == '"foo^bar" "baz^rex"', script