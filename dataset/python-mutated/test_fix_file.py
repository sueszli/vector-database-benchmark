import pytest
import os
from collections import namedtuple
from thefuck.rules.fix_file import match, get_new_command
from thefuck.types import Command
FixFileTest = namedtuple('FixFileTest', ['script', 'file', 'line', 'col', 'output'])
tests = (FixFileTest('gcc a.c', 'a.c', 3, 1, "\na.c: In function 'main':\na.c:3:1: error: expected expression before '}' token\n }\n  ^\n"), FixFileTest('clang a.c', 'a.c', 3, 1, '\na.c:3:1: error: expected expression\n}\n^\n'), FixFileTest('perl a.pl', 'a.pl', 3, None, '\nsyntax error at a.pl line 3, at EOF\nExecution of a.pl aborted due to compilation errors.\n'), FixFileTest('perl a.pl', 'a.pl', 2, None, '\nSearch pattern not terminated at a.pl line 2.\n'), FixFileTest('sh a.sh', 'a.sh', 2, None, '\na.sh: line 2: foo: command not found\n'), FixFileTest('zsh a.sh', 'a.sh', 2, None, '\na.sh:2: command not found: foo\n'), FixFileTest('bash a.sh', 'a.sh', 2, None, '\na.sh: line 2: foo: command not found\n'), FixFileTest('rustc a.rs', 'a.rs', 2, 5, '\na.rs:2:5: 2:6 error: unexpected token: `+`\na.rs:2     +\n           ^\n'), FixFileTest('cargo build', 'src/lib.rs', 3, 5, '\n   Compiling test v0.1.0 (file:///tmp/fix-error/test)\n   src/lib.rs:3:5: 3:6 error: unexpected token: `+`\n   src/lib.rs:3     +\n                    ^\nCould not compile `test`.\n\nTo learn more, run the command again with --verbose.\n'), FixFileTest('python a.py', 'a.py', 2, None, '\n  File "a.py", line 2\n      +\n          ^\nSyntaxError: invalid syntax\n'), FixFileTest('python a.py', 'a.py', 8, None, '\nTraceback (most recent call last):\n  File "a.py", line 8, in <module>\n    match("foo")\n  File "a.py", line 5, in match\n    m = re.search(None, command)\n  File "/usr/lib/python3.4/re.py", line 170, in search\n    return _compile(pattern, flags).search(string)\n  File "/usr/lib/python3.4/re.py", line 293, in _compile\n    raise TypeError("first argument must be string or compiled pattern")\nTypeError: first argument must be string or compiled pattern\n'), FixFileTest(u'python café.py', u'café.py', 8, None, u'\nTraceback (most recent call last):\n  File "café.py", line 8, in <module>\n    match("foo")\n  File "café.py", line 5, in match\n    m = re.search(None, command)\n  File "/usr/lib/python3.4/re.py", line 170, in search\n    return _compile(pattern, flags).search(string)\n  File "/usr/lib/python3.4/re.py", line 293, in _compile\n    raise TypeError("first argument must be string or compiled pattern")\nTypeError: first argument must be string or compiled pattern\n'), FixFileTest('ruby a.rb', 'a.rb', 3, None, '\na.rb:3: syntax error, unexpected keyword_end\n'), FixFileTest('lua a.lua', 'a.lua', 2, None, "\nlua: a.lua:2: unexpected symbol near '+'\n"), FixFileTest('fish a.sh', '/tmp/fix-error/a.sh', 2, None, "\nfish: Unknown command 'foo'\n/tmp/fix-error/a.sh (line 2): foo\n                              ^\n"), FixFileTest('./a', './a', 2, None, '\nawk: ./a:2: BEGIN { print "Hello, world!" + }\nawk: ./a:2:                                 ^ syntax error\n'), FixFileTest('llc a.ll', 'a.ll', 1, 2, '\nllc: a.ll:1:2: error: expected top-level entity\n+\n^\n'), FixFileTest('go build a.go', 'a.go', 1, 2, "\ncan't load package:\na.go:1:2: expected 'package', found '+'\n"), FixFileTest('make', 'Makefile', 2, None, "\nbidule\nmake: bidule: Command not found\nMakefile:2: recipe for target 'target' failed\nmake: *** [target] Error 127\n"), FixFileTest('git st', '/home/martin/.config/git/config', 1, None, '\nfatal: bad config file line 1 in /home/martin/.config/git/config\n'), FixFileTest('node fuck.js asdf qwer', '/Users/pablo/Workspace/barebones/fuck.js', '2', 5, '\n/Users/pablo/Workspace/barebones/fuck.js:2\nconole.log(arg);  // this should read console.log(arg);\n^\nReferenceError: conole is not defined\n    at /Users/pablo/Workspace/barebones/fuck.js:2:5\n    at Array.forEach (native)\n    at Object.<anonymous> (/Users/pablo/Workspace/barebones/fuck.js:1:85)\n    at Module._compile (module.js:460:26)\n    at Object.Module._extensions..js (module.js:478:10)\n    at Module.load (module.js:355:32)\n    at Function.Module._load (module.js:310:12)\n    at Function.Module.runMain (module.js:501:10)\n    at startup (node.js:129:16)\n    at node.js:814:3\n'), FixFileTest('pep8', './tests/rules/test_systemctl.py', 17, 80, '\n./tests/rules/test_systemctl.py:17:80: E501 line too long (93 > 79 characters)\n./tests/rules/test_systemctl.py:18:80: E501 line too long (103 > 79 characters)\n./tests/rules/test_whois.py:20:80: E501 line too long (89 > 79 characters)\n./tests/rules/test_whois.py:22:80: E501 line too long (83 > 79 characters)\n'), FixFileTest('pytest', '/home/thefuck/tests/rules/test_fix_file.py', 218, None, '\nmonkeypatch = <_pytest.monkeypatch.monkeypatch object at 0x7fdb76a25b38>\ntest = (\'fish a.sh\', \'/tmp/fix-error/a.sh\', 2, None, \'\', "\\nfish: Unknown command \'foo\'\\n/tmp/fix-error/a.sh (line 2): foo\\n                              ^\\n")\n\n    @pytest.mark.parametrize(\'test\', tests)\n    @pytest.mark.usefixtures(\'no_memoize\')\n    def test_get_new_command(monkeypatch, test):\n>       mocker.patch(\'os.path.isfile\', return_value=True)\nE       NameError: name \'mocker\' is not defined\n\n/home/thefuck/tests/rules/test_fix_file.py:218: NameError\n'))

@pytest.mark.parametrize('test', tests)
@pytest.mark.usefixtures('no_memoize')
def test_match(mocker, monkeypatch, test):
    if False:
        while True:
            i = 10
    mocker.patch('os.path.isfile', return_value=True)
    monkeypatch.setenv('EDITOR', 'dummy_editor')
    assert match(Command('', test.output))

@pytest.mark.parametrize('test', tests)
@pytest.mark.usefixtures('no_memoize')
def test_no_editor(mocker, monkeypatch, test):
    if False:
        print('Hello World!')
    mocker.patch('os.path.isfile', return_value=True)
    if 'EDITOR' in os.environ:
        monkeypatch.delenv('EDITOR')
    assert not match(Command('', test.output))

@pytest.mark.parametrize('test', tests)
@pytest.mark.usefixtures('no_memoize')
def test_not_file(mocker, monkeypatch, test):
    if False:
        print('Hello World!')
    mocker.patch('os.path.isfile', return_value=False)
    monkeypatch.setenv('EDITOR', 'dummy_editor')
    assert not match(Command('', test.output))

@pytest.mark.parametrize('test', tests)
@pytest.mark.usefixtures('no_memoize')
def test_get_new_command(mocker, monkeypatch, test):
    if False:
        print('Hello World!')
    mocker.patch('os.path.isfile', return_value=True)
    monkeypatch.setenv('EDITOR', 'dummy_editor')

@pytest.mark.parametrize('test', tests)
@pytest.mark.usefixtures('no_memoize')
def test_get_new_command_with_settings(mocker, monkeypatch, test, settings):
    if False:
        i = 10
        return i + 15
    mocker.patch('os.path.isfile', return_value=True)
    monkeypatch.setenv('EDITOR', 'dummy_editor')
    cmd = Command(test.script, test.output)
    settings.fixcolcmd = '{editor} {file} +{line}:{col}'
    if test.col:
        assert get_new_command(cmd) == u'dummy_editor {} +{}:{} && {}'.format(test.file, test.line, test.col, test.script)
    else:
        assert get_new_command(cmd) == u'dummy_editor {} +{} && {}'.format(test.file, test.line, test.script)