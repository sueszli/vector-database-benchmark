"""Tests involving running Xonsh in subproc.
This requires Xonsh installed in venv or otherwise available on PATH
"""
import os
import shutil
import subprocess as sp
import tempfile
from pathlib import Path
import pytest
import xonsh
from xonsh.dirstack import with_pushd
from xonsh.pytest.tools import ON_DARWIN, ON_TRAVIS, ON_WINDOWS, skip_if_on_darwin, skip_if_on_msys, skip_if_on_unix, skip_if_on_windows
PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'bin') + os.pathsep + os.environ['PATH']
skip_if_no_xonsh = pytest.mark.skipif(shutil.which('xonsh') is None, reason='xonsh not on PATH')
skip_if_no_make = pytest.mark.skipif(shutil.which('make') is None, reason='make command not on PATH')
skip_if_no_sleep = pytest.mark.skipif(shutil.which('sleep') is None, reason='sleep command not on PATH')

def run_xonsh(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, single_command=False, interactive=False, path=None):
    if False:
        i = 10
        return i + 15
    env = dict(os.environ)
    if path is None:
        env['PATH'] = PATH
    else:
        env['PATH'] = path
    env['XONSH_DEBUG'] = '0'
    env['XONSH_SHOW_TRACEBACK'] = '1'
    env['RAISE_SUBPROC_ERROR'] = '0'
    env['FOREIGN_ALIASES_SUPPRESS_SKIP_MESSAGE'] = '1'
    env['PROMPT'] = ''
    xonsh = shutil.which('xonsh', path=PATH)
    args = [xonsh, '--no-rc']
    if interactive:
        args.append('-i')
    if single_command:
        args += ['-c', cmd]
        input = None
    else:
        input = cmd
    proc = sp.Popen(args, env=env, stdin=stdin, stdout=stdout, stderr=stderr, universal_newlines=True)
    try:
        (out, err) = proc.communicate(input=input, timeout=20)
    except sp.TimeoutExpired:
        proc.kill()
        raise
    return (out, err, proc.returncode)

def check_run_xonsh(cmd, fmt, exp, exp_rtn=0):
    if False:
        while True:
            i = 10
    'The ``fmt`` parameter is a function\n    that formats the output of cmd, can be None.\n    '
    (out, err, rtn) = run_xonsh(cmd, stderr=sp.PIPE)
    if callable(fmt):
        out = fmt(out)
    if callable(exp):
        exp = exp()
    assert out == exp, err
    assert rtn == exp_rtn, err
ALL_PLATFORMS = [("\ndef _f():\n    print('hello')\n\naliases['f'] = _f\nf\n", 'hello\n', 0), ("\ndef _f():\n    print('Wow Mom!')\n\naliases['f'] = _f\nf > tttt\n\nwith open('tttt') as tttt:\n    s = tttt.read().strip()\nprint('REDIRECTED OUTPUT: ' + s)\n", 'REDIRECTED OUTPUT: Wow Mom!\n', 0), ("\ndef _f(args, stdin, stdout, stderr):\n    print('The Truth is Out There', file=stderr)\n\naliases['f'] = _f\nf e>o\n", 'The Truth is Out There\n', 0), ("\nimport sys\ndef _f():\n    sys.exit(42)\n\naliases['f'] = _f\nprint(![f].returncode)\n", '42\n', 0), ("\ndef _test_stream(args, stdin, stdout, stderr):\n    print('hallo on stream', file=stderr)\n    print('hallo on stream', file=stdout)\n    return 1\n\naliases['test-stream'] = _test_stream\nx = ![test-stream]\nprint(x.returncode)\n", 'hallo on stream\nhallo on stream\n1\n', 0), ("\ndef _test_stream(args, stdin, stdout, stderr):\n    print('hallo on err', file=stderr)\n    print('hallo on out', file=stdout)\n    return 1\n\naliases['test-stream'] = _test_stream\nx = !(test-stream)\nprint(x.returncode)\n", '1\n', 0), ("\ndef _test_stream(args, stdin, stdout, stderr):\n    print('hallo on err', file=stderr)\n    print('hallo on out', file=stdout)\n    return 1\n\naliases['test-stream'] = _test_stream\nwith __xonsh__.env.swap(XONSH_SUBPROC_CAPTURED_PRINT_STDERR=True):\n    x = !(test-stream)\n    print(x.returncode)\n", 'hallo on err\n1\n', 0), ("\ndef dummy(args, inn, out, err):\n    out.write('hey!')\n    return 0\n\ndef dummy2(args, inn, out, err):\n    s = inn.read()\n    out.write(s.upper())\n    return 0\n\naliases['d'] = dummy\naliases['d2'] = dummy2\nd | d2\n", 'HEY!', 0), ("\ndef _g(args, stdin=None):\n    for i in range(1000):\n        print('x' * 100)\n\naliases['g'] = _g\ng\n", ('x' * 100 + '\n') * 1000, 0), ('\nwith open(\'tttt\', \'w\') as fp:\n    fp.write("Wow mom!\\n")\n\n![python tests/bin/cat tttt | python tests/bin/wc]\n', ' 1  2 10 <stdin>\n' if ON_WINDOWS else ' 1  2 9 <stdin>\n', 0), ('\nwith open(\'tttt\', \'w\') as fp:\n    fp.write("Wow mom!\\n")\n\n![python tests/bin/cat tttt | python tests/bin/wc | python tests/bin/wc]\n', ' 1  4 18 <stdin>\n' if ON_WINDOWS else ' 1  4 16 <stdin>\n', 0), ("\nfrom xonsh.tools import unthreadable\n\n@unthreadable\ndef _f():\n    return 'hello\\n'\n\naliases['f'] = _f\nf\n", 'hello\n', 0), ("\nimport os\n\ndef _echo(args):\n    print(' '.join(args))\naliases['echo'] = _echo\n\nfiles = ['Actually_test.tst', 'Actually.tst', 'Complete_test.tst', 'Complete.tst']\n\n# touch the file\nfor f in files:\n    with open(f, 'w'):\n        pass\n\n# echo the files\necho *.tst and echo *_test.tst\necho *_test.tst\necho *_test.tst and echo *.tst\n\n# remove the files\nfor f in files:\n    os.remove(f)\n", 'Actually.tst Actually_test.tst Complete.tst Complete_test.tst\nActually_test.tst Complete_test.tst\nActually_test.tst Complete_test.tst\nActually_test.tst Complete_test.tst\nActually.tst Actually_test.tst Complete.tst Complete_test.tst\n', 0), ("\ndef _echo(args):\n    print(' '.join(args))\naliases['echo'] = _echo\n\necho --option1 \\\n--option2\necho missing \\\nEOL", '--option1 --option2\nmissing EOL\n', 0), ("\naliases['ls'] = 'spam spam sausage spam'\n\necho @$(which ls)\n", 'spam spam sausage spam\n', 0), ("\ndef _echo(args):\n    print(' '.join(args))\naliases['echo'] = _echo\n\necho foo_@$(echo spam)_bar\n", 'foo_spam_bar\n', 0), ("\ndef _echo(args):\n    print(' '.join(args))\naliases['echo'] = _echo\n\necho foo_@$(echo spam sausage)_bar\n", 'foo_spam_bar foo_sausage_bar\n', 0), ('\necho Just the place for a snark. >tttt\npython tests/bin/cat tttt\n', 'Just the place for a snark.\n', 0), ('\ndef _f():\n    def j():\n        pass\n\n    global aliases\n    aliases[\'j\'] = j\n\n    def completions(pref, *args):\n        return set([\'hello\', \'world\'])\n\n    completer add j completions "start"\n\n\n_f()\ndel _f\n\n', '', 0), ('\ndef _echo(args):\n    print(\' \'.join(args))\naliases[\'echo\'] = _echo\n\nfrom xonsh.lib.subprocess import check_output\n\nprint(check_output(["echo", "hello"]).decode("utf8"))\n', 'hello\n\n', 0), ('\nimport sys\n\nif sys.version_info[:2] >= (3, 7):\n    with open("sourced-file.xsh", "w") as f:\n        f.write(\'\'\'\nfrom contextvars import ContextVar\n\nvar = ContextVar(\'var\', default=\'spam\')\nvar.set(\'foo\')\n        \'\'\')\n\n    source sourced-file.xsh\n\n    print("Var " + var.get())\n\n    import os\n    os.remove(\'sourced-file.xsh\')\nelse:\n    print("Var foo")\n', 'Var foo\n', 0), ("\ndef _echo(args):\n    print(' '.join(args))\naliases['echo'] = _echo\n\necho --version and echo a\necho --version && echo a\necho --version or echo a\necho --version || echo a\necho -+version and echo a\necho -+version && echo a\necho -+version or echo a\necho -+version || echo a\necho -~version and echo a\necho -~version && echo a\necho -~version or echo a\necho -~version || echo a\n", '--version\na\n--version\na\n--version\n--version\n-+version\na\n-+version\na\n-+version\n-+version\n-~version\na\n-~version\na\n-~version\n-~version\n', 0)]
UNIX_TESTS = [("\ndef _echo():\n    echo hello\n\naliases['echo'] = _echo\necho\n", 'hello\n', 0), ('\naliases[\'echo\'] = "echo @(\'hello\')"\necho\n', 'hello\n', 0), ('\naliases[\'first\'] = "second @(1)"\naliases[\'second\'] = "first @(1)"\nfirst\n', lambda out: 'Recursive calls to "first" alias.' in out, 0), ('\nfrom time import sleep\naliases[\'a\'] = lambda: print(1, end="") or sleep(0.2) or print(1, end="")\naliases[\'b\'] = \'a\'\na | a\na | a\na | b | a\na | a | b | b\n', '1' * 2 * 4, 0), ('\n# test parsing of $SHLVL\n\n$SHLVL = "1"\necho $SHLVL # == 1\n\n$SHLVL = 1\necho $SHLVL # == 1\n\n$SHLVL = "-13"\necho $SHLVL # == 0\n\n$SHLVL = "error"\necho $SHLVL # == 0\n\n$SHLVL = 999\necho $SHLVL # == 999\n\n$SHLVL = 1000\necho $SHLVL # == 1\n\n# sourcing a script should maintain $SHLVL\n\n$SHLVL = 5\ntouch temp_shlvl_test.sh\nsource-bash temp_shlvl_test.sh\nrm temp_shlvl_test.sh\necho $SHLVL # == 5\n\n# creating a subshell should increment the child\'s $SHLVL and maintain the parents $SHLVL\n\n$SHLVL = 5\nxonsh -c r\'echo $SHLVL\' # == 6\necho $SHLVL # == 5\n\n# replacing the current process with another process should derease $SHLVL\n# (so that if the new process is a shell, $SHLVL is maintained)\n\n$SHLVL = 5\nxexec python3 -c \'import os; print(os.environ["SHLVL"])\' # == 4\n', '1\n1\n0\n0\n999\n1\n5\n6\n5\n4\n', 0), ('\ndef _callme(args):\n    result = $(python -c \'print("tree");print("car")\')\n    print(result[::-1])\n    print(\'one\\ntwo\\nthree\')\n\naliases[\'callme\'] = _callme\ncallme | grep t\n', 'eert\ntwo\nthree\n', 0), ('\ndef _callme(args):\n    python -c \'print("tree");print("car")\'\n    print(\'one\\ntwo\\nthree\')\n\naliases[\'callme\'] = _callme\ncallme | grep t\n', 'tree\ntwo\nthree\n', 0), pytest.param(('\ndef _callme(args):\n    $[python -c \'print("tree");print("car")\']\n    print(\'one\\ntwo\\nthree\')\n\naliases[\'callme\'] = _callme\ncallme | grep t\n', 'tree\ntwo\nthree\n', 0), marks=pytest.mark.xfail(reason='$[] does not send stdout through the pipe'))]
if not ON_WINDOWS:
    ALL_PLATFORMS = tuple(ALL_PLATFORMS) + tuple(UNIX_TESTS)

@skip_if_no_xonsh
@pytest.mark.parametrize('case', ALL_PLATFORMS)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_script(case):
    if False:
        print('Hello World!')
    (script, exp_out, exp_rtn) = case
    (out, err, rtn) = run_xonsh(script)
    if callable(exp_out):
        assert exp_out(out)
    else:
        assert exp_out == out
    assert exp_rtn == rtn
ALL_PLATFORMS_STDERR = [("\ndef _f(args, stdin, stdout):\n    print('Wow Mom!', file=stdout)\n\naliases['f'] = _f\nf o>e\n", 'Wow Mom!\n', 0)]

@skip_if_no_xonsh
@pytest.mark.parametrize('case', ALL_PLATFORMS_STDERR)
def test_script_stderr(case):
    if False:
        print('Hello World!')
    (script, exp_err, exp_rtn) = case
    (out, err, rtn) = run_xonsh(script, stderr=sp.PIPE)
    assert exp_err == err
    assert exp_rtn == rtn

@skip_if_no_xonsh
@skip_if_on_windows
@pytest.mark.parametrize('cmd, fmt, exp', [('pwd', None, lambda : os.getcwd() + '\n'), ('echo WORKING', None, 'WORKING\n'), ('ls -f', lambda out: out.splitlines().sort(), os.listdir().sort()), ("$FOO='foo' $BAR=2 xonsh -c r'echo -n $FOO$BAR'", None, 'foo2')])
def test_single_command_no_windows(cmd, fmt, exp):
    if False:
        i = 10
        return i + 15
    check_run_xonsh(cmd, fmt, exp)

@skip_if_no_xonsh
def test_eof_syntax_error():
    if False:
        while True:
            i = 10
    'Ensures syntax errors for EOF appear on last line.'
    script = 'x = 1\na = (1, 0\n'
    (out, err, rtn) = run_xonsh(script, stderr=sp.PIPE)
    assert 'line 0' not in err
    assert 'EOF in multi-line statement' in err and 'line 2' in err

@skip_if_no_xonsh
def test_open_quote_syntax_error():
    if False:
        while True:
            i = 10
    script = '#!/usr/bin/env xonsh\n\necho "This is line 3"\nprint ("This is line 4")\nx = "This is a string where I forget the closing quote on line 5\necho "This is line 6"\n'
    (out, err, rtn) = run_xonsh(script, stderr=sp.PIPE)
    assert '(\'code: "This is line 3"\',)' not in err
    assert 'line 5' in err
    assert 'SyntaxError:' in err
_bad_case = pytest.mark.skipif(ON_DARWIN or ON_WINDOWS or ON_TRAVIS, reason='bad platforms')

@skip_if_no_xonsh
def test_atdollar_no_output():
    if False:
        return 10
    script = "\ndef _echo(args):\n    print(' '.join(args))\naliases['echo'] = _echo\n@$(echo)\n"
    (out, err, rtn) = run_xonsh(script, stderr=sp.PIPE)
    assert 'command is empty' in err

@skip_if_no_xonsh
def test_empty_command():
    if False:
        for i in range(10):
            print('nop')
    script = "$['']\n"
    (out, err, rtn) = run_xonsh(script, stderr=sp.PIPE)
    assert 'command is empty' in err

@skip_if_no_xonsh
@_bad_case
def test_printfile():
    if False:
        for i in range(10):
            print('nop')
    check_run_xonsh('printfile.xsh', None, 'printfile.xsh\n')

@skip_if_no_xonsh
@_bad_case
def test_printname():
    if False:
        print('Hello World!')
    check_run_xonsh('printfile.xsh', None, 'printfile.xsh\n')

@skip_if_no_xonsh
@_bad_case
def test_sourcefile():
    if False:
        for i in range(10):
            print('nop')
    check_run_xonsh('printfile.xsh', None, 'printfile.xsh\n')

@skip_if_no_xonsh
@_bad_case
@pytest.mark.parametrize('cmd, fmt, exp', [('\nwith open(\'tttt\', \'w\') as fp:\n    fp.write("Wow mom!\\n")\n\n(wc) < tttt\n', None, ' 1  2 9 <stdin>\n'), ('\nwith open(\'tttt\', \'w\') as fp:\n    fp.write("Wow mom!\\n")\n\n(wc;) < tttt\n', None, ' 1  2 9 <stdin>\n')])
def test_subshells(cmd, fmt, exp):
    if False:
        for i in range(10):
            print('nop')
    check_run_xonsh(cmd, fmt, exp)

@skip_if_no_xonsh
@skip_if_on_windows
@pytest.mark.parametrize('cmd, exp', [('pwd', lambda : os.getcwd() + '\n')])
def test_redirect_out_to_file(cmd, exp, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    outfile = tmpdir.mkdir('xonsh_test_dir').join('xonsh_test_file')
    command = f'{cmd} > {outfile}\n'
    (out, _, _) = run_xonsh(command)
    content = outfile.read()
    if callable(exp):
        exp = exp()
    assert content == exp

@skip_if_no_make
@skip_if_no_xonsh
@skip_if_no_sleep
@skip_if_on_windows
@pytest.mark.xfail(strict=False)
def test_xonsh_no_close_fds():
    if False:
        while True:
            i = 10
    makefile = 'default: all\nall:\n\t$(MAKE) s\ns:\n\t$(MAKE) a b\na:\n\tsleep 1\nb:\n\tsleep 1\n'
    with tempfile.TemporaryDirectory() as d, with_pushd(d):
        with open('Makefile', 'w') as f:
            f.write(makefile)
        out = sp.check_output(['make', '-sj2', 'SHELL=xonsh'], universal_newlines=True)
        assert 'warning' not in out

@skip_if_no_xonsh
@pytest.mark.parametrize('cmd, fmt, exp', [('cat tttt | wc', lambda x: x > '', True)])
def test_pipe_between_subprocs(cmd, fmt, exp):
    if False:
        i = 10
        return i + 15
    "verify pipe between subprocesses doesn't throw an exception"
    check_run_xonsh(cmd, fmt, exp)

@skip_if_no_xonsh
@skip_if_on_windows
def test_negative_exit_codes_fail():
    if False:
        for i in range(10):
            print('nop')
    script = 'python -c "import os; os.abort()" && echo OK\n'
    (out, err, rtn) = run_xonsh(script)
    assert 'OK' != out
    assert 'OK' != err

@skip_if_no_xonsh
@pytest.mark.parametrize('cmd, exp', [("echo '&'", '&\n'), ("echo foo'&'", "foo'&'\n"), ("echo foo '&'", 'foo &\n'), ("echo foo '&' bar", 'foo & bar\n')])
def test_ampersand_argument(cmd, exp):
    if False:
        print('Hello World!')
    script = f"\n#!/usr/bin/env xonsh\ndef _echo(args):\n    print(' '.join(args))\naliases['echo'] = _echo\n{cmd}\n"
    (out, _, _) = run_xonsh(script)
    assert out == exp

@skip_if_no_xonsh
@skip_if_on_windows
@pytest.mark.parametrize('cmd, exp_rtn', [('import sys; sys.exit(0)', 0), ('import sys; sys.exit(100)', 100), ("sh -c 'exit 0'", 0), ("sh -c 'exit 1'", 1)])
def test_single_command_return_code(cmd, exp_rtn):
    if False:
        while True:
            i = 10
    (_, _, rtn) = run_xonsh(cmd, single_command=True)
    assert rtn == exp_rtn

@skip_if_no_xonsh
@skip_if_on_msys
@skip_if_on_windows
@skip_if_on_darwin
def test_argv0():
    if False:
        for i in range(10):
            print('nop')
    check_run_xonsh('checkargv0.xsh', None, 'OK\n')

@pytest.mark.parametrize('interactive', [True, False])
def test_loading_correctly(monkeypatch, interactive):
    if False:
        return 10
    monkeypatch.setenv('SHELL_TYPE', 'prompt_toolkit')
    monkeypatch.setenv('XONSH_LOGIN', '1')
    monkeypatch.setenv('XONSH_INTERACTIVE', '1')
    (out, err, ret) = run_xonsh('import xonsh; echo -n AAA @(xonsh.__file__) BBB', interactive=interactive, single_command=True)
    assert not err
    assert ret == 0
    our_xonsh = xonsh.__file__
    assert f'AAA {our_xonsh} BBB' in out

@skip_if_no_xonsh
@pytest.mark.parametrize('cmd', ['x = 0; (lambda: x)()', 'x = 0; [x for _ in [0]]'])
def test_exec_function_scope(cmd):
    if False:
        return 10
    (_, _, rtn) = run_xonsh(cmd, single_command=True)
    assert rtn == 0

@skip_if_on_unix
def test_run_currentfolder(monkeypatch):
    if False:
        print('Hello World!')
    'Ensure we can run an executable in the current folder\n    when file is not on path\n    '
    batfile = Path(__file__).parent / 'bin' / 'hello_world.bat'
    monkeypatch.chdir(batfile.parent)
    cmd = batfile.name
    (out, _, _) = run_xonsh(cmd, stdout=sp.PIPE, stderr=sp.PIPE, path=os.environ['PATH'])
    assert out.strip() == 'hello world'

@skip_if_on_unix
def test_run_dynamic_on_path():
    if False:
        while True:
            i = 10
    'Ensure we can run an executable which is added to the path\n    after xonsh is loaded\n    '
    batfile = Path(__file__).parent / 'bin' / 'hello_world.bat'
    cmd = f"$PATH.add(r'{batfile.parent}');![hello_world.bat]"
    (out, _, _) = run_xonsh(cmd, path=os.environ['PATH'])
    assert out.strip() == 'hello world'

@skip_if_on_unix
def test_run_fail_not_on_path():
    if False:
        i = 10
        return i + 15
    'Test that xonsh fails to run an executable when not on path\n    or in current folder\n    '
    cmd = 'hello_world.bat'
    (out, _, _) = run_xonsh(cmd, stdout=sp.PIPE, stderr=sp.PIPE, path=os.environ['PATH'])
    assert out != 'Hello world'