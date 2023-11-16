import builtins
import os
import platform
import re
import shlex
import subprocess
from importlib.util import cache_from_source
from pathlib import Path
import pytest
from hy._compat import PY3_9, PYODIDE
if PYODIDE:
    pytest.skip('`subprocess.Popen` not implemented on Pyodide', allow_module_level=True)

def pyr(s=''):
    if False:
        return 10
    return 'hy --repl-output-fn=repr ' + s

def run_cmd(cmd, stdin_data=None, expect=0, dontwritebytecode=False, cwd=None, stdout=subprocess.PIPE):
    if False:
        for i in range(10):
            print('nop')
    env = dict(os.environ)
    if dontwritebytecode:
        env['PYTHONDONTWRITEBYTECODE'] = '1'
    else:
        env.pop('PYTHONDONTWRITEBYTECODE', None)
    env['PYTHONPATH'] = str(Path().resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(shlex.split(cmd) if isinstance(cmd, str) else cmd, input=stdin_data, stdout=stdout, stderr=subprocess.PIPE, universal_newlines=True, shell=False, env=env, cwd=cwd)
    assert result.returncode == expect
    return (result.stdout, result.stderr)

def rm(fpath):
    if False:
        return 10
    try:
        os.remove(fpath)
    except OSError:
        try:
            os.rmdir(fpath)
        except OSError:
            pass

def test_simple():
    if False:
        return 10
    run_cmd('hy', '')

def test_stdin():
    if False:
        return 10
    code = '(+ "P" "Q")\n(print (+ "R" "S"))\n(+ "T" "U")'
    (out, _) = run_cmd('hy', code)
    assert 'PQ' not in out
    assert 'RS' in out
    assert 'TU' not in out
    (out, _) = run_cmd('hy -i', code)
    assert 'PQ' in out
    assert 'RS' in out
    assert 'TU' in out

def test_error_parts_length():
    if False:
        i = 10
        return i + 15
    'Confirm that exception messages print arrows surrounding the affected\n    expression.'
    prg_str = '\n    (import hy.errors\n            hy.importer [read-many])\n\n    (setv test-expr (read-many "(+ 1\n\n\'a 2 3\n\n 1)"))\n    (setv test-expr.start-line {})\n    (setv test-expr.end-line {})\n    (setv test-expr.start-column {})\n    (setv test-expr.end-column {})\n\n    (raise (hy.errors.HyLanguageError\n             "this\nis\na\nmessage"\n             test-expr\n             None\n             None))\n    '
    (_, err) = run_cmd('hy -i', prg_str.format(3, 3, 1, 2))
    msg_idx = err.rindex('HyLanguageError:')
    assert msg_idx
    err_parts = err[msg_idx:].splitlines()[1:]
    expected = ['  File "<string>", line 3', "    'a 2 3", '    ^^', 'this', 'is', 'a', 'message']
    for (obs, exp) in zip(err_parts, expected):
        assert obs.startswith(exp)
    (_, err) = run_cmd('hy -i', prg_str.format(3, 3, 1, 1))
    msg_idx = err.rindex('HyLanguageError:')
    assert msg_idx
    err_parts = err[msg_idx:].splitlines()[1:]
    assert err_parts[2] == '    ^'
    (_, err) = run_cmd('hy -i', prg_str.format(3, 3, 1, 6))
    msg_idx = err.rindex('HyLanguageError:')
    assert msg_idx
    err_parts = err[msg_idx:].splitlines()[1:]
    assert err_parts[2] == '    ^----^'

def test_mangle_m():
    if False:
        print('Hello World!')
    (output, _) = run_cmd('hy -m tests.resources.hello_world')
    assert 'hello world' in output
    (output, _) = run_cmd('hy -m tests.resources.hello-world')
    assert 'hello world' in output

def test_ignore_python_env():
    if False:
        for i in range(10):
            print('nop')
    os.environ.update({'PYTHONTEST': '0'})
    (output, _) = run_cmd("hy -c '(print (do (import os) (. os environ)))'")
    assert 'PYTHONTEST' in output
    (output, _) = run_cmd('hy -m tests.resources.bin.printenv')
    assert 'PYTHONTEST' in output
    (output, _) = run_cmd('hy tests/resources/bin/printenv.hy')
    assert 'PYTHONTEST' in output
    (output, _) = run_cmd("hy -E -c '(print (do (import os) (. os environ)))'")
    assert 'PYTHONTEST' not in output
    os.environ.update({'PYTHONTEST': '0'})
    (output, _) = run_cmd('hy -E -m tests.resources.bin.printenv')
    assert 'PYTHONTEST' not in output
    os.environ.update({'PYTHONTEST': '0'})
    (output, _) = run_cmd('hy -E tests/resources/bin/printenv.hy')
    assert 'PYTHONTEST' not in output

def test_cmd():
    if False:
        for i in range(10):
            print('nop')
    (output, _) = run_cmd('hy -c \'(print (.upper "hello"))\'')
    assert 'HELLO' in output
    (_, err) = run_cmd('hy -c \'(print (.upper "hello")\'', expect=1)
    assert 'Premature end of input' in err
    (_, err) = run_cmd("hy -c '#!/usr/bin/env hy'", expect=1)
    assert 'LexException' in err
    (output, _) = run_cmd('hy -c \'(setv x "bing") (defn f [] (+ "fiz" x)) (print (f))\'')
    assert 'fizbing' in output
    (output, _) = run_cmd(' '.join(('hy -c ', repr('(import sys) (print (+ "<" (.join "|" sys.argv) ">"))'), 'AA', 'ZZ', '-m')))
    assert '<-c|AA|ZZ|-m>' in output

def test_icmd_string():
    if False:
        for i in range(10):
            print('nop')
    (output, _) = run_cmd('hy -i -c \'(.upper "hello")\'', '(.upper "bye")')
    assert 'HELLO' in output
    assert 'BYE' in output

def test_icmd_file():
    if False:
        while True:
            i = 10
    (output, _) = run_cmd('hy -i tests/resources/icmd_test_file.hy', '(.upper species)')
    assert 'CUTTLEFISH' in output

def test_icmd_shebang(tmp_path):
    if False:
        while True:
            i = 10
    (tmp_file := (tmp_path / 'icmd_with_shebang.hy')).write_text('#!/usr/bin/env hy\n(setv order "Sepiida")')
    (output, error) = run_cmd(['hy', '-i', tmp_file], '(.upper order)')
    assert '#!/usr/bin/env' not in error
    assert 'SEPIIDA' in output

def test_icmd_and_spy():
    if False:
        print('Hello World!')
    (output, _) = run_cmd('hy --spy -i -c "(+ [] [])"', '(+ 1 1)')
    assert '[] + []' in output

def test_empty_file(tmp_path):
    if False:
        i = 10
        return i + 15
    (tmp_path / 'foo.hy').write_text('')
    run_cmd(['hy', tmp_path / 'foo.hy'])

def test_missing_file():
    if False:
        while True:
            i = 10
    (_, err) = run_cmd('hy foobarbaz', expect=2)
    assert 'No such file' in err

def test_file_with_args():
    if False:
        return 10
    cmd = 'hy tests/resources/argparse_ex.hy'
    assert 'usage' in run_cmd(f'{cmd} -h')[0]
    assert 'got c' in run_cmd(f'{cmd} -c bar')[0]
    assert 'foo' in run_cmd(f'{cmd} -i foo')[0]
    assert 'foo' in run_cmd(f'{cmd} -i foo -c bar')[0]

def test_ifile_with_args():
    if False:
        while True:
            i = 10
    cmd = 'hy -i tests/resources/argparse_ex.hy'
    assert 'usage' in run_cmd(f'{cmd} -h')[0]
    assert 'got c' in run_cmd(f'{cmd} -c bar')[0]
    assert 'foo' in run_cmd(f'{cmd} -i foo')[0]
    assert 'foo' in run_cmd(f'{cmd} -i foo -c bar')[0]

def test_hyc():
    if False:
        i = 10
        return i + 15
    (output, _) = run_cmd('hyc -h')
    assert 'usage' in output
    path = 'tests/resources/argparse_ex.hy'
    (_, err) = run_cmd(['hyc', path])
    assert 'Compiling' in err
    assert os.path.exists(cache_from_source(path))
    rm(cache_from_source(path))

def test_hyc_missing_file():
    if False:
        while True:
            i = 10
    (_, err) = run_cmd('hyc foobarbaz', expect=1)
    assert '[Errno 2]' in err

def test_no_main():
    if False:
        while True:
            i = 10
    (output, _) = run_cmd('hy tests/resources/bin/nomain.hy')
    assert 'This Should Still Work' in output

@pytest.mark.parametrize('scenario', ['normal', 'prevent_by_force', 'prevent_by_env', 'prevent_by_option'])
@pytest.mark.parametrize('cmd_fmt', [['hy', '{fpath}'], ['hy', '-m', '{modname}'], ['hy', '-c', "'(import {modname})'"]])
def test_byte_compile(scenario, cmd_fmt):
    if False:
        print('Hello World!')
    modname = 'tests.resources.bin.bytecompile'
    fpath = modname.replace('.', '/') + '.hy'
    if scenario == 'prevent_by_option':
        cmd_fmt.insert(1, '-B')
    cmd = ' '.join(cmd_fmt).format(**locals())
    rm(cache_from_source(fpath))
    if scenario == 'prevent_by_force':
        os.mkdir(cache_from_source(fpath))
    (output, _) = run_cmd(cmd, dontwritebytecode=scenario == 'prevent_by_env')
    assert 'Hello from macro' in output
    assert 'The macro returned: boink' in output
    if scenario == 'normal':
        assert os.path.exists(cache_from_source(fpath))
    elif scenario == 'prevent_by_env' or scenario == 'prevent_by_option':
        assert not os.path.exists(cache_from_source(fpath))
    (output, _) = run_cmd(cmd)
    assert ('Hello from macro' in output) ^ (scenario == 'normal')
    assert 'The macro returned: boink' in output

def test_module_main_file():
    if False:
        i = 10
        return i + 15
    (output, _) = run_cmd('hy -m tests.resources.bin')
    assert 'This is a __main__.hy' in output
    (output, _) = run_cmd('hy -m .tests.resources.bin', expect=1)

def test_file_main_file():
    if False:
        for i in range(10):
            print('nop')
    (output, _) = run_cmd('hy tests/resources/bin')
    assert 'This is a __main__.hy' in output

def test_file_sys_path():
    if False:
        while True:
            i = 10
    "The test resource `relative_import.hy` will perform an absolute import\n    of a module in its directory: a directory that is not on the `sys.path` of\n    the script executing the module (i.e. `hy`).  We want to make sure that Hy\n    adopts the file's location in `sys.path`, instead of the runner's current\n    dir (e.g. '' in `sys.path`).\n    "
    (file_path, _) = os.path.split('tests/resources/relative_import.hy')
    file_relative_path = os.path.realpath(file_path)
    (output, _) = run_cmd('hy tests/resources/relative_import.hy')
    assert repr(file_relative_path) in output

def testc_file_sys_path():
    if False:
        for i in range(10):
            print('nop')
    test_file = 'tests/resources/relative_import_compile_time.hy'
    file_relative_path = os.path.realpath(os.path.dirname(test_file))
    for binary in ('hy', 'hyc', 'hy2py'):
        rm(cache_from_source(test_file))
        assert not os.path.exists(cache_from_source(file_relative_path))
        (output, _) = run_cmd([binary, test_file])
        assert repr(file_relative_path) in output

def test_module_no_main():
    if False:
        print('Hello World!')
    (output, _) = run_cmd('hy -m tests.resources.bin.nomain')
    assert 'This Should Still Work' in output

def test_sys_executable():
    if False:
        i = 10
        return i + 15
    (output, _) = run_cmd("hy -c '(do (import sys) (print sys.executable))'")
    assert os.path.basename(output.strip()) == 'hy'

def test_file_no_extension():
    if False:
        i = 10
        return i + 15
    'Confirm that a file with no extension is processed as Hy source'
    (output, _) = run_cmd('hy tests/resources/no_extension')
    assert 'This Should Still Work' in output

def test_circular_macro_require():
    if False:
        print('Hello World!')
    'Confirm that macros can require themselves during expansion and when\n    run from the command line.'
    test_file = 'tests/resources/bin/circular_macro_require.hy'
    rm(cache_from_source(test_file))
    assert not os.path.exists(cache_from_source(test_file))
    (output, _) = run_cmd(['hy', test_file])
    assert output.strip() == 'WOWIE'
    assert os.path.exists(cache_from_source(test_file))
    (output, _) = run_cmd(['hy', test_file])
    assert output.strip() == 'WOWIE'

def test_macro_require():
    if False:
        while True:
            i = 10
    'Confirm that a `require` will load macros into the non-module namespace\n    (i.e. `exec(code, locals)`) used by `runpy.run_path`.\n    In other words, this confirms that the AST generated for a `require` will\n    load macros into the unnamed namespace its run in.'
    test_file = 'tests/resources/bin/require_and_eval.hy'
    rm(cache_from_source(test_file))
    assert not os.path.exists(cache_from_source(test_file))
    (output, _) = run_cmd(['hy', test_file])
    assert output.strip() == 'abc'
    assert os.path.exists(cache_from_source(test_file))
    (output, _) = run_cmd(['hy', test_file])
    assert output.strip() == 'abc'

def test_tracebacks():
    if False:
        while True:
            i = 10
    'Make sure the printed tracebacks are correct.'

    def req_err(x):
        if False:
            while True:
                i = 10
        assert x == "hy.errors.HyRequireError: No module named 'not_a_real_module'"
    (_, error) = run_cmd('hy', '(require not-a-real-module)', expect=1)
    error_lines = error.splitlines()
    if error_lines[-1] == '':
        del error_lines[-1]
    assert len(error_lines) <= 10
    req_err(error_lines[-1])
    (_, error) = run_cmd('hy -c "(require not-a-real-module)"', expect=1)
    error_lines = error.splitlines()
    assert len(error_lines) <= 4
    req_err(error_lines[-1])
    (output, error) = run_cmd('hy -i -c "(require not-a-real-module)"', '')
    assert output.startswith('=> ')
    req_err(error.splitlines()[2])
    (_, error) = run_cmd('hy -c "(print \\""', expect=1)
    peoi_re = 'Traceback \\(most recent call last\\):\\n  File "(?:<string>|string-[0-9a-f]+)", line 1\\n    \\(print "\\n           \\^\\nhy.reader.exceptions.PrematureEndOfInput'
    assert re.search(peoi_re, error)
    (output, error) = run_cmd('hy -c "(print \\""', expect=1)
    assert output == ''
    assert re.match(peoi_re, error)
    (output, error) = run_cmd('hy -c "(print a)"', expect=1)
    error_lines = [x for x in error.splitlines() if set(x) != {' ', '^'}]
    assert error_lines[3] == '  File "<string>", line 1, in <module>'
    assert error_lines[-1].strip().replace(' global', '') == "NameError: name 'a' is not defined"
    (output, error) = run_cmd('hy -c "(compile)"', expect=1)
    error_lines = error.splitlines()
    assert error_lines[-2] == '  File "<string>", line 1, in <module>'
    assert error_lines[-1].startswith('TypeError')

def test_traceback_shebang(tmp_path):
    if False:
        return 10
    (tmp_path / 'ex.hy').write_text('#!my cool shebang\n(/ 1 0)')
    (_, error) = run_cmd(['hy', tmp_path / 'ex.hy'], expect=1)
    assert 'ZeroDivisionError'
    assert 'my cool shebang' not in error
    assert '(/ 1 0)' in error

def test_hystartup():
    if False:
        i = 10
        return i + 15
    os.environ['HYSTARTUP'] = 'tests/resources/hystartup.hy'
    (output, _) = run_cmd('hy -i', '[1 2]')
    assert '[1, 2]' in output
    assert '[1,_2]' in output
    (output, _) = run_cmd('hy -i', '(hello-world)')
    assert '(hello-world)' not in output
    assert '1 + 1' in output
    assert '2' in output
    (output, _) = run_cmd('hy -i', '#rad')
    assert '#rad' not in output
    assert "'totally' + 'rad'" in output
    assert "'totallyrad'" in output
    (output, _) = run_cmd('hy -i --repl-output-fn repr', '[1 2 3 4]')
    assert '[1, 2, 3, 4]' in output
    assert '[1 2 3 4]' not in output
    assert '[1,_2,_3,_4]' not in output
    os.environ['HYSTARTUP'] = 'tests/resources/spy_off_startup.hy'
    (output, _) = run_cmd('hy -i --spy', '[1 2]')
    assert '[1, 2]' in output
    assert '[1,~2]' in output
    del os.environ['HYSTARTUP']

def test_output_buffering(tmp_path):
    if False:
        print('Hello World!')
    tf = tmp_path / 'file.txt'
    pf = tmp_path / 'program.hy'
    pf.write_text(f'\n        (print "line 1")\n        (import  sys  pathlib [Path])\n        (print :file sys.stderr (.strip (.read-text (Path #[=[{tf}]=]))))\n        (print "line 2")')
    for (flags, expected) in (([], ''), (['--unbuffered'], 'line 1')):
        with open(tf, 'wb') as o:
            (_, stderr) = run_cmd(['hy', *flags, pf], stdout=o)
        assert stderr.strip() == expected
        assert tf.read_text().splitlines() == ['line 1', 'line 2']

def test_uufileuu(tmp_path, monkeypatch):
    if False:
        while True:
            i = 10
    (tmp_path / 'realdir').mkdir()
    (tmp_path / 'realdir' / 'hyex.hy').write_text('(print __file__)')
    (tmp_path / 'realdir' / 'pyex.py').write_text('print(__file__)')

    def file_is(arg, expected_py3_9):
        if False:
            for i in range(10):
                print('nop')
        expected = expected_py3_9 if PY3_9 else Path(arg)
        (output, _) = run_cmd(['python3', arg + 'pyex.py'])
        assert output.rstrip() == str(expected / 'pyex.py')
        (output, _) = run_cmd(['hy', arg + 'hyex.hy'])
        assert output.rstrip() == str(expected / 'hyex.hy')
    monkeypatch.chdir(tmp_path)
    file_is('realdir/', tmp_path / 'realdir')
    monkeypatch.chdir(tmp_path / 'realdir')
    file_is('', tmp_path / 'realdir')
    (tmp_path / 'symdir').symlink_to('realdir', target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    file_is('symdir/', tmp_path / 'symdir')
    (tmp_path / 'realdir' / 'child').mkdir()
    monkeypatch.chdir(tmp_path / 'realdir' / 'child')
    file_is('../', tmp_path / 'realdir' if platform.system() == 'Windows' else tmp_path / 'realdir' / 'child' / '..')

def test_assert(tmp_path, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.chdir(tmp_path)
    for has_msg in (False, True):
        Path('ex.hy').write_text('(defn f [test] (assert {} {}))'.format('(do (print "testing") test)', '(do (print "msging") "bye")' if has_msg else ''))
        for (optim, test) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            (out, err) = run_cmd(cmd='python3 {} {}'.format('-O' if optim else '', f"-c 'import hy, ex; ex.f({test})'"), expect=1 if not optim and (not test) else 0)
            assert ('testing' in out) == (not optim)
            show_msg = has_msg and (not optim) and (not test)
            assert ('msging' in out) == show_msg
            assert ('bye' in err) == show_msg

def test_hy2py_stdin():
    if False:
        for i in range(10):
            print('nop')
    (out, _) = run_cmd('hy2py', '(+ 482 223)')
    assert '482 + 223' in out
    assert '705' not in out

def test_hy2py_compile_only(monkeypatch):
    if False:
        for i in range(10):
            print('nop')

    def check(args):
        if False:
            i = 10
            return i + 15
        (output, _) = run_cmd(f'hy2py {args}')
        assert not re.search('^hello world$', output, re.M)
    monkeypatch.chdir('tests/resources')
    check('hello_world.hy')
    check('-m hello_world')
    monkeypatch.chdir('..')
    check('resources/hello_world.hy')
    check('-m resources.hello_world')

def test_hy2py_recursive(monkeypatch, tmp_path):
    if False:
        return 10
    (tmp_path / 'foo').mkdir()
    (tmp_path / 'foo/__init__.py').touch()
    (tmp_path / 'foo/first.hy').write_text('\n        (import foo.folder.second [a b])\n        (print a)\n        (print b)')
    (tmp_path / 'foo/folder').mkdir()
    (tmp_path / 'foo/folder/__init__.py').touch()
    (tmp_path / 'foo/folder/second.hy').write_text('\n        (setv a 1)\n        (setv b "hello world")')
    monkeypatch.chdir(tmp_path)
    (_, err) = run_cmd('hy2py -m foo', expect=1)
    assert 'ValueError' in err
    run_cmd('hy2py -m foo --output bar')
    assert set((tmp_path / 'bar').rglob('*')) == {tmp_path / 'bar' / p for p in ('first.py', 'folder', 'folder/second.py')}
    (output, _) = run_cmd('python3 first.py', cwd=tmp_path / 'bar')
    assert output == '1\nhello world\n'

@pytest.mark.parametrize('case', ['hy -m', 'hy2py -m'])
def test_relative_require(case, monkeypatch, tmp_path):
    if False:
        print('Hello World!')
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / '__init__.py').touch()
    (tmp_path / 'pkg' / 'a.hy').write_text('\n        (defmacro m []\n          \'(setv x (.upper "hello")))')
    (tmp_path / 'pkg' / 'b.hy').write_text('\n        (require .a [m])\n        (m)\n        (print x)')
    monkeypatch.chdir(tmp_path)
    if case == 'hy -m':
        (output, _) = run_cmd('hy -m pkg.b')
    elif case == 'hy2py -m':
        run_cmd('hy2py -m pkg -o out')
        (tmp_path / 'out' / '__init__.py').touch()
        (output, _) = run_cmd('python3 -m out.b')
    assert 'HELLO' in output

def test_require_doesnt_pollute_core(monkeypatch, tmp_path):
    if False:
        while True:
            i = 10
    'Macros loaded from an external module should not pollute\n    `_hy_macros` with macros from core.'
    (tmp_path / 'aaa.hy').write_text('\n        (defmacro foo []\n          \'(setv x (.upper "argelfraster")))')
    (tmp_path / 'bbb.hy').write_text('\n        (require aaa :as A)\n        (A.foo)\n        (print\n          x\n          (not-in "if" _hy_macros)\n          (not-in "cond" _hy_macros))')
    monkeypatch.chdir(tmp_path)
    for _ in (1, 2):
        assert 'ARGELFRASTER True True' in run_cmd('hy bbb.hy')[0]

def test_run_dir_or_zip(tmp_path):
    if False:
        print('Hello World!')
    (tmp_path / 'dir').mkdir()
    (tmp_path / 'dir' / '__main__.hy').write_text('(print (+ "A" "Z"))')
    (out, _) = run_cmd(['hy', tmp_path / 'dir'])
    assert 'AZ' in out
    from zipfile import ZipFile
    with ZipFile(tmp_path / 'zoom.zip', 'w') as o:
        o.writestr('__main__.hy', '(print (+ "B" "Y"))')
    (out, _) = run_cmd(['hy', tmp_path / 'zoom.zip'])
    assert 'BY' in out