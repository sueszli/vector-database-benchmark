"""Test CLI usage."""
import logging
import subprocess
import sys
from functools import wraps
from os import linesep
from tqdm.cli import TqdmKeyError, TqdmTypeError, main
from tqdm.utils import IS_WIN
from .tests_tqdm import BytesIO, closing, mark, raises

def restore_sys(func):
    if False:
        for i in range(10):
            print('nop')
    'Decorates `func(capsysbinary)` to save & restore `sys.(stdin|argv)`.'

    @wraps(func)
    def inner(capsysbinary):
        if False:
            while True:
                i = 10
        'function requiring capsysbinary which may alter `sys.(stdin|argv)`'
        _SYS = (sys.stdin, sys.argv)
        try:
            res = func(capsysbinary)
        finally:
            (sys.stdin, sys.argv) = _SYS
        return res
    return inner

def norm(bytestr):
    if False:
        for i in range(10):
            print('nop')
    'Normalise line endings.'
    return bytestr if linesep == '\n' else bytestr.replace(linesep.encode(), b'\n')

@mark.slow
def test_pipes():
    if False:
        i = 10
        return i + 15
    'Test command line pipes'
    ls_out = subprocess.check_output(['ls'])
    ls = subprocess.Popen(['ls'], stdout=subprocess.PIPE)
    res = subprocess.Popen([sys.executable, '-c', 'from tqdm.cli import main; main()'], stdin=ls.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = res.communicate()
    assert ls.poll() == 0
    assert norm(ls_out) == norm(out)
    assert b'it/s' in err
    assert b'Error' not in err
if sys.version_info[:2] >= (3, 8):
    test_pipes = mark.filterwarnings('ignore:unclosed file:ResourceWarning')(test_pipes)

def test_main_import():
    if False:
        print('Hello World!')
    'Test main CLI import'
    N = 123
    _SYS = (sys.stdin, sys.argv)
    sys.stdin = [str(i).encode() for i in range(N)]
    sys.argv = ['', '--desc', 'Test CLI import', '--ascii', 'True', '--unit_scale', 'True']
    try:
        import tqdm.__main__
    finally:
        (sys.stdin, sys.argv) = _SYS

@restore_sys
def test_main_bytes(capsysbinary):
    if False:
        i = 10
        return i + 15
    'Test CLI --bytes'
    N = 123
    IN_DATA = '\x00'.join(map(str, range(N))).encode()
    with closing(BytesIO()) as sys.stdin:
        sys.stdin.write(IN_DATA)
        sys.stdin.seek(0)
        main(sys.stderr, ['--desc', 'Test CLI delim', '--ascii', 'True', '--delim', '\\0', '--buf_size', '64'])
        (out, err) = capsysbinary.readouterr()
        assert out == IN_DATA
        assert str(N) + 'it' in err.decode('U8')
    IN_DATA = IN_DATA.replace(b'\x00', b'\n')
    with closing(BytesIO()) as sys.stdin:
        sys.stdin.write(IN_DATA)
        sys.stdin.seek(0)
        main(sys.stderr, ['--ascii', '--bytes=True', '--unit_scale', 'False'])
        (out, err) = capsysbinary.readouterr()
        assert out == IN_DATA
        assert str(len(IN_DATA)) + 'B' in err.decode('U8')

def test_main_log(capsysbinary, caplog):
    if False:
        i = 10
        return i + 15
    'Test CLI --log'
    _SYS = (sys.stdin, sys.argv)
    N = 123
    sys.stdin = [(str(i) + '\n').encode() for i in range(N)]
    IN_DATA = b''.join(sys.stdin)
    try:
        with caplog.at_level(logging.INFO):
            main(sys.stderr, ['--log', 'INFO'])
            (out, err) = capsysbinary.readouterr()
            assert norm(out) == IN_DATA and b'123/123' in err
            assert not caplog.record_tuples
        with caplog.at_level(logging.DEBUG):
            main(sys.stderr, ['--log', 'DEBUG'])
            (out, err) = capsysbinary.readouterr()
            assert norm(out) == IN_DATA and b'123/123' in err
            assert caplog.record_tuples
    finally:
        (sys.stdin, sys.argv) = _SYS

@restore_sys
def test_main(capsysbinary):
    if False:
        for i in range(10):
            print('nop')
    'Test misc CLI options'
    N = 123
    sys.stdin = [(str(i) + '\n').encode() for i in range(N)]
    IN_DATA = b''.join(sys.stdin)
    main(sys.stderr, ['--mininterval', '0', '--miniters', '1'])
    (out, err) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA and b'123/123' in err
    assert N <= len(err.split(b'\r')) < N + 5
    len_err = len(err)
    main(sys.stderr, ['--tee', '--mininterval', '0', '--miniters', '1'])
    (out, err) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA and b'123/123' in err
    assert len_err + len(norm(out)) <= len(err)
    main(sys.stderr, ['--null'])
    (out, err) = capsysbinary.readouterr()
    assert not out and b'123/123' in err
    main(sys.stderr, ['--update'])
    (out, err) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA
    assert (str(N // 2 * N) + 'it').encode() in err, 'expected arithmetic sum formula'
    main(sys.stderr, ['--update-to'])
    (out, err) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA
    assert (str(N - 1) + 'it').encode() in err
    assert (str(N) + 'it').encode() not in err
    with closing(BytesIO()) as sys.stdin:
        sys.stdin.write(IN_DATA.replace(b'\n', b'D'))
        sys.stdin.seek(0)
        main(sys.stderr, ['--update', '--delim', 'D'])
        (out, err) = capsysbinary.readouterr()
        assert out == IN_DATA.replace(b'\n', b'D')
        assert (str(N // 2 * N) + 'it').encode() in err, 'expected arithmetic sum'
        sys.stdin.seek(0)
        main(sys.stderr, ['--update-to', '--delim', 'D'])
        (out, err) = capsysbinary.readouterr()
        assert out == IN_DATA.replace(b'\n', b'D')
        assert (str(N - 1) + 'it').encode() in err
        assert (str(N) + 'it').encode() not in err
    sys.stdin = [(str(i / 2.0) + '\n').encode() for i in range(N)]
    IN_DATA = b''.join(sys.stdin)
    main(sys.stderr, ['--update-to'])
    (out, err) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA
    assert (str((N - 1) / 2.0) + 'it').encode() in err
    assert (str(N / 2.0) + 'it').encode() not in err

@mark.slow
@mark.skipif(IS_WIN, reason='no manpages on windows')
def test_manpath(tmp_path):
    if False:
        print('Hello World!')
    'Test CLI --manpath'
    man = tmp_path / 'tqdm.1'
    assert not man.exists()
    with raises(SystemExit):
        main(argv=['--manpath', str(tmp_path)])
    assert man.is_file()

@mark.slow
@mark.skipif(IS_WIN, reason='no completion on windows')
def test_comppath(tmp_path):
    if False:
        i = 10
        return i + 15
    'Test CLI --comppath'
    man = tmp_path / 'tqdm_completion.sh'
    assert not man.exists()
    with raises(SystemExit):
        main(argv=['--comppath', str(tmp_path)])
    assert man.is_file()
    script = man.read_text()
    opts = {'--help', '--desc', '--total', '--leave', '--ncols', '--ascii', '--dynamic_ncols', '--position', '--bytes', '--nrows', '--delim', '--manpath', '--comppath'}
    assert all((args in script for args in opts))

@restore_sys
def test_exceptions(capsysbinary):
    if False:
        i = 10
        return i + 15
    'Test CLI Exceptions'
    N = 123
    sys.stdin = [str(i) + '\n' for i in range(N)]
    IN_DATA = ''.join(sys.stdin).encode()
    with raises(TqdmKeyError, match='bad_arg_u_ment'):
        main(sys.stderr, argv=['-ascii', '-unit_scale', '--bad_arg_u_ment', 'foo'])
    (out, _) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA
    with raises(TqdmTypeError, match='invalid_bool_value'):
        main(sys.stderr, argv=['-ascii', '-unit_scale', 'invalid_bool_value'])
    (out, _) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA
    with raises(TqdmTypeError, match='invalid_int_value'):
        main(sys.stderr, argv=['-ascii', '--total', 'invalid_int_value'])
    (out, _) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA
    with raises(TqdmKeyError, match='Can only have one of --'):
        main(sys.stderr, argv=['--update', '--update_to'])
    (out, _) = capsysbinary.readouterr()
    assert norm(out) == IN_DATA
    for i in ('-h', '--help', '-v', '--version'):
        with raises(SystemExit):
            main(argv=[i])