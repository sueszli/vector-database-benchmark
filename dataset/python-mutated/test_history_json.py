"""Tests the json history backend."""
import shlex
import pytest
from xonsh.history.json import JsonHistory, _xhj_gc_bytes_to_rmfiles, _xhj_gc_commands_to_rmfiles, _xhj_gc_files_to_rmfiles, _xhj_gc_seconds_to_rmfiles
from xonsh.history.main import HistoryAlias, history_main
from xonsh.lazyjson import LazyJSON
CMDS = ['ls', 'cat hello kitty', 'abc', 'def', 'touch me', 'grep from me']
IGNORE_OPTS = ','.join(['ignoredups', 'ignoreerr', 'ignorespace'])

@pytest.fixture
def hist(tmpdir, xession, monkeypatch):
    if False:
        i = 10
        return i + 15
    file = tmpdir / 'xonsh-HISTORY-TEST.json'
    h = JsonHistory(filename=str(file), here='yup', sessionid='SESSIONID', gc=False)
    monkeypatch.setattr(xession, 'history', h)
    yield h

def test_hist_init(hist, xession):
    if False:
        for i in range(10):
            print('nop')
    'Test initialization of the shell history.'
    with LazyJSON(hist.filename) as lj:
        obs = lj['here']
    assert 'yup' == obs

def test_hist_append(hist, xession):
    if False:
        return 10
    'Verify appending to the history works.'
    xession.env['HISTCONTROL'] = set()
    hf = hist.append({'inp': 'still alive', 'rtn': 0})
    assert hf is None
    assert 'still alive' == hist.buffer[0]['inp']
    assert 0 == hist.buffer[0]['rtn']
    assert 0 == hist.rtns[-1]
    hf = hist.append({'inp': 'dead now', 'rtn': 1})
    assert 'dead now' == hist.buffer[1]['inp']
    assert 1 == hist.buffer[1]['rtn']
    assert 1 == hist.rtns[-1]
    hf = hist.append({'inp': 'reborn', 'rtn': 0})
    assert 'reborn' == hist.buffer[2]['inp']
    assert 0 == hist.buffer[2]['rtn']
    assert 0 == hist.rtns[-1]

def test_hist_flush(hist, xession):
    if False:
        i = 10
        return i + 15
    'Verify explicit flushing of the history works.'
    hf = hist.flush()
    assert hf is None
    xession.env['HISTCONTROL'] = set()
    hist.append({'inp': 'still alive?', 'rtn': 0, 'out': 'yes'})
    hf = hist.flush()
    assert hf is not None
    while hf.is_alive():
        pass
    with LazyJSON(hist.filename) as lj:
        assert len(lj['cmds']) == 1
        cmd = lj['cmds'][0]
        assert cmd['inp'] == 'still alive?'
        assert not cmd.get('out', None)

def test_hist_flush_with_store_stdout(hist, xession):
    if False:
        while True:
            i = 10
    'Verify explicit flushing of the history works.'
    hf = hist.flush()
    assert hf is None
    xession.env['HISTCONTROL'] = set()
    xession.env['XONSH_STORE_STDOUT'] = True
    hist.append({'inp': 'still alive?', 'rtn': 0, 'out': 'yes'})
    hf = hist.flush()
    assert hf is not None
    while hf.is_alive():
        pass
    with LazyJSON(hist.filename) as lj:
        assert len(lj['cmds']) == 1
        assert lj['cmds'][0]['inp'] == 'still alive?'
        assert lj['cmds'][0]['out'].strip() == 'yes'

def test_hist_flush_with_store_cwd(hist, xession):
    if False:
        print('Hello World!')
    hf = hist.flush()
    assert hf is None
    hist.save_cwd = True
    hist.append({'inp': '# saving with cwd', 'rtn': 0, 'out': 'yes', 'cwd': '/tmp'})
    hf = hist.flush()
    assert hf is not None
    hist.save_cwd = False
    hist.append({'inp': '# saving without cwd', 'rtn': 0, 'out': 'yes', 'cwd': '/tmp'})
    hf = hist.flush()
    assert hf is not None
    while hf.is_alive():
        pass
    with LazyJSON(hist.filename) as lj:
        assert len(lj['cmds']) == 2
        assert lj['cmds'][0]['cwd'] == '/tmp'
        assert 'cwd' not in lj['cmds'][1]

def test_hist_flush_with_hist_control(hist, xession):
    if False:
        return 10
    'Verify explicit flushing of the history works.'
    hf = hist.flush()
    assert hf is None
    xession.env['HISTCONTROL'] = IGNORE_OPTS
    hist.append({'inp': 'ls foo1', 'rtn': 0})
    hist.append({'inp': 'ls foo1', 'rtn': 1})
    hist.append({'inp': 'ls foo1', 'rtn': 0})
    hist.append({'inp': 'ls foo2', 'rtn': 2})
    hist.append({'inp': 'ls foo3', 'rtn': 0})
    hist.append({'inp': 'ls secret', 'rtn': 0, 'spc': True})
    hf = hist.flush()
    assert hf is not None
    while hf.is_alive():
        pass
    assert len(hist.buffer) == 0
    with LazyJSON(hist.filename) as lj:
        cmds = list(lj['cmds'])
        assert len(cmds) == 2
        assert [x['inp'] for x in cmds] == ['ls foo1', 'ls foo3']
        assert [x['rtn'] for x in cmds] == [0, 0]

def test_cmd_field(hist, xession):
    if False:
        for i in range(10):
            print('nop')
    xession.env['HISTCONTROL'] = set()
    hf = hist.append({'inp': 'ls foo', 'rtn': 1})
    assert hf is None
    assert 1 == hist.rtns[0]
    assert 1 == hist.rtns[-1]
    assert hist.outs[-1] is None
    assert [1] == hist.rtns[:]
    hf = hist.flush()
    assert hf is not None
    assert 1 == hist.rtns[0]
    assert 1 == hist.rtns[-1]
    assert hist.outs[-1] is None

@pytest.mark.parametrize('inp, commands, offset', [('', CMDS, (0, 1)), ('-r', list(reversed(CMDS)), (len(CMDS) - 1, -1)), ('0', CMDS[0:1], (0, 1)), ('1', CMDS[1:2], (1, 1)), ('-2', CMDS[-2:-1], (len(CMDS) - 2, 1)), ('1:3', CMDS[1:3], (1, 1)), ('1::2', CMDS[1::2], (1, 2)), ('-4:-2', CMDS[-4:-2], (len(CMDS) - 4, 1))])
def test_show_cmd_numerate(inp, commands, offset, hist, xession, capsys):
    if False:
        return 10
    'Verify that CLI history commands work.'
    (base_idx, step) = offset
    xession.env['HISTCONTROL'] = set()
    for (ts, cmd) in enumerate(CMDS):
        hist.append({'inp': cmd, 'rtn': 0, 'ts': (ts + 1, ts + 1.5)})
    exp = (f'{base_idx + idx * step}: {cmd}' for (idx, cmd) in enumerate(list(commands)))
    exp = '\n'.join(exp)
    history_main(['show', '-n'] + shlex.split(inp))
    (out, err) = capsys.readouterr()
    assert out.rstrip() == exp

def test_history_diff(tmpdir, xession, monkeypatch, capsys):
    if False:
        i = 10
        return i + 15
    files = [tmpdir / f'xonsh-HISTORY-TEST-{idx}.json' for idx in range(2)]
    for file in files:
        hist = JsonHistory(filename=str(file), here='yup', sessionid='SESSIONID', gc=False)
        monkeypatch.setattr(xession, 'history', hist)
        xession.env['HISTCONTROL'] = set()
        for (ts, cmd) in enumerate(CMDS):
            hist.append({'inp': cmd, 'rtn': 0, 'ts': (ts + 1, ts + 1.5)})
        flush = hist.flush()
        if flush.queue:
            time.sleep(0.1)
    (left, right) = (str(f) for f in files)
    history_main(['diff', left, right])
    (out, err) = capsys.readouterr()
    assert out.rstrip()

def test_histcontrol(hist, xession):
    if False:
        while True:
            i = 10
    'Test HISTCONTROL=ignoredups,ignoreerr,ignorespacee'
    xession.env['HISTCONTROL'] = IGNORE_OPTS
    assert len(hist.buffer) == 0
    hist.append({'inp': 'ls foo', 'rtn': 2})
    assert len(hist.buffer) == 1
    assert hist.rtns[-1] == 2
    assert hist.inps[-1] == 'ls foo'
    hist.append({'inp': 'ls foobazz', 'rtn': 0})
    assert len(hist.buffer) == 2
    assert 'ls foobazz' == hist.buffer[-1]['inp']
    assert 0 == hist.buffer[-1]['rtn']
    assert hist.rtns[-1] == 0
    assert hist.inps[-1] == 'ls foobazz'
    hist.append({'inp': 'ls foo', 'rtn': 2})
    assert len(hist.buffer) == 3
    assert 'ls foo' == hist.buffer[-1]['inp']
    assert 2 == hist.buffer[-1]['rtn']
    assert hist.rtns[-1] == 2
    assert hist.inps[-1] == 'ls foo'
    hist.append({'inp': 'ls foo', 'rtn': 0})
    assert len(hist.buffer) == 4
    assert 'ls foo' == hist.buffer[-1]['inp']
    assert 0 == hist.buffer[-1]['rtn']
    assert hist.rtns[-1] == 0
    assert hist.inps[-1] == 'ls foo'
    hist.append({'inp': 'ls', 'rtn': 0})
    assert len(hist.buffer) == 5
    assert 'ls' == hist.buffer[-1]['inp']
    assert 0 == hist.buffer[-1]['rtn']
    assert hist.rtns[-1] == 0
    assert hist.inps[-1] == 'ls'
    hist.append({'inp': 'ls', 'rtn': 0})
    assert len(hist.buffer) == 6
    assert hist.rtns[-1] == 0
    assert hist.inps[-1] == 'ls'
    hist.append({'inp': '/bin/ls', 'rtn': 0})
    assert len(hist.buffer) == 7
    assert '/bin/ls' == hist.buffer[-1]['inp']
    assert 0 == hist.buffer[-1]['rtn']
    assert hist.rtns[-1] == 0
    assert hist.inps[-1] == '/bin/ls'
    hist.append({'inp': 'ls bazz', 'rtn': 1})
    assert len(hist.buffer) == 8
    assert 'ls bazz' == hist.buffer[-1]['inp']
    assert 1 == hist.buffer[-1]['rtn']
    assert hist.rtns[-1] == 1
    assert hist.inps[-1] == 'ls bazz'
    hist.append({'inp': 'ls bazz', 'rtn': -1})
    assert len(hist.buffer) == 9
    assert 'ls bazz' == hist.buffer[-1]['inp']
    assert -1 == hist.buffer[-1]['rtn']
    assert hist.rtns[-1] == -1
    assert hist.inps[-1] == 'ls bazz'
    hist.append({'inp': 'echo not secret', 'rtn': 0, 'spc': False})
    assert len(hist.buffer) == 10
    assert 'echo not secret' == hist.buffer[-1]['inp']
    assert 0 == hist.buffer[-1]['rtn']
    assert hist.rtns[-1] == 0
    assert hist.inps[-1] == 'echo not secret'
    hist.append({'inp': 'echo secret command', 'rtn': 0, 'spc': True})
    assert len(hist.buffer) == 10
    assert hist.rtns[-1] == 0
    assert hist.inps[-1] == 'echo not secret'

@pytest.mark.parametrize('args', ['-h', '--help', 'show -h', 'show --help'])
def test_parse_args_help(args, capsys):
    if False:
        i = 10
        return i + 15
    with pytest.raises(SystemExit):
        history_main(shlex.split(args))
    assert 'show this help message and exit' in capsys.readouterr()[0]

@pytest.mark.parametrize('args, session, slice, numerate, reverse', [('', 'session', [], False, False), ('1:5', 'session', ['1:5'], False, False), ('show', 'session', [], False, False), ('show 15', 'session', ['15'], False, False), ('show bash 3:5 15:66', 'bash', ['3:5', '15:66'], False, False), ('show -r', 'session', [], False, True), ('show -rn bash', 'bash', [], True, True), ('show -n -r -30:20', 'session', ['-30:20'], True, True), ('show -n zsh 1:2:3', 'zsh', ['1:2:3'], True, False)])
def test_parser_show(args, session, slice, numerate, reverse, mocker, hist, xession):
    if False:
        return 10
    exp_ns = {'session': session, 'slices': slice, 'numerate': numerate, 'reverse': reverse, 'start_time': None, 'end_time': None, 'datetime_format': None, 'timestamp': False, 'null_byte': False}
    from xonsh.history import main as mod
    main = HistoryAlias()
    spy = mocker.spy(mod.xcli, '_dispatch_func')
    main(shlex.split(args))
    spy.assert_called_once()
    (args, _) = spy.call_args
    (_, kwargs) = args
    called_with = {attr: kwargs[attr] for attr in exp_ns}
    if kwargs['_unparsed']:
        called_with['slices'] = kwargs['_unparsed']
    assert called_with == exp_ns

@pytest.mark.parametrize('index, exp', [(-1, ('grep from me', 'out', 0, (5, 6))), (1, ('cat hello kitty', 'out', 0, (1, 2))), (slice(1, 3), [('cat hello kitty', 'out', 0, (1, 2)), ('abc', 'out', 0, (2, 3))])])
def test_history_getitem(index, exp, hist, xession):
    if False:
        print('Hello World!')
    xession.env['HISTCONTROL'] = set()
    attrs = ('inp', 'out', 'rtn', 'ts')
    for (ts, cmd) in enumerate(CMDS):
        entry = {k: v for (k, v) in zip(attrs, [cmd, 'out', 0, (ts, ts + 1)])}
        hist.append(entry)
    entry = hist[index]
    if isinstance(entry, list):
        assert [(e.cmd, e.out, e.rtn, e.ts) for e in entry] == exp
    else:
        assert (entry.cmd, entry.out, entry.rtn, entry.ts) == exp
import calendar
import time
HF_FIRST_DAY = calendar.timegm(time.struct_time((2018, 5, 13, 0, 0, 0, 0, 0, 0)))

def history_files_list(gen_count) -> (float, int, str, int):
    if False:
        i = 10
        return i + 15
    'Generate a list of history file tuples'
    retval = []
    for i in range(int((gen_count + 1) / 2)):
        retval.append((HF_FIRST_DAY + ((i * 24 + 9) * 60 + 0) * 60 + 0, 100, f'.argle/xonsh-{2 * i:05n}.json', 10000))
        retval.append((HF_FIRST_DAY + ((i * 24 + 23) * 60 + 0) * 60 + 0, 50, f'.argle/xonsh-{2 * i + 1:05n}.json', 2500))
    return retval
HISTORY_FILES_LIST = history_files_list(100)
SEC_FROM_LATEST = time.time() - HISTORY_FILES_LIST[-1][0]
SEC_PER_DAY = 24 * 60 * 60
MAX_RUNTIME = 30 * 60
MIN_DIFF = min((HISTORY_FILES_LIST[i + 1][0] - HISTORY_FILES_LIST[i][0] for i in range(len(HISTORY_FILES_LIST) - 1)))
assert MAX_RUNTIME < MIN_DIFF / 2

@pytest.mark.parametrize('fn, hsize, in_files, exp_size, exp_files', [(_xhj_gc_commands_to_rmfiles, 1001 * (100 + 50), HISTORY_FILES_LIST, 0, []), (_xhj_gc_commands_to_rmfiles, 20 * (100 + 50), HISTORY_FILES_LIST, 30 * (100 + 50), HISTORY_FILES_LIST[:2 * 30]), (_xhj_gc_commands_to_rmfiles, 20 * (100 + 50) + 100, HISTORY_FILES_LIST, 30 * (100 + 50) - 50, HISTORY_FILES_LIST[:2 * 30 - 1]), (_xhj_gc_files_to_rmfiles, 1001, HISTORY_FILES_LIST, 0, []), (_xhj_gc_files_to_rmfiles, 40, HISTORY_FILES_LIST, 60, HISTORY_FILES_LIST[:60]), (_xhj_gc_files_to_rmfiles, 41, HISTORY_FILES_LIST, 59, HISTORY_FILES_LIST[:2 * 30 - 1]), (_xhj_gc_bytes_to_rmfiles, 1001 * (10000 + 2500), HISTORY_FILES_LIST, 0, []), (_xhj_gc_bytes_to_rmfiles, 20 * (10000 + 2500), HISTORY_FILES_LIST, 30 * (10000 + 2500), HISTORY_FILES_LIST[:2 * 30]), (_xhj_gc_bytes_to_rmfiles, 20 * (10000 + 2500) + 10000, HISTORY_FILES_LIST, 30 * (10000 + 2500) - 2500, HISTORY_FILES_LIST[:2 * 30 - 1]), (_xhj_gc_seconds_to_rmfiles, SEC_FROM_LATEST + 1001 * SEC_PER_DAY, HISTORY_FILES_LIST, 0, []), (_xhj_gc_seconds_to_rmfiles, SEC_FROM_LATEST + MAX_RUNTIME, HISTORY_FILES_LIST, HISTORY_FILES_LIST[-1][0] - HISTORY_FILES_LIST[0][0], HISTORY_FILES_LIST[:-1]), (_xhj_gc_seconds_to_rmfiles, SEC_FROM_LATEST + SEC_PER_DAY + MAX_RUNTIME, HISTORY_FILES_LIST, HISTORY_FILES_LIST[-3][0] - HISTORY_FILES_LIST[0][0], HISTORY_FILES_LIST[:-3])])
def test__xhj_gc_xx_to_rmfiles(fn, hsize, in_files, exp_size, exp_files, xession):
    if False:
        for i in range(10):
            print('nop')
    (act_size, act_files) = fn(hsize, in_files)
    assert act_files == exp_files
    if fn == _xhj_gc_seconds_to_rmfiles:
        assert abs(act_size - exp_size) < MAX_RUNTIME
    else:
        assert act_size == exp_size

def test_hist_clear_cmd(hist, xession, capsys, tmpdir):
    if False:
        print('Hello World!')
    'Verify that the CLI history clear command works.'
    xession.env.update({'XONSH_DATA_DIR': str(tmpdir)})
    xession.env['HISTCONTROL'] = set()
    for (ts, cmd) in enumerate(CMDS):
        hist.append({'inp': cmd, 'rtn': 0, 'ts': (ts + 1, ts + 1.5)})
    assert len(xession.history) == 6
    history_main(['clear'])
    (out, err) = capsys.readouterr()
    assert err.rstrip() == 'History cleared'
    assert len(xession.history) == 0

def test_hist_off_cmd(hist, xession, capsys, tmpdir):
    if False:
        print('Hello World!')
    'Verify that the CLI history off command works.'
    xession.env.update({'XONSH_DATA_DIR': str(tmpdir)})
    xession.env['HISTCONTROL'] = set()
    for (ts, cmd) in enumerate(CMDS):
        hist.append({'inp': cmd, 'rtn': 0, 'ts': (ts + 1, ts + 1.5)})
    assert len(xession.history) == 6
    history_main(['off'])
    (out, err) = capsys.readouterr()
    assert err.rstrip() == 'History off'
    assert len(xession.history) == 0
    for (ts, cmd) in enumerate(CMDS):
        hist.append({'inp': cmd, 'rtn': 0, 'ts': (ts + 1, ts + 1.5)})
    assert len(xession.history) == 0

def test_hist_on_cmd(hist, xession, capsys, tmpdir):
    if False:
        i = 10
        return i + 15
    'Verify that the CLI history on command works.'
    xession.env.update({'XONSH_DATA_DIR': str(tmpdir)})
    xession.env['HISTCONTROL'] = set()
    for (ts, cmd) in enumerate(CMDS):
        hist.append({'inp': cmd, 'rtn': 0, 'ts': (ts + 1, ts + 1.5)})
    assert len(xession.history) == 6
    history_main(['off'])
    history_main(['on'])
    (out, err) = capsys.readouterr()
    assert err.rstrip().endswith('History on')
    assert len(xession.history) == 0
    for (ts, cmd) in enumerate(CMDS):
        hist.append({'inp': cmd, 'rtn': 0, 'ts': (ts + 1, ts + 1.5)})
    assert len(xession.history) == 6