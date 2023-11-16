from __future__ import annotations
import multiprocessing
import sys
import traceback
import pytest
from conda.base.context import conda_tests_ctxt_mgmt_def_pol, context
from conda.common.compat import on_win
from conda.common.io import env_vars
from conda.gateways.repodata import RepodataCache, lock

def locker(cache: RepodataCache, qout, qin):
    if False:
        print('Hello World!')
    print(f'Attempt to lock {cache.cache_path_state}')
    qout.put('ready')
    print('sent ready to parent')
    assert qin.get(timeout=6) == 'locked'
    print('parent locked. try to save in child (should fail)')
    try:
        cache.save('{}')
        qout.put('not locked')
    except OSError as e:
        print('OSError', e)
        qout.put(e)
    except Exception as e:
        print('Not OSError', e, file=sys.stderr)
        traceback.print_exception(e)
        qout.put(e)
    else:
        print('no exception')
        qout.put(None)
    print('exit child')

@pytest.mark.parametrize('use_lock', [True, False])
def test_lock_can_lock(tmp_path, use_lock: bool):
    if False:
        while True:
            i = 10
    '\n    Open lockfile, then open it again in a spawned subprocess. Assert subprocess\n    times out (should take 10 seconds).\n    '
    multiprocessing.set_start_method('spawn', force=True)
    vars = {'CONDA_PLATFORM': 'osx-64'}
    if not use_lock:
        vars['CONDA_NO_LOCK'] = '1'
    with env_vars(vars, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        assert context.no_lock == (not use_lock)
        cache = RepodataCache(tmp_path / 'lockme', 'repodata.json')
        qout = multiprocessing.Queue()
        qin = multiprocessing.Queue()
        p = multiprocessing.Process(target=locker, args=(cache, qin, qout))
        p.start()
        assert qin.get(timeout=6) == 'ready'
        print('subprocess ready')
        with cache.cache_path_state.open('a+') as lock_file, lock(lock_file):
            print('lock acquired in parent process')
            qout.put('locked')
            if use_lock:
                assert isinstance(qin.get(timeout=13), OSError)
            else:
                assert qin.get(timeout=5) == 'not locked'
            p.join(1)
            assert p.exitcode == 0

@pytest.mark.skipif(on_win, reason='emulate windows behavior for code coverage')
def test_lock_rename(tmp_path):
    if False:
        print('Hello World!')

    class PunyPath(type(tmp_path)):

        def rename(self, path):
            if False:
                print('Hello World!')
            if path.exists():
                raise FileExistsError()
            return super().rename(path)
    with env_vars({'CONDA_EXPERIMENTAL': 'lock'}, stack_callback=conda_tests_ctxt_mgmt_def_pol):
        cache = RepodataCache(tmp_path / 'lockme', 'puny.json')
        cache.save('{}')
        puny = PunyPath(tmp_path, 'puny.json.tmp')
        puny.write_text('{"info":{}}')
        cache.replace(puny)