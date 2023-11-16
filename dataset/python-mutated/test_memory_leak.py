import time
from multiprocessing import Manager, Process
import psutil
import pytest
pytestmark = [pytest.mark.slow_test]

@pytest.fixture
def testfile_path(tmp_path):
    if False:
        return 10
    return tmp_path / 'testfile'

@pytest.fixture
def file_add_delete_sls(testfile_path, base_env_state_tree_root_dir):
    if False:
        return 10
    sls_name = 'file_add'
    sls_contents = '\n    add_file:\n      file.managed:\n        - name: {path}\n        - source: salt://testfile\n        - makedirs: true\n        - require:\n          - cmd: echo\n\n    delete_file:\n      file.absent:\n        - name: {path}\n        - require:\n          - file: add_file\n\n    echo:\n      cmd.run:\n        - name: "echo \'This is a test!\'"\n    '.format(path=testfile_path)
    with pytest.helpers.temp_file('{}.sls'.format(sls_name), sls_contents, base_env_state_tree_root_dir):
        yield sls_name

@pytest.mark.skip_on_darwin(reason="MacOS is a spawning platform, won't work")
@pytest.mark.flaky(max_runs=4)
def test_memory_leak(salt_cli, salt_minion, file_add_delete_sls):
    if False:
        return 10
    max_usg = None
    with Manager() as manager:
        done_flag = manager.list()
        during_run_data = manager.list()

        def _func(data, flag):
            if False:
                i = 10
                return i + 15
            while len(flag) == 0:
                time.sleep(0.05)
                usg = psutil.virtual_memory()
                data.append(usg.total - usg.available)
        proc = Process(target=_func, args=(during_run_data, done_flag))
        proc.start()
        for _ in range(50):
            salt_cli.run('state.sls', file_add_delete_sls, minion_tgt=salt_minion.id)
        done_flag.append(1)
        proc.join()
        start_usg = during_run_data[0]
        max_usg = during_run_data[0]
        for row in during_run_data[1:]:
            max_usg = row if row >= max_usg else max_usg
    if max_usg > start_usg:
        max_tries = 50
        threshold = (max_usg - start_usg) * 0.25 + start_usg
        for _ in range(max_tries):
            usg = psutil.virtual_memory()
            current_usg = usg.total - usg.available
            if current_usg <= start_usg:
                break
            if current_usg <= threshold:
                break
            time.sleep(2)
        else:
            pytest.fail('Memory usage did not drop off appropriately')