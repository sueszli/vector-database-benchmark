import pytest

@pytest.fixture
def saltcheck(modules):
    if False:
        print('Hello World!')
    return modules.saltcheck

@pytest.mark.slow_test
def test_saltcheck_render_pyobjects_state(state_tree, saltcheck):
    if False:
        for i in range(10):
            print('nop')
    with pytest.helpers.temp_file('pyobj_touched.txt') as tpath:
        sls_content = f'\n        #!pyobjects\n\n        File.touch("{tpath}")\n        '
        tst_content = f'\n        is_stuff_there:\n          module_and_function: file.file_exists\n          args:\n            - "{tpath}"\n          assertion: assertTrue\n        '
        with pytest.helpers.temp_file('pyobj_touched/init.sls', sls_content, state_tree), pytest.helpers.temp_file('pyobj_touched/saltcheck-tests/init.tst', tst_content, state_tree):
            ret = saltcheck.run_state_tests('pyobj_touched')
            assert ret[0]['pyobj_touched']['is_stuff_there']['status'] == 'Pass'
            assert ret[1]['TEST RESULTS']['Passed'] == 1
            assert ret[1]['TEST RESULTS']['Missing Tests'] == 0
            assert ret[1]['TEST RESULTS']['Failed'] == 0
            assert ret[1]['TEST RESULTS']['Skipped'] == 0

@pytest.mark.slow_test
def test_saltcheck_allow_remote_fileclient(state_tree, saltcheck):
    if False:
        while True:
            i = 10
    sls_content = '\n    test_state:\n      test.show_notification:\n        - text: The test state\n    '
    tst_content = '\n    test cp.cache_file:\n      module_and_function: cp.cache_file\n      args:\n        - salt://sltchk_remote/download_me.txt\n      kwargs:\n        saltenv: base\n      assertion: assertNotEmpty\n      output_details: True\n    '
    with pytest.helpers.temp_file('sltchk_remote/init.sls', sls_content, state_tree), pytest.helpers.temp_file('sltchk_remote/saltcheck-tests/init.tst', tst_content, state_tree), pytest.helpers.temp_file('sltchk_remote/download_me.txt', 'salty', state_tree):
        ret = saltcheck.run_state_tests('sltchk_remote')
        assert ret[0]['sltchk_remote']['test cp.cache_file']['status'] == 'Pass'
        assert ret[1]['TEST RESULTS']['Passed'] == 1
        assert ret[1]['TEST RESULTS']['Missing Tests'] == 0
        assert ret[1]['TEST RESULTS']['Failed'] == 0
        assert ret[1]['TEST RESULTS']['Skipped'] == 0