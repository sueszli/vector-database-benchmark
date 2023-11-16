import re
import pytest
from tests.fixtures import RunSemgrep
from semgrep.constants import OutputFormat

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test1_no_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        return 10
    (results, _) = run_semgrep_in_tmp('rules/fixtest/test1.yaml', target_name='fixtest/test1.py', options=['--test'], output_format=OutputFormat.TEXT)
    snapshot.assert_match(results, 'output.txt')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test1_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        while True:
            i = 10
    (stdout, _) = run_semgrep_in_tmp('rules/fixtest/test1.yaml', target_name='fixtest/test1.py', options=['--test'], output_format=OutputFormat.JSON)
    snapshot.assert_match(stdout, 'test-results.json')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test2_no_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        while True:
            i = 10
    (results, _) = run_semgrep_in_tmp('rules/fixtest/test2.yaml', target_name='fixtest/test2.py', options=['--test'], output_format=OutputFormat.TEXT)
    snapshot.assert_match(results, 'output.txt')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test2_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        for i in range(10):
            print('nop')
    (stdout, _) = run_semgrep_in_tmp('rules/fixtest/test2.yaml', target_name='fixtest/test2.py', options=['--test'], output_format=OutputFormat.JSON)
    snapshot.assert_match(stdout, 'test-results.json')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test3_no_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        return 10
    (results, _) = run_semgrep_in_tmp('rules/fixtest/test3.yaml', target_name='fixtest/test3.py', options=['--test'], output_format=OutputFormat.TEXT, assert_exit_code=1)
    snapshot.assert_match(results, 'output.txt')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test3_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        print('Hello World!')
    (stdout, _) = run_semgrep_in_tmp('rules/fixtest/test3.yaml', target_name='fixtest/test3.py', options=['--test'], output_format=OutputFormat.JSON, assert_exit_code=1)
    snapshot.assert_match(stdout, 'test-results.json')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test4_no_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        while True:
            i = 10
    results = run_semgrep_in_tmp('rules/fixtest/test4.yaml', target_name='fixtest/test4.py', options=['--test'], output_format=OutputFormat.TEXT, assert_exit_code=1, use_click_runner=True)
    snapshot.assert_match(results.as_snapshot(mask=[re.compile('test file path: (.+?)/fixtest/test4.py')]), 'results.txt')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test4_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        i = 10
        return i + 15
    results = run_semgrep_in_tmp('rules/fixtest/test4.yaml', target_name='fixtest/test4.py', options=['--test'], output_format=OutputFormat.JSON, assert_exit_code=1, use_click_runner=True)
    snapshot.assert_match(results.as_snapshot(mask=[re.compile('test file path: (.+?)/fixtest/test4.py')]), 'results.txt')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test5_no_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        while True:
            i = 10
    (results, _) = run_semgrep_in_tmp('rules/fixtest/test5.yaml', target_name='fixtest/test5.py', options=['--test'], output_format=OutputFormat.TEXT)
    snapshot.assert_match(results, 'output.txt')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_fixtest_test5_json(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        print('Hello World!')
    (stdout, _) = run_semgrep_in_tmp('rules/fixtest/test5.yaml', target_name='fixtest/test5.py', options=['--test'], output_format=OutputFormat.JSON)
    snapshot.assert_match(stdout, 'test-results.json')