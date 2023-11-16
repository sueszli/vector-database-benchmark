import pytest
from tests.fixtures import RunSemgrep
from semgrep.constants import OutputFormat

def idfn(options):
    if False:
        return 10
    return '-and-'.join((flag.strip('-') for flag in options if flag.startswith('--')))

@pytest.mark.kinda_slow
@pytest.mark.parametrize('options', [['--exclude', 'excluded.*'], ['--include', 'included.*'], ['--exclude', 'excluded'], ['--include', 'included'], ['--include', 'included', '--exclude', 'excluded.*'], ['--exclude', 'excluded', '--include', 'included.*'], ['--exclude', 'excluded.*', '--exclude', 'included.*'], ['--exclude', 'excluded', '--exclude', 'included'], ['--include', 'excluded.*', '--include', 'included.*'], ['--include', 'excluded', '--include', 'included'], ['--include', 'included.vue'], ['--include', 'included.vue', '--skip-unknown-extensions']], ids=idfn)
@pytest.mark.osemfail
def test_exclude_include(run_semgrep_in_tmp: RunSemgrep, snapshot, options):
    if False:
        print('Hello World!')
    (stdout, stderr) = run_semgrep_in_tmp('rules/eqeq.yaml', options=options, target_name='exclude_include', assert_exit_code=None)
    snapshot.assert_match(stdout, 'results.json')
    snapshot.assert_match(stderr, 'err.out')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_exclude_include_verbose_sorted_1(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        for i in range(10):
            print('nop')
    snapshot.assert_match(run_semgrep_in_tmp('rules/eqeq.yaml', options=['--exclude', 'excluded.*', '--exclude', 'included.*', '--verbose'], output_format=OutputFormat.TEXT, target_name='exclude_include', assert_exit_code=None).stderr, 'results.err')

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_exclude_include_verbose_sorted_2(run_semgrep_in_tmp: RunSemgrep, snapshot):
    if False:
        while True:
            i = 10
    snapshot.assert_match(run_semgrep_in_tmp('rules/nosem.yaml', options=['--exclude', '*.*', '--verbose'], output_format=OutputFormat.TEXT, target_name='basic', assert_exit_code=None).stderr, 'results.err')