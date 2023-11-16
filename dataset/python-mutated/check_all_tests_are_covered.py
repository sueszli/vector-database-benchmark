"""
In .github/workflows/pytest.yml, tests are split between multiple jobs.

Here, we check that the jobs ignored by the first job actually end up getting
run by the other jobs.
This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run check-no-tests-are-ignored --all`.
"""
import itertools
import logging
import os
from pathlib import Path
import pandas
import yaml
_log = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

def find_testfiles():
    if False:
        print('Hello World!')
    dp_repo = Path(__file__).parent.parent
    all_tests = {str(fp.relative_to(dp_repo)).replace(os.sep, '/') for fp in (dp_repo / 'tests').glob('**/test_*.py')}
    _log.info('Found %i tests in total.', len(all_tests))
    return all_tests

def from_yaml():
    if False:
        while True:
            i = 10
    'Determines how often each test file is run per platform and floatX setting.\n\n    An exception is raised if tests run multiple times with the same configuration.\n    '
    matrices = {}
    for wf in ['tests.yml']:
        wfname = wf.rstrip('.yml')
        wfdef = yaml.safe_load(open(Path('.github', 'workflows', wf)))
        for (jobname, jobdef) in wfdef['jobs'].items():
            matrix = jobdef.get('strategy', {}).get('matrix', {})
            if matrix:
                matrices[wfname, jobname] = matrix
            else:
                _log.warning('No matrix in %s/%s', wf, jobname)
    all_os = []
    all_floatX = []
    for matrix in matrices.values():
        all_os += matrix['os']
        all_floatX += matrix['floatx']
    all_os = tuple(sorted(set(all_os)))
    all_floatX = tuple(sorted(set(all_floatX)))
    all_tests = find_testfiles()
    df = pandas.DataFrame(columns=pandas.MultiIndex.from_product([sorted(all_floatX), sorted(all_os)], names=['floatX', 'os']), index=pandas.Index(sorted(all_tests), name='testfile'))
    df.loc[:, :] = 0
    for matrix in matrices.values():
        for (os_, floatX, subset) in itertools.product(matrix['os'], matrix['floatx'], matrix['test-subset']):
            lines = [l for l in subset.split('\n') if l]
            if 'windows' in os_:
                if lines and lines[-1].endswith(' \\'):
                    raise Exception(f"Last entry '{line}' in Windows test subset should end WITHOUT ' \\'.")
                for line in lines[:-1]:
                    if not line.endswith(' \\'):
                        raise Exception(f"Missing ' \\' after '{line}' in Windows test-subset.")
                lines = [line.rstrip(' \\') for line in lines]
            testfiles = []
            for line in lines:
                testfiles += line.split(' ')
            ignored = {item[8:].lstrip(' =') for item in testfiles if item.startswith('--ignore')}
            included = {item for item in testfiles if item and (not item.startswith('--ignore'))}
            if ignored and (not included):
                included = all_tests - ignored
            for testfile in included:
                df.loc[testfile, (floatX, os_)] += 1
    ignored_by_all = set(df[df.eq(0).all(axis=1)].index)
    run_multiple_times = set(df[df.gt(1).any(axis=1)].index)
    _log.info('Number of test runs (❌=0, ✅=once)\n%s', df.replace(0, '❌').replace(1, '✅'))
    if ignored_by_all:
        raise AssertionError(f'{len(ignored_by_all)} tests are completely ignored:\n{ignored_by_all}')
    if run_multiple_times:
        raise AssertionError(f'{len(run_multiple_times)} tests are run multiple times with the same OS and floatX setting:\n{run_multiple_times}')
    return
if __name__ == '__main__':
    from_yaml()