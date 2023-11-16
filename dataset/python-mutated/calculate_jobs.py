import json
import os

def set_output(key: str, value: str):
    if False:
        while True:
            i = 10
    with open(os.environ['GITHUB_OUTPUT'], 'at') as f:
        print(f'{key}={value}', file=f)
IS_PR = os.environ['GITHUB_REF'].startswith('refs/pull/')
trial_sqlite_tests = [{'python-version': '3.8', 'database': 'sqlite', 'extras': 'all'}]
if not IS_PR:
    trial_sqlite_tests.extend(({'python-version': version, 'database': 'sqlite', 'extras': 'all'} for version in ('3.9', '3.10', '3.11', '3.12')))
trial_postgres_tests = [{'python-version': '3.8', 'database': 'postgres', 'postgres-version': '11', 'extras': 'all'}]
if not IS_PR:
    trial_postgres_tests.append({'python-version': '3.12', 'database': 'postgres', 'postgres-version': '16', 'extras': 'all'})
trial_no_extra_tests = [{'python-version': '3.8', 'database': 'sqlite', 'extras': ''}]
print('::group::Calculated trial jobs')
print(json.dumps(trial_sqlite_tests + trial_postgres_tests + trial_no_extra_tests, indent=4))
print('::endgroup::')
test_matrix = json.dumps(trial_sqlite_tests + trial_postgres_tests + trial_no_extra_tests)
set_output('trial_test_matrix', test_matrix)
sytest_tests = [{'sytest-tag': 'focal'}, {'sytest-tag': 'focal', 'postgres': 'postgres'}, {'sytest-tag': 'focal', 'postgres': 'multi-postgres', 'workers': 'workers'}, {'sytest-tag': 'focal', 'postgres': 'multi-postgres', 'workers': 'workers', 'reactor': 'asyncio'}]
if not IS_PR:
    sytest_tests.extend([{'sytest-tag': 'focal', 'reactor': 'asyncio'}, {'sytest-tag': 'focal', 'postgres': 'postgres', 'reactor': 'asyncio'}, {'sytest-tag': 'testing', 'postgres': 'postgres'}])
print('::group::Calculated sytest jobs')
print(json.dumps(sytest_tests, indent=4))
print('::endgroup::')
test_matrix = json.dumps(sytest_tests)
set_output('sytest_test_matrix', test_matrix)