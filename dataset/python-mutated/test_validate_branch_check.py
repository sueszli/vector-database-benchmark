import json
import os
import subprocess
import sys
from hypothesistooling.projects.hypothesispython import BASE_DIR
BRANCH_CHECK = 'branch-check'
VALIDATE_BRANCH_CHECK = os.path.join(BASE_DIR, 'scripts', 'validate_branch_check.py')

def write_entries(tmp_path, entries):
    if False:
        return 10
    with open(tmp_path / BRANCH_CHECK, 'w', encoding='utf-8') as f:
        f.writelines([json.dumps(entry) + '\n' for entry in entries])

def run_validate_branch_check(tmp_path, *, check, **kwargs):
    if False:
        i = 10
        return i + 15
    return subprocess.run([sys.executable, VALIDATE_BRANCH_CHECK], cwd=tmp_path, text=True, capture_output=True, check=check, **kwargs)

def test_validates_branches(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    write_entries(tmp_path, [{'name': name, 'value': value} for name in ('first', 'second', 'third') for value in (False, True)])
    output = run_validate_branch_check(tmp_path, check=True)
    assert output.stdout == 'Successfully validated 3 branches.\n'

def test_validates_one_branch(tmp_path):
    if False:
        return 10
    write_entries(tmp_path, [{'name': 'sole', 'value': value} for value in (False, True)])
    output = run_validate_branch_check(tmp_path, check=True)
    assert output.stdout == 'Successfully validated 1 branch.\n'

def test_fails_on_zero_branches(tmp_path):
    if False:
        i = 10
        return i + 15
    write_entries(tmp_path, [])
    output = run_validate_branch_check(tmp_path, check=False)
    assert output.returncode == 1
    assert output.stdout == 'No branches found in the branch-check file?\n'

def test_reports_uncovered_branches(tmp_path):
    if False:
        print('Hello World!')
    write_entries(tmp_path, [{'name': 'branch that is always taken', 'value': True}, {'name': 'some other branch that is never taken', 'value': False}, {'name': 'covered branch', 'value': True}, {'name': 'covered branch', 'value': False}])
    output = run_validate_branch_check(tmp_path, check=False)
    assert output.returncode == 1
    expected = 'Some branches were not properly covered.\n\nThe following were always True:\n  * branch that is always taken\n\nThe following were always False:\n  * some other branch that is never taken\n'
    assert output.stdout == expected