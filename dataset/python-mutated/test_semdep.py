from pathlib import Path
from time import time
import pytest

@pytest.mark.kinda_slow
@pytest.mark.parametrize('file_size,target,max_time', [(file_size, target, max_time) for (file_size, max_time) in [('10k', 4), ('50k', 15), ('100k', 30)] for target in ['Gemfile.lock', 'go.mod', 'gradle.lockfile', 'maven_dep_tree.txt', 'package-lock.json', 'poetry.lock', 'requirements.txt', 'yarn.lock', 'Pipfile.lock']])
def test_dependency_aware_timing(parse_lockfile_path_in_tmp_for_perf, file_size, target, max_time):
    if False:
        for i in range(10):
            print('nop')
    start = time()
    (_, error) = parse_lockfile_path_in_tmp_for_perf(Path(f'targets_perf_sca/{file_size}/{target}'))
    end = time()
    assert len(error) == 0
    exec_time = end - start
    assert exec_time < max_time