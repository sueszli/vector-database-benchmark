import math
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests
from tools.testing.test_run import ShardedTest, TestRun
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IS_MEM_LEAK_CHECK = os.getenv('PYTORCH_TEST_CUDA_MEM_LEAK_CHECK', '0') == '1'
IS_ROCM = os.path.exists('/opt/rocm')
NUM_PROCS = 1 if IS_MEM_LEAK_CHECK else 2
NUM_PROCS_FOR_SHARDING_CALC = NUM_PROCS if not IS_ROCM or IS_MEM_LEAK_CHECK else 2
THRESHOLD = 60 * 10
if IS_ROCM and (not IS_MEM_LEAK_CHECK):
    try:
        lines = subprocess.check_output(['rocminfo'], encoding='ascii').strip().split('\n')
        count = 0
        for line in lines:
            if ' gfx' in line:
                count += 1
        assert count > 0
        NUM_PROCS = 8 if count > 8 else count
    except subprocess.CalledProcessError as e:
        NUM_PROCS = 1

class ShardJob:

    def __init__(self) -> None:
        if False:
            return 10
        self.serial: List[ShardedTest] = []
        self.parallel: List[ShardedTest] = []

    def get_total_time(self) -> float:
        if False:
            return 10
        procs = [0.0 for _ in range(NUM_PROCS_FOR_SHARDING_CALC)]
        for test in self.parallel:
            min_index = procs.index(min(procs))
            procs[min_index] += test.get_time()
        time = max(procs) + sum((test.get_time() for test in self.serial))
        return time

    def convert_to_tuple(self) -> Tuple[float, List[ShardedTest]]:
        if False:
            print('Hello World!')
        return (self.get_total_time(), self.serial + self.parallel)

def get_with_pytest_shard(tests: Sequence[TestRun], test_file_times: Dict[str, float], test_class_times: Optional[Dict[str, Dict[str, float]]]) -> List[ShardedTest]:
    if False:
        for i in range(10):
            print('nop')
    sharded_tests: List[ShardedTest] = []

    def get_duration_for_classes(test_file: str, test_classes: Set[str]) -> Optional[float]:
        if False:
            return 10
        duration: float = 0
        if not test_class_times:
            return None
        for test_class in test_classes:
            class_duration = test_class_times.get(test_file, {}).get(test_class, None)
            if class_duration is None:
                return None
            if class_duration:
                duration += class_duration
        return duration
    for test in tests:
        file_duration = test_file_times.get(test.test_file, None)
        included = test.included()
        excluded = test.excluded()
        included_classes_duration = get_duration_for_classes(test.test_file, included)
        excluded_classes_duration = get_duration_for_classes(test.test_file, excluded)
        if included:
            duration = included_classes_duration if included_classes_duration is not None else file_duration
        elif excluded:
            duration = file_duration - excluded_classes_duration if excluded_classes_duration is not None and file_duration is not None else file_duration
        else:
            duration = file_duration
        if duration and duration > THRESHOLD:
            num_shards = math.ceil(duration / THRESHOLD)
            for i in range(num_shards):
                sharded_tests.append(ShardedTest(test, i + 1, num_shards, duration / num_shards))
        else:
            sharded_tests.append(ShardedTest(test, 1, 1, duration))
    return sharded_tests

def calculate_shards(num_shards: int, tests: Sequence[TestRun], test_file_times: Dict[str, float], test_class_times: Optional[Dict[str, Dict[str, float]]], must_serial: Optional[Callable[[str], bool]]=None, sort_by_time: bool=True) -> List[Tuple[float, List[ShardedTest]]]:
    if False:
        for i in range(10):
            print('nop')
    must_serial = must_serial or (lambda x: True)
    known_tests: Sequence[TestRun] = tests
    unknown_tests: Sequence[TestRun] = []
    if sort_by_time:
        known_tests = [x for x in tests if x.test_file in test_file_times or (test_class_times and x.test_file in test_class_times)]
        unknown_tests = [x for x in tests if x not in known_tests]
    known_tests = get_with_pytest_shard(known_tests, test_file_times, test_class_times)
    if sort_by_time:
        known_tests = sorted(known_tests, key=lambda j: j.get_time(), reverse=True)
    sharded_jobs: List[ShardJob] = [ShardJob() for _ in range(num_shards)]
    for test in known_tests:
        if must_serial(test.name):
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.serial.append(test)
        else:
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.parallel.append(test)
    index = min(range(num_shards), key=lambda i: sharded_jobs[i].get_total_time())
    for unknown_test in unknown_tests:
        sharded_jobs[index].serial.append(ShardedTest(unknown_test, 1, 1, None))
        index = (index + 1) % num_shards
    return [job.convert_to_tuple() for job in sharded_jobs]

def get_test_case_configs(dirpath: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)