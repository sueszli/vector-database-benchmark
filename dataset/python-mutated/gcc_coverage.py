import os
import subprocess
import time
from typing import Dict
from ..oss.utils import get_gcda_files, run_oss_python_test
from ..util.setting import JSON_FOLDER_BASE_DIR, TestType
from ..util.utils import print_log, print_time
from .utils import run_cpp_test

def update_gzip_dict(gzip_dict: Dict[str, int], file_name: str) -> str:
    if False:
        i = 10
        return i + 15
    file_name = file_name.lower()
    gzip_dict[file_name] = gzip_dict.get(file_name, 0) + 1
    num = gzip_dict[file_name]
    return str(num) + '_' + file_name

def run_target(binary_file: str, test_type: TestType) -> None:
    if False:
        print('Hello World!')
    print_log('start run', test_type.value, 'test: ', binary_file)
    start_time = time.time()
    assert test_type in {TestType.CPP, TestType.PY}
    if test_type == TestType.CPP:
        run_cpp_test(binary_file)
    else:
        run_oss_python_test(binary_file)
    print_time(' time: ', start_time)

def export() -> None:
    if False:
        while True:
            i = 10
    start_time = time.time()
    gcda_files = get_gcda_files()
    gzip_dict: Dict[str, int] = {}
    for gcda_item in gcda_files:
        subprocess.check_call(['gcov', '-i', gcda_item])
        gz_file_name = os.path.basename(gcda_item) + '.gcov.json.gz'
        new_file_path = os.path.join(JSON_FOLDER_BASE_DIR, update_gzip_dict(gzip_dict, gz_file_name))
        os.rename(gz_file_name, new_file_path)
        subprocess.check_output(['gzip', '-d', new_file_path])
    print_time('export take time: ', start_time, summary_time=True)