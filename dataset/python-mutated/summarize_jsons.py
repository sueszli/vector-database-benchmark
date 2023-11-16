import json
import os
import time
from typing import Any, Dict, List, Set, Tuple
from ..util.setting import CompilerType, JSON_FOLDER_BASE_DIR, TestList, TestPlatform, TestStatusType
from ..util.utils import detect_compiler_type, print_error, print_time, related_to_test_list
from .parser.coverage_record import CoverageRecord
from .parser.gcov_coverage_parser import GcovCoverageParser
from .parser.llvm_coverage_parser import LlvmCoverageParser
from .print_report import file_oriented_report, html_oriented_report, line_oriented_report
covered_lines: Dict[str, Set[int]] = {}
uncovered_lines: Dict[str, Set[int]] = {}
tests_type: TestStatusType = {'success': set(), 'partial': set(), 'fail': set()}

def transform_file_name(file_path: str, interested_folders: List[str], platform: TestPlatform) -> str:
    if False:
        for i in range(10):
            print('nop')
    remove_patterns: Set[str] = {'.DEFAULT.cpp', '.AVX.cpp', '.AVX2.cpp'}
    for pattern in remove_patterns:
        file_path = file_path.replace(pattern, '')
    if interested_folders:
        for folder in interested_folders:
            if folder in file_path:
                return file_path[file_path.find(folder):]
    if platform == TestPlatform.OSS:
        from package.oss.utils import get_pytorch_folder
        pytorch_foler = get_pytorch_folder()
        assert file_path.startswith(pytorch_foler)
        file_path = file_path[len(pytorch_foler) + 1:]
    return file_path

def is_intrested_file(file_path: str, interested_folders: List[str], platform: TestPlatform) -> bool:
    if False:
        return 10
    ignored_patterns = ['cuda', 'aten/gen_aten', 'aten/aten_', 'build/']
    if any((pattern in file_path for pattern in ignored_patterns)):
        return False
    if platform == TestPlatform.OSS:
        from package.oss.utils import get_pytorch_folder
        if not file_path.startswith(get_pytorch_folder()):
            return False
    if interested_folders:
        for folder in interested_folders:
            intersted_folder_path = folder if folder.endswith('/') else f'{folder}/'
            if intersted_folder_path in file_path:
                return True
        return False
    else:
        return True

def get_json_obj(json_file: str) -> Tuple[Any, int]:
    if False:
        i = 10
        return i + 15
    '\n    Sometimes at the start of file llvm/gcov will complains "fail to find coverage data",\n    then we need to skip these lines\n      -- success read: 0      -  this json file have the full json coverage information\n      -- partial success: 1   -  this json file starts with some error prompt, but still have the coverage information\n      -- fail to read: 2      -  this json file doesn\'t have any coverage information\n    '
    read_status = -1
    with open(json_file) as f:
        lines = f.readlines()
        for line in lines:
            try:
                json_obj = json.loads(line)
            except json.JSONDecodeError:
                read_status = 1
                continue
            else:
                if read_status == -1:
                    read_status = 0
                return (json_obj, read_status)
    return (None, 2)

def parse_json(json_file: str, platform: TestPlatform) -> List[CoverageRecord]:
    if False:
        i = 10
        return i + 15
    print('start parse:', json_file)
    (json_obj, read_status) = get_json_obj(json_file)
    if read_status == 0:
        tests_type['success'].add(json_file)
    elif read_status == 1:
        tests_type['partial'].add(json_file)
    else:
        tests_type['fail'].add(json_file)
        raise RuntimeError('Fail to do code coverage! Fail to load json file: ', json_file)
    cov_type = detect_compiler_type(platform)
    coverage_records: List[CoverageRecord] = []
    if cov_type == CompilerType.CLANG:
        coverage_records = LlvmCoverageParser(json_obj).parse('fbcode')
    elif cov_type == CompilerType.GCC:
        coverage_records = GcovCoverageParser(json_obj).parse()
    return coverage_records

def parse_jsons(test_list: TestList, interested_folders: List[str], platform: TestPlatform) -> None:
    if False:
        return 10
    g = os.walk(JSON_FOLDER_BASE_DIR)
    for (path, _, file_list) in g:
        for file_name in file_list:
            if file_name.endswith('.json'):
                cov_type = detect_compiler_type(platform)
                if cov_type == CompilerType.CLANG and (not related_to_test_list(file_name, test_list)):
                    continue
                json_file = os.path.join(path, file_name)
                try:
                    coverage_records = parse_json(json_file, platform)
                except RuntimeError:
                    print_error('Fail to load json file: ', json_file)
                    continue
                update_coverage(coverage_records, interested_folders, platform)

def update_coverage(coverage_records: List[CoverageRecord], interested_folders: List[str], platform: TestPlatform) -> None:
    if False:
        while True:
            i = 10
    for item in coverage_records:
        record = item.to_dict()
        file_path = record['filepath']
        if not is_intrested_file(file_path, interested_folders, platform):
            continue
        covered_range = record['covered_lines']
        uncovered_range = record['uncovered_lines']
        file_path = transform_file_name(file_path, interested_folders, platform)
        if file_path not in covered_lines:
            covered_lines[file_path] = set()
        if file_path not in uncovered_lines:
            uncovered_lines[file_path] = set()
        if covered_range is not None:
            covered_lines[file_path].update(covered_range)
        if uncovered_range is not None:
            uncovered_lines[file_path].update(uncovered_range)

def update_set() -> None:
    if False:
        return 10
    for file_name in covered_lines:
        uncovered_lines[file_name].difference_update(covered_lines[file_name])

def summarize_jsons(test_list: TestList, interested_folders: List[str], coverage_only: List[str], platform: TestPlatform) -> None:
    if False:
        for i in range(10):
            print('nop')
    start_time = time.time()
    if detect_compiler_type(platform) == CompilerType.GCC:
        html_oriented_report()
    else:
        parse_jsons(test_list, interested_folders, platform)
        update_set()
        line_oriented_report(test_list, tests_type, interested_folders, coverage_only, covered_lines, uncovered_lines)
        file_oriented_report(test_list, tests_type, interested_folders, coverage_only, covered_lines, uncovered_lines)
    print_time('summary jsons take time: ', start_time)