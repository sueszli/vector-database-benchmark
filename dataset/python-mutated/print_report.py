import os
import subprocess
from typing import Dict, IO, List, Set, Tuple
from ..oss.utils import get_pytorch_folder
from ..util.setting import SUMMARY_FOLDER_DIR, TestList, TestStatusType
CoverageItem = Tuple[str, float, int, int]

def key_by_percentage(x: CoverageItem) -> float:
    if False:
        for i in range(10):
            print('nop')
    return x[1]

def key_by_name(x: CoverageItem) -> str:
    if False:
        i = 10
        return i + 15
    return x[0]

def is_intrested_file(file_path: str, interested_folders: List[str]) -> bool:
    if False:
        print('Hello World!')
    if 'cuda' in file_path:
        return False
    if 'aten/gen_aten' in file_path or 'aten/aten_' in file_path:
        return False
    for folder in interested_folders:
        if folder in file_path:
            return True
    return False

def is_this_type_of_tests(target_name: str, test_set_by_type: Set[str]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    for test in test_set_by_type:
        if target_name in test:
            return True
    return False

def print_test_by_type(tests: TestList, test_set_by_type: Set[str], type_name: str, summary_file: IO[str]) -> None:
    if False:
        while True:
            i = 10
    print('Tests ' + type_name + ' to collect coverage:', file=summary_file)
    for test in tests:
        if is_this_type_of_tests(test.name, test_set_by_type):
            print(test.target_pattern, file=summary_file)
    print(file=summary_file)

def print_test_condition(tests: TestList, tests_type: TestStatusType, interested_folders: List[str], coverage_only: List[str], summary_file: IO[str], summary_type: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    print_test_by_type(tests, tests_type['success'], 'fully success', summary_file)
    print_test_by_type(tests, tests_type['partial'], 'partially success', summary_file)
    print_test_by_type(tests, tests_type['fail'], 'failed', summary_file)
    print('\n\nCoverage Collected Over Interested Folders:\n', interested_folders, file=summary_file)
    print('\n\nCoverage Compilation Flags Only Apply To: \n', coverage_only, file=summary_file)
    print('\n\n---------------------------------- ' + summary_type + ' ----------------------------------', file=summary_file)

def line_oriented_report(tests: TestList, tests_type: TestStatusType, interested_folders: List[str], coverage_only: List[str], covered_lines: Dict[str, Set[int]], uncovered_lines: Dict[str, Set[int]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    with open(os.path.join(SUMMARY_FOLDER_DIR, 'line_summary'), 'w+') as report_file:
        print_test_condition(tests, tests_type, interested_folders, coverage_only, report_file, 'LINE SUMMARY')
        for file_name in covered_lines:
            covered = covered_lines[file_name]
            uncovered = uncovered_lines[file_name]
            print(f'{file_name}\n  covered lines: {sorted(covered)}\n  unconvered lines:{sorted(uncovered)}', file=report_file)

def print_file_summary(covered_summary: int, total_summary: int, summary_file: IO[str]) -> float:
    if False:
        print('Hello World!')
    try:
        coverage_percentage = 100.0 * covered_summary / total_summary
    except ZeroDivisionError:
        coverage_percentage = 0
    print(f'SUMMARY\ncovered: {covered_summary}\nuncovered: {total_summary}\npercentage: {coverage_percentage:.2f}%\n\n', file=summary_file)
    if coverage_percentage == 0:
        print('Coverage is 0, Please check if json profiles are valid')
    return coverage_percentage

def print_file_oriented_report(tests_type: TestStatusType, coverage: List[CoverageItem], covered_summary: int, total_summary: int, summary_file: IO[str], tests: TestList, interested_folders: List[str], coverage_only: List[str]) -> None:
    if False:
        print('Hello World!')
    coverage_percentage = print_file_summary(covered_summary, total_summary, summary_file)
    print_test_condition(tests, tests_type, interested_folders, coverage_only, summary_file, 'FILE SUMMARY')
    for item in coverage:
        print(item[0].ljust(75), (str(item[1]) + '%').rjust(10), str(item[2]).rjust(10), str(item[3]).rjust(10), file=summary_file)
    print(f'summary percentage:{coverage_percentage:.2f}%')

def file_oriented_report(tests: TestList, tests_type: TestStatusType, interested_folders: List[str], coverage_only: List[str], covered_lines: Dict[str, Set[int]], uncovered_lines: Dict[str, Set[int]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    with open(os.path.join(SUMMARY_FOLDER_DIR, 'file_summary'), 'w+') as summary_file:
        covered_summary = 0
        total_summary = 0
        coverage = []
        for file_name in covered_lines:
            covered_count = len(covered_lines[file_name])
            total_count = covered_count + len(uncovered_lines[file_name])
            try:
                percentage = round(covered_count / total_count * 100, 2)
            except ZeroDivisionError:
                percentage = 0
            coverage.append((file_name, percentage, covered_count, total_count))
            covered_summary = covered_summary + covered_count
            total_summary = total_summary + total_count
        coverage.sort(key=key_by_name)
        coverage.sort(key=key_by_percentage)
        print_file_oriented_report(tests_type, coverage, covered_summary, total_summary, summary_file, tests, interested_folders, coverage_only)

def get_html_ignored_pattern() -> List[str]:
    if False:
        i = 10
        return i + 15
    return ['/usr/*', '*anaconda3/*', '*third_party/*']

def html_oriented_report() -> None:
    if False:
        i = 10
        return i + 15
    build_folder = os.path.join(get_pytorch_folder(), 'build')
    coverage_info_file = os.path.join(SUMMARY_FOLDER_DIR, 'coverage.info')
    subprocess.check_call(['lcov', '--capture', '--directory', build_folder, '--output-file', coverage_info_file])
    cmd_array = ['lcov', '--remove', coverage_info_file] + get_html_ignored_pattern() + ['--output-file', coverage_info_file]
    subprocess.check_call(cmd_array)
    subprocess.check_call(['genhtml', coverage_info_file, '--output-directory', os.path.join(SUMMARY_FOLDER_DIR, 'html_report')])