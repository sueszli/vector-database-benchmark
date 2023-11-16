import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, Tuple
from tools.stats.upload_stats_lib import download_gha_artifacts, download_s3_artifacts, is_rerun_disabled_tests, unzip, upload_workflow_stats_to_s3
from tools.stats.upload_test_stats import process_xml_element
TESTCASE_TAG = 'testcase'
SEPARATOR = ';'

def process_report(report: Path) -> Dict[str, Dict[str, int]]:
    if False:
        print('Hello World!')
    '\n    Return a list of disabled tests that should be re-enabled and those that are still\n    flaky (failed or skipped)\n    '
    root = ET.parse(report)
    all_tests: Dict[str, Dict[str, int]] = {}
    for test_case in root.iter(TESTCASE_TAG):
        parsed_test_case = process_xml_element(test_case)
        skipped = parsed_test_case.get('skipped', None)
        if skipped and (type(skipped) is list or 'num_red' not in skipped.get('message', '')):
            continue
        name = parsed_test_case.get('name', '')
        classname = parsed_test_case.get('classname', '')
        filename = parsed_test_case.get('file', '')
        if not name or not classname or (not filename):
            continue
        failure = parsed_test_case.get('failure', None)
        disabled_test_id = SEPARATOR.join([name, classname, filename])
        if disabled_test_id not in all_tests:
            all_tests[disabled_test_id] = {'num_green': 0, 'num_red': 0}
        if skipped:
            try:
                stats = json.loads(skipped.get('message', ''))
            except json.JSONDecodeError:
                stats = {}
            all_tests[disabled_test_id]['num_green'] += stats.get('num_green', 0)
            all_tests[disabled_test_id]['num_red'] += stats.get('num_red', 0)
        elif failure:
            all_tests[disabled_test_id]['num_red'] += 1
        else:
            all_tests[disabled_test_id]['num_green'] += 1
    return all_tests

def get_test_reports(repo: str, workflow_run_id: int, workflow_run_attempt: int) -> Generator[Path, None, None]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Gather all the test reports from S3 and GHA. It is currently not possible to guess which\n    test reports are from rerun_disabled_tests workflow because the name doesn't include the\n    test config. So, all reports will need to be downloaded and examined\n    "
    with TemporaryDirectory() as temp_dir:
        print('Using temporary directory:', temp_dir)
        os.chdir(temp_dir)
        artifact_paths = download_s3_artifacts('test-reports', workflow_run_id, workflow_run_attempt)
        for path in artifact_paths:
            unzip(path)
        artifact_paths = download_gha_artifacts('test-report', workflow_run_id, workflow_run_attempt)
        for path in artifact_paths:
            unzip(path)
        yield from Path('.').glob('**/*.xml')

def get_disabled_test_name(test_id: str) -> Tuple[str, str, str, str]:
    if False:
        return 10
    '\n    Follow flaky bot convention here, if that changes, this will also need to be updated\n    '
    (name, classname, filename) = test_id.split(SEPARATOR)
    return (f'{name} (__main__.{classname})', name, classname, filename)

def prepare_record(workflow_id: int, workflow_run_attempt: int, name: str, classname: str, filename: str, flaky: bool, num_red: int=0, num_green: int=0) -> Tuple[Any, Dict[str, Any]]:
    if False:
        print('Hello World!')
    '\n    Prepare the record to save onto S3\n    '
    key = (workflow_id, workflow_run_attempt, name, classname, filename)
    record = {'workflow_id': workflow_id, 'workflow_run_attempt': workflow_run_attempt, 'name': name, 'classname': classname, 'filename': filename, 'flaky': flaky, 'num_green': num_green, 'num_red': num_red}
    return (key, record)

def save_results(workflow_id: int, workflow_run_attempt: int, all_tests: Dict[str, Dict[str, int]]) -> None:
    if False:
        return 10
    '\n    Save the result to S3, so it can go to Rockset\n    '
    should_be_enabled_tests = {name: stats for (name, stats) in all_tests.items() if 'num_green' in stats and stats['num_green'] and ('num_red' in stats) and (stats['num_red'] == 0)}
    still_flaky_tests = {name: stats for (name, stats) in all_tests.items() if name not in should_be_enabled_tests}
    records = {}
    for (test_id, stats) in all_tests.items():
        num_green = stats.get('num_green', 0)
        num_red = stats.get('num_red', 0)
        (disabled_test_name, name, classname, filename) = get_disabled_test_name(test_id)
        (key, record) = prepare_record(workflow_id=workflow_id, workflow_run_attempt=workflow_run_attempt, name=name, classname=classname, filename=filename, flaky=test_id in still_flaky_tests, num_green=num_green, num_red=num_red)
        records[key] = record
    print(f'The following {len(should_be_enabled_tests)} tests should be re-enabled:')
    for (test_id, stats) in should_be_enabled_tests.items():
        (disabled_test_name, name, classname, filename) = get_disabled_test_name(test_id)
        print(f'  {disabled_test_name} from {filename}')
    print(f'The following {len(still_flaky_tests)} are still flaky:')
    for (test_id, stats) in still_flaky_tests.items():
        num_green = stats.get('num_green', 0)
        num_red = stats.get('num_red', 0)
        (disabled_test_name, name, classname, filename) = get_disabled_test_name(test_id)
        print(f'  {disabled_test_name} from {filename}, failing {num_red}/{num_red + num_green}')
    upload_workflow_stats_to_s3(workflow_id, workflow_run_attempt, 'rerun_disabled_tests', list(records.values()))

def main(repo: str, workflow_run_id: int, workflow_run_attempt: int) -> None:
    if False:
        return 10
    '\n    Find the list of all disabled tests that should be re-enabled\n    '
    all_tests: Dict[str, Dict[str, int]] = {}
    for report in get_test_reports(args.repo, args.workflow_run_id, args.workflow_run_attempt):
        tests = process_report(report)
        if not is_rerun_disabled_tests(tests):
            continue
        for (name, stats) in tests.items():
            if name not in all_tests:
                all_tests[name] = stats.copy()
            else:
                all_tests[name]['num_green'] += stats.get('num_green', 0)
                all_tests[name]['num_red'] += stats.get('num_red', 0)
    save_results(workflow_run_id, workflow_run_attempt, all_tests)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload test artifacts from GHA to S3')
    parser.add_argument('--workflow-run-id', type=int, required=True, help='id of the workflow to get artifacts from')
    parser.add_argument('--workflow-run-attempt', type=int, required=True, help='which retry of the workflow this is')
    parser.add_argument('--repo', type=str, required=True, help='which GitHub repo this workflow run belongs to')
    args = parser.parse_args()
    main(args.repo, args.workflow_run_id, args.workflow_run_attempt)