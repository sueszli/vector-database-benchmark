import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from tools.stats.upload_stats_lib import download_gha_artifacts, download_s3_artifacts, unzip, upload_workflow_stats_to_s3

def get_job_id(report: Path) -> Optional[int]:
    if False:
        while True:
            i = 10
    try:
        return int(report.parts[0].rpartition('_')[2])
    except ValueError:
        return None

def parse_xml_report(tag: str, report: Path, workflow_id: int, workflow_run_attempt: int) -> List[Dict[str, Any]]:
    if False:
        print('Hello World!')
    'Convert a test report xml file into a JSON-serializable list of test cases.'
    print(f'Parsing {tag}s for test report: {report}')
    job_id = get_job_id(report)
    print(f'Found job id: {job_id}')
    test_cases: List[Dict[str, Any]] = []
    root = ET.parse(report)
    for test_case in root.iter(tag):
        case = process_xml_element(test_case)
        case['workflow_id'] = workflow_id
        case['workflow_run_attempt'] = workflow_run_attempt
        case['job_id'] = job_id
        case['invoking_file'] = report.parent.name
        test_cases.append(case)
    return test_cases

def process_xml_element(element: ET.Element) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Convert a test suite element into a JSON-serializable dict.'
    ret: Dict[str, Any] = {}
    ret.update(element.attrib)
    for (k, v) in ret.items():
        try:
            ret[k] = int(v)
        except ValueError:
            pass
        try:
            ret[k] = float(v)
        except ValueError:
            pass
    if element.text and element.text.strip():
        ret['text'] = element.text
    if element.tail and element.tail.strip():
        ret['tail'] = element.tail
    for child in element:
        if child.tag not in ret:
            ret[child.tag] = process_xml_element(child)
        else:
            if not isinstance(ret[child.tag], list):
                ret[child.tag] = [ret[child.tag]]
            ret[child.tag].append(process_xml_element(child))
    return ret

def get_tests(workflow_run_id: int, workflow_run_attempt: int) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    with TemporaryDirectory() as temp_dir:
        print('Using temporary directory:', temp_dir)
        os.chdir(temp_dir)
        s3_paths = download_s3_artifacts('test-report', workflow_run_id, workflow_run_attempt)
        for path in s3_paths:
            unzip(path)
        artifact_paths = download_gha_artifacts('test-report', workflow_run_id, workflow_run_attempt)
        for path in artifact_paths:
            unzip(path)
        test_cases = []
        for xml_report in Path('.').glob('**/*.xml'):
            test_cases.extend(parse_xml_report('testcase', xml_report, workflow_run_id, workflow_run_attempt))
        return test_cases

def get_tests_for_circleci(workflow_run_id: int, workflow_run_attempt: int) -> List[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    test_cases = []
    for xml_report in Path('.').glob('**/test/test-reports/**/*.xml'):
        test_cases.extend(parse_xml_report('testcase', xml_report, workflow_run_id, workflow_run_attempt))
    return test_cases

def summarize_test_cases(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    'Group test cases by classname, file, and job_id. We perform the aggregation\n    manually instead of using the `test-suite` XML tag because xmlrunner does\n    not produce reliable output for it.\n    '

    def get_key(test_case: Dict[str, Any]) -> Any:
        if False:
            i = 10
            return i + 15
        return (test_case.get('file'), test_case.get('classname'), test_case['job_id'], test_case['workflow_id'], test_case['workflow_run_attempt'], test_case['invoking_file'])

    def init_value(test_case: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {'file': test_case.get('file'), 'classname': test_case.get('classname'), 'job_id': test_case['job_id'], 'workflow_id': test_case['workflow_id'], 'workflow_run_attempt': test_case['workflow_run_attempt'], 'invoking_file': test_case['invoking_file'], 'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0, 'successes': 0, 'time': 0.0}
    ret = {}
    for test_case in test_cases:
        key = get_key(test_case)
        if key not in ret:
            ret[key] = init_value(test_case)
        ret[key]['tests'] += 1
        if 'failure' in test_case:
            ret[key]['failures'] += 1
        elif 'error' in test_case:
            ret[key]['errors'] += 1
        elif 'skipped' in test_case:
            ret[key]['skipped'] += 1
        else:
            ret[key]['successes'] += 1
        ret[key]['time'] += test_case['time']
    return list(ret.values())
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload test stats to Rockset')
    parser.add_argument('--workflow-run-id', required=True, help='id of the workflow to get artifacts from')
    parser.add_argument('--workflow-run-attempt', type=int, required=True, help='which retry of the workflow this is')
    parser.add_argument('--head-branch', required=True, help='Head branch of the workflow')
    parser.add_argument('--head-repository', required=True, help='Head repository of the workflow')
    parser.add_argument('--circleci', action='store_true', help='If this is being run through circleci')
    args = parser.parse_args()
    print(f'Workflow id is: {args.workflow_run_id}')
    if args.circleci:
        test_cases = get_tests_for_circleci(args.workflow_run_id, args.workflow_run_attempt)
    else:
        test_cases = get_tests(args.workflow_run_id, args.workflow_run_attempt)
    sys.stdout.flush()
    test_case_summary = summarize_test_cases(test_cases)
    upload_workflow_stats_to_s3(args.workflow_run_id, args.workflow_run_attempt, 'test_run_summary', test_case_summary)
    failed_tests_cases = []
    for test_case in test_cases:
        if 'rerun' in test_case or 'failure' in test_case or 'error' in test_case:
            failed_tests_cases.append(test_case)
    upload_workflow_stats_to_s3(args.workflow_run_id, args.workflow_run_attempt, 'failed_test_runs', failed_tests_cases)
    if args.head_branch == 'main' and args.head_repository == 'pytorch/pytorch':
        upload_workflow_stats_to_s3(args.workflow_run_id, args.workflow_run_attempt, 'test_run', test_cases)