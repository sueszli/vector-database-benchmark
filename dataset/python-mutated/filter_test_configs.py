import json
import os
import re
import subprocess
import sys
import warnings
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.request import Request, urlopen
import yaml
REENABLE_TEST_REGEX = '(?i)(Close(d|s)?|Resolve(d|s)?|Fix(ed|es)?) (#|https://github.com/pytorch/pytorch/issues/)([0-9]+)'
PREFIX = 'test-config/'
VALID_TEST_CONFIG_LABELS = {f'{PREFIX}{label}' for label in {'backwards_compat', 'crossref', 'default', 'deploy', 'distributed', 'docs_tests', 'dynamo', 'force_on_cpu', 'functorch', 'inductor', 'inductor_distributed', 'inductor_huggingface', 'inductor_timm', 'inductor_torchbench', 'jit_legacy', 'multigpu', 'nogpu_AVX512', 'nogpu_NO_AVX2', 'slow', 'tsan', 'xla'}}

def is_cuda_or_rocm_job(job_name: Optional[str]) -> bool:
    if False:
        i = 10
        return i + 15
    if not job_name:
        return False
    return 'cuda' in job_name or 'rocm' in job_name
SUPPORTED_PERIODICAL_MODES: Dict[str, Callable[[Optional[str]], bool]] = {'mem_leak_check': is_cuda_or_rocm_job, 'rerun_disabled_tests': lambda job_name: True}
DISABLED_JOBS_URL = 'https://ossci-metrics.s3.amazonaws.com/disabled-jobs.json'
UNSTABLE_JOBS_URL = 'https://ossci-metrics.s3.amazonaws.com/unstable-jobs.json'
JOB_NAME_SEP = '/'
BUILD_JOB_NAME = 'build'
TEST_JOB_NAME = 'test'
BUILD_AND_TEST_JOB_NAME = 'build-and-test'
JOB_NAME_CFG_REGEX = re.compile('(?P<job>[\\w-]+)\\s+\\((?P<cfg>[\\w-]+)\\)')
EXCLUDED_BRANCHES = ['nightly']
MEM_LEAK_LABEL = 'enable-mem-leak-check'

class IssueType(Enum):
    DISABLED = 'disabled'
    UNSTABLE = 'unstable'

def parse_args() -> Any:
    if False:
        print('Hello World!')
    from argparse import ArgumentParser
    parser = ArgumentParser('Filter all test configurations and keep only requested ones')
    parser.add_argument('--test-matrix', type=str, required=True, help='the original test matrix')
    parser.add_argument('--workflow', type=str, help='the name of the current workflow, i.e. pull')
    parser.add_argument('--job-name', type=str, help='the name of the current job, i.e. linux-focal-py3.8-gcc7 / build')
    parser.add_argument('--pr-number', type=str, help='the pull request number')
    parser.add_argument('--tag', type=str, help='the associated tag if it exists')
    parser.add_argument('--event-name', type=str, help='name of the event that triggered the job (pull, schedule, etc)')
    parser.add_argument('--schedule', type=str, help='cron schedule that triggered the job')
    parser.add_argument('--branch', type=str, default='main', help='the branch name')
    return parser.parse_args()

@lru_cache(maxsize=None)
def get_pr_info(pr_number: int) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    '\n    Dynamically get PR information\n    '
    pytorch_repo = os.environ.get('GITHUB_REPOSITORY', 'pytorch/pytorch')
    pytorch_github_api = f'https://api.github.com/repos/{pytorch_repo}'
    github_token = os.environ['GITHUB_TOKEN']
    headers = {'Accept': 'application/vnd.github.v3+json', 'Authorization': f'token {github_token}'}
    json_response: Dict[str, Any] = download_json(url=f'{pytorch_github_api}/issues/{pr_number}', headers=headers)
    if not json_response:
        warnings.warn(f'Failed to get the labels for #{pr_number}')
        return {}
    return json_response

def get_labels(pr_number: int) -> Set[str]:
    if False:
        while True:
            i = 10
    '\n    Dynamically get the latest list of labels from the pull request\n    '
    pr_info = get_pr_info(pr_number)
    return {label.get('name') for label in pr_info.get('labels', []) if label.get('name')}

def filter(test_matrix: Dict[str, List[Any]], labels: Set[str]) -> Dict[str, List[Any]]:
    if False:
        while True:
            i = 10
    '\n    Select the list of test config to run from the test matrix. The logic works\n    as follows:\n\n    If the PR has one or more labels as specified in the VALID_TEST_CONFIG_LABELS set, only\n    these test configs will be selected.  This also works with ciflow labels, for example,\n    if a PR has both ciflow/trunk and test-config/functorch, only trunk functorch builds\n    and tests will be run\n\n    If the PR has none of the test-config label, all tests are run as usual.\n    '
    filtered_test_matrix: Dict[str, List[Any]] = {'include': []}
    for entry in test_matrix.get('include', []):
        config_name = entry.get('config', '')
        if not config_name:
            continue
        label = f'{PREFIX}{config_name.strip()}'
        if label in labels:
            print(f'Select {config_name} because label {label} is presented in the pull request by the time the test starts')
            filtered_test_matrix['include'].append(entry)
    valid_test_config_labels = labels.intersection(VALID_TEST_CONFIG_LABELS)
    if not filtered_test_matrix['include'] and (not valid_test_config_labels):
        return test_matrix
    else:
        return filtered_test_matrix

def set_periodic_modes(test_matrix: Dict[str, List[Any]], job_name: Optional[str]) -> Dict[str, List[Any]]:
    if False:
        while True:
            i = 10
    '\n    Apply all periodic modes when running under a schedule\n    '
    scheduled_test_matrix: Dict[str, List[Any]] = {'include': []}
    for config in test_matrix.get('include', []):
        for (mode, cond) in SUPPORTED_PERIODICAL_MODES.items():
            if not cond(job_name):
                continue
            cfg = config.copy()
            cfg[mode] = mode
            scheduled_test_matrix['include'].append(cfg)
    return scheduled_test_matrix

def mark_unstable_jobs(workflow: str, job_name: str, test_matrix: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    if False:
        print('Hello World!')
    '\n    Check the list of unstable jobs and mark them accordingly. Note that if a job\n    is unstable, all its dependents will also be marked accordingly\n    '
    return process_jobs(workflow=workflow, job_name=job_name, test_matrix=test_matrix, issue_type=IssueType.UNSTABLE, url=UNSTABLE_JOBS_URL)

def remove_disabled_jobs(workflow: str, job_name: str, test_matrix: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    if False:
        print('Hello World!')
    '\n    Check the list of disabled jobs, remove the current job and all its dependents\n    if it exists in the list\n    '
    return process_jobs(workflow=workflow, job_name=job_name, test_matrix=test_matrix, issue_type=IssueType.DISABLED, url=DISABLED_JOBS_URL)

def _filter_jobs(test_matrix: Dict[str, List[Any]], issue_type: IssueType, target_cfg: Optional[str]=None) -> Dict[str, List[Any]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    An utility function used to actually apply the job filter\n    '
    filtered_test_matrix: Dict[str, List[Any]] = {'include': []}
    if issue_type == IssueType.DISABLED:
        if target_cfg:
            filtered_test_matrix['include'] = [r for r in test_matrix['include'] if r.get('config', '') != target_cfg]
        return filtered_test_matrix
    if issue_type == IssueType.UNSTABLE:
        for r in test_matrix['include']:
            cpy = r.copy()
            if target_cfg and r.get('config', '') == target_cfg or not target_cfg:
                cpy[IssueType.UNSTABLE.value] = IssueType.UNSTABLE.value
            filtered_test_matrix['include'].append(cpy)
        return filtered_test_matrix
    return test_matrix

def process_jobs(workflow: str, job_name: str, test_matrix: Dict[str, List[Any]], issue_type: IssueType, url: str) -> Dict[str, List[Any]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Both disabled and unstable jobs are in the following format:\n\n    {\n        "WORKFLOW / PLATFORM / JOB (CONFIG)": [\n            AUTHOR,\n            ISSUE_NUMBER,\n            ISSUE_URL,\n            WORKFLOW,\n            PLATFORM,\n            JOB (CONFIG),\n        ],\n        "pull / linux-bionic-py3.8-clang9 / test (dynamo)": [\n            "pytorchbot",\n            "94861",\n            "https://github.com/pytorch/pytorch/issues/94861",\n            "pull",\n            "linux-bionic-py3.8-clang9",\n            "test (dynamo)",\n        ],\n    }\n    '
    try:
        (current_platform, _) = (n.strip() for n in job_name.split(JOB_NAME_SEP, 1) if n)
    except ValueError as error:
        warnings.warn(f'Invalid job name {job_name}, returning')
        return test_matrix
    for record in download_json(url=url, headers={}).values():
        (author, _, target_url, target_workflow, target_platform, target_job_cfg) = record
        if target_workflow != workflow:
            continue
        cleanup_regex = f'(-{BUILD_JOB_NAME}|-{TEST_JOB_NAME})$'
        target_platform_no_suffix = re.sub(cleanup_regex, '', target_platform)
        current_platform_no_suffix = re.sub(cleanup_regex, '', current_platform)
        if target_platform != current_platform and target_platform_no_suffix != current_platform_no_suffix:
            continue
        if not target_job_cfg:
            print(f'Issue {target_url} created by {author} has {issue_type.value} ' + f'all CI jobs for {workflow} / {job_name}')
            return _filter_jobs(test_matrix=test_matrix, issue_type=issue_type)
        if target_job_cfg == BUILD_JOB_NAME:
            print(f'Issue {target_url} created by {author} has {issue_type.value} ' + f'the build job for {workflow} / {job_name}')
            return _filter_jobs(test_matrix=test_matrix, issue_type=issue_type)
        if target_job_cfg in (TEST_JOB_NAME, BUILD_AND_TEST_JOB_NAME):
            print(f'Issue {target_url} created by {author} has {issue_type.value} ' + f'all the test jobs for {workflow} / {job_name}')
            return _filter_jobs(test_matrix=test_matrix, issue_type=issue_type)
        m = JOB_NAME_CFG_REGEX.match(target_job_cfg)
        if m:
            target_job = m.group('job')
            if target_job in (TEST_JOB_NAME, BUILD_AND_TEST_JOB_NAME):
                target_cfg = m.group('cfg')
                test_matrix = _filter_jobs(test_matrix=test_matrix, issue_type=issue_type, target_cfg=target_cfg)
        else:
            warnings.warn(f'Found a matching {issue_type.value} issue {target_url} for {workflow} / {job_name}, ' + f'but the name {target_job_cfg} is invalid')
    return test_matrix

def download_json(url: str, headers: Dict[str, str], num_retries: int=3) -> Any:
    if False:
        while True:
            i = 10
    for _ in range(num_retries):
        try:
            req = Request(url=url, headers=headers)
            content = urlopen(req, timeout=5).read().decode('utf-8')
            return json.loads(content)
        except Exception as e:
            warnings.warn(f'Could not download {url}: {e}')
    warnings.warn(f'All {num_retries} retries exhausted, downloading {url} failed')
    return {}

def set_output(name: str, val: Any) -> None:
    if False:
        i = 10
        return i + 15
    if os.getenv('GITHUB_OUTPUT'):
        with open(str(os.getenv('GITHUB_OUTPUT')), 'a') as env:
            print(f'{name}={val}', file=env)
    else:
        print(f'::set-output name={name}::{val}')

def parse_reenabled_issues(s: Optional[str]) -> List[str]:
    if False:
        print('Hello World!')
    if not s:
        return []
    issue_numbers = [x[5] for x in re.findall(REENABLE_TEST_REGEX, s)]
    return issue_numbers

def get_reenabled_issues(pr_body: str='') -> List[str]:
    if False:
        while True:
            i = 10
    default_branch = os.getenv('GIT_DEFAULT_BRANCH', 'main')
    try:
        commit_messages = subprocess.check_output(f'git cherry -v {default_branch}'.split(' ')).decode('utf-8')
    except Exception as e:
        warnings.warn(f'failed to get commit messages: {e}')
        commit_messages = ''
    return parse_reenabled_issues(pr_body) + parse_reenabled_issues(commit_messages)

def perform_misc_tasks(labels: Set[str], test_matrix: Dict[str, List[Any]], job_name: str, pr_body: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    In addition to apply the filter logic, the script also does the following\n    misc tasks to set keep-going and is-unstable variables\n    '
    set_output('keep-going', 'keep-going' in labels)
    is_unstable = job_name and IssueType.UNSTABLE.value in job_name
    if not is_unstable and test_matrix:
        is_unstable = all((IssueType.UNSTABLE.value in r for r in test_matrix['include']))
    set_output('is-unstable', is_unstable)
    set_output('reenabled-issues', ','.join(get_reenabled_issues(pr_body=pr_body)))
    if MEM_LEAK_LABEL in labels:
        for config in test_matrix.get('include', []):
            if is_cuda_or_rocm_job(job_name):
                config['mem_leak_check'] = 'mem_leak_check'

def main() -> None:
    if False:
        print('Hello World!')
    args = parse_args()
    test_matrix = yaml.safe_load(args.test_matrix)
    if test_matrix is None:
        warnings.warn(f"Invalid test matrix input '{args.test_matrix}', exiting")
        set_output('is-test-matrix-empty', True)
        sys.exit(0)
    pr_number = args.pr_number
    tag = args.tag
    tag_regex = re.compile('^ciflow/\\w+/(?P<pr_number>\\d+)$')
    labels = set()
    if pr_number:
        labels = get_labels(int(pr_number))
        filtered_test_matrix = filter(test_matrix, labels)
    elif tag:
        m = tag_regex.match(tag)
        if m:
            pr_number = m.group('pr_number')
            labels = get_labels(int(pr_number))
            filtered_test_matrix = filter(test_matrix, labels)
        else:
            filtered_test_matrix = test_matrix
    else:
        filtered_test_matrix = test_matrix
    if args.event_name == 'schedule' and args.schedule == '29 8 * * *':
        filtered_test_matrix = set_periodic_modes(filtered_test_matrix, args.job_name)
    if args.workflow and args.job_name and (args.branch not in EXCLUDED_BRANCHES):
        filtered_test_matrix = remove_disabled_jobs(args.workflow, args.job_name, filtered_test_matrix)
        filtered_test_matrix = mark_unstable_jobs(args.workflow, args.job_name, filtered_test_matrix)
    pr_body = get_pr_info(int(pr_number)).get('body', '') if pr_number else ''
    perform_misc_tasks(labels=labels, test_matrix=filtered_test_matrix, job_name=args.job_name, pr_body=pr_body)
    set_output('test-matrix', json.dumps(filtered_test_matrix))
    filtered_test_matrix_len = len(filtered_test_matrix.get('include', []))
    set_output('is-test-matrix-empty', filtered_test_matrix_len == 0)
if __name__ == '__main__':
    main()