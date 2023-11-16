import json
import logging
import os
from typing import List
from typing import Optional
from typing import Tuple
import requests
from apache_beam.testing.analyzers import constants
from apache_beam.testing.analyzers.perf_analysis_utils import MetricContainer
from apache_beam.testing.analyzers.perf_analysis_utils import TestConfigContainer
try:
    _GITHUB_TOKEN: Optional[str] = os.environ['GITHUB_TOKEN']
except KeyError as e:
    _GITHUB_TOKEN = None
    logging.warning('A Github Personal Access token is required to create Github Issues.')
_GITHUB_REPO_OWNER = os.environ.get('REPO_OWNER', 'apache')
_GITHUB_REPO_NAME = os.environ.get('REPO_NAME', 'beam')
_HEADERS = {'Authorization': 'token {}'.format(_GITHUB_TOKEN), 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
_ISSUE_TITLE_TEMPLATE = '\n  Performance Regression or Improvement: {}:{}\n'
_ISSUE_DESCRIPTION_TEMPLATE = '\n  Performance change found in the\n  test: `{}` for the metric: `{}`.\n\n  For more information on how to triage the alerts, please look at\n  `Triage performance alert issues` section of the [README](https://github.com/apache/beam/tree/master/sdks/python/apache_beam/testing/analyzers/README.md#triage-performance-alert-issues).\n'
_METRIC_INFO_TEMPLATE = 'timestamp: {}, metric_value: {}'
_AWAITING_TRIAGE_LABEL = 'awaiting triage'
_PERF_ALERT_LABEL = 'perf-alert'
_REQUEST_TIMEOUT_SECS = 60

def create_issue(title: str, description: str, labels: Optional[List[str]]=None) -> Tuple[int, str]:
    if False:
        i = 10
        return i + 15
    '\n  Create an issue with title, description with a label.\n\n  Args:\n    title:  GitHub issue title.\n    description: GitHub issue description.\n    labels: Labels used to tag the GitHub issue.\n  Returns:\n    Tuple containing GitHub issue number and issue URL.\n  '
    url = 'https://api.github.com/repos/{}/{}/issues'.format(_GITHUB_REPO_OWNER, _GITHUB_REPO_NAME)
    data = {'owner': _GITHUB_REPO_OWNER, 'repo': _GITHUB_REPO_NAME, 'title': title, 'body': description, 'labels': [_AWAITING_TRIAGE_LABEL, _PERF_ALERT_LABEL]}
    if labels:
        data['labels'].extend(labels)
    response = requests.post(url=url, data=json.dumps(data), headers=_HEADERS, timeout=_REQUEST_TIMEOUT_SECS).json()
    return (response['number'], response['html_url'])

def comment_on_issue(issue_number: int, comment_description: str) -> Tuple[bool, str]:
    if False:
        while True:
            i = 10
    '\n  This method looks for an issue with provided issue_number. If an open\n  issue is found, comment on the open issue with provided description else\n  do nothing.\n\n  Args:\n    issue_number: A GitHub issue number.\n    comment_description: If an issue with issue_number is open,\n      then comment on the issue with the using comment_description.\n  Returns:\n    tuple[bool, Optional[str]] indicating if a comment was added to\n      issue, and the comment URL.\n  '
    url = 'https://api.github.com/repos/{}/{}/issues/{}'.format(_GITHUB_REPO_OWNER, _GITHUB_REPO_NAME, issue_number)
    open_issue_response = requests.get(url, json.dumps({'owner': _GITHUB_REPO_OWNER, 'repo': _GITHUB_REPO_NAME, 'issue_number': issue_number}, default=str), headers=_HEADERS, timeout=_REQUEST_TIMEOUT_SECS).json()
    if open_issue_response['state'] == 'open':
        data = {'owner': _GITHUB_REPO_OWNER, 'repo': _GITHUB_REPO_NAME, 'body': comment_description, issue_number: issue_number}
        response = requests.post(open_issue_response['comments_url'], json.dumps(data), headers=_HEADERS, timeout=_REQUEST_TIMEOUT_SECS)
        return (True, response.json()['html_url'])
    return (False, '')

def add_awaiting_triage_label(issue_number: int):
    if False:
        return 10
    url = 'https://api.github.com/repos/{}/{}/issues/{}/labels'.format(_GITHUB_REPO_OWNER, _GITHUB_REPO_NAME, issue_number)
    requests.post(url, json.dumps({'labels': [_AWAITING_TRIAGE_LABEL]}), headers=_HEADERS, timeout=_REQUEST_TIMEOUT_SECS)

def get_issue_description(test_config_container: TestConfigContainer, metric_container: MetricContainer, change_point_index: int, max_results_to_display: int=5) -> str:
    if False:
        print('Hello World!')
    '\n  Args:\n    test_config_container: TestConfigContainer containing test metadata.\n    metric_container: MetricContainer containing metric data.\n    change_point_index: Index of the change point in the metric data.\n    max_results_to_display: Max number of results to display from the change\n      point index, in both directions of the change point index.\n\n  Returns:\n    str: Description used to fill the GitHub issues description.\n  '
    description = []
    description.append(_ISSUE_DESCRIPTION_TEMPLATE.format(test_config_container.test_id, test_config_container.metric_name))
    if test_config_container.test_name:
        description.append('`test_name:` ' + f'{test_config_container.test_name}')
    if test_config_container.test_description:
        description.append('`Test description:` ' + f'{test_config_container.test_description}')
    description.append('```')
    runs_to_display = []
    max_timestamp_index = min(change_point_index + max_results_to_display, len(metric_container.values) - 1)
    min_timestamp_index = max(0, change_point_index - max_results_to_display)
    for i in reversed(range(min_timestamp_index, max_timestamp_index + 1)):
        row_template = _METRIC_INFO_TEMPLATE.format(metric_container.timestamps[i].ctime(), format(metric_container.values[i], '.2f'))
        if i == change_point_index:
            row_template += constants._ANOMALY_MARKER
        runs_to_display.append(row_template)
    description.append(os.linesep.join(runs_to_display))
    description.append('```')
    return (2 * os.linesep).join(description)

def report_change_point_on_issues(title: str, description: str, labels: Optional[List[str]]=None, existing_issue_number: Optional[int]=None) -> Tuple[int, str]:
    if False:
        i = 10
        return i + 15
    '\n  Comments the description on the existing issue (if provided and still open),\n   or creates a new issue.\n  '
    if existing_issue_number is not None:
        (commented_on_issue, issue_url) = comment_on_issue(issue_number=existing_issue_number, comment_description=description)
        if commented_on_issue:
            add_awaiting_triage_label(issue_number=existing_issue_number)
            return (existing_issue_number, issue_url)
    return create_issue(title=title, description=description, labels=labels)