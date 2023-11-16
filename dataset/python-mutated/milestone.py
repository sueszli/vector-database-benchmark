import datetime
import logging
import os
import sys
from itertools import groupby
from pathlib import Path
import click
import requests
VALID_TYPES = ('type: bug', 'type: feature', 'type: task')
SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent

def query_github(query, token):
    if False:
        while True:
            i = 10
    ' Hits the GitHub GraphQL API with the given query and returns the data or None.\n\n    '
    API_HEADERS = {'Authorization': f'Bearer {token}'}
    BASE_URL = 'https://api.github.com/graphql'
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        query_string = ' '.join((line.strip() for line in query.split('\n')))
        logging.debug('POST https://api.github.com/graphql; query:%s', query_string)
    response = requests.post(BASE_URL, json={'query': query}, headers=API_HEADERS)
    errors = response.json().get('errors', [])
    for error in errors:
        path = '/'.join(error['path'])
        msg = error['message']
        print(f'error: {path}: {msg}', file=sys.stderr)
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        logging.debug(f'Response {response.status_code}: {response.text}')
    return response.json()['data'] if not errors else None

def get_labels(data):
    if False:
        for i in range(10):
            print('nop')
    ' Returns the list of labels for the given issue or PR data.\n\n    '
    return [edge['node']['name'] for edge in data['node']['labels']['edges']]

def get_label_for(data, kind):
    if False:
        print('Hello World!')
    labels = get_labels(data)
    for label in labels:
        if label.startswith(kind):
            return label.replace(kind, '')
    return None

def get_label_type(data):
    if False:
        print('Hello World!')
    ' Returns the type label of the given issue or PR data, otherwise None.\n\n    '
    return get_label_for(data, 'type: ')

def get_label_component(data):
    if False:
        i = 10
        return i + 15
    ' Returns the component label of the given issue or PR data, otherwise None.\n\n    '
    return get_label_for(data, 'tag: component: ')

def description(data):
    if False:
        while True:
            i = 10
    ' Returns a humanized description of the given issue or PR data.\n\n    '
    component = get_label_component(data)
    component_str = '' if not component else f'[component: {component}] '
    return f"#{data['node']['number']} {component_str}{data['node']['title']}"

def get_milestone_number(title, token, allow_closed):
    if False:
        for i in range(10):
            print('nop')
    ' Iterates over all open milestones looking for one with the given title.\n\n    '
    open_str = '' if allow_closed else 'states: OPEN,'

    def helper(cursor=None):
        if False:
            for i in range(10):
                print('nop')
        cursor_or_null = f'"{cursor}"' if cursor else 'null'
        query = f'\n        {{\n            repository(owner: "bokeh", name: "bokeh") {{\n                milestones(first: 10, {open_str} after: {cursor_or_null}) {{\n                    edges {{\n                        node {{\n                            number\n                            title\n                        }}\n                    }}\n                    pageInfo {{\n                        endCursor\n                    }}\n                }}\n            }}\n        }}\n        '
        data = query_github(query, token)
        if not data:
            print('error: graphql query failure', file=sys.stderr)
            sys.exit(1)
        milestones = data['repository']['milestones']
        end_cursor = milestones['pageInfo']['endCursor']
        for edge in milestones['edges']:
            if edge['node']['title'] == title:
                return edge['node']['number']
        return helper(end_cursor) if end_cursor else None
    return helper()

def check_issue(data, problems):
    if False:
        while True:
            i = 10
    if data['node']['state'] != 'CLOSED':
        problems.append(f'issue not closed: {description(data)}')
    labels = get_labels(data)
    if 'reso: completed' not in labels:
        problems.append(f'issue missing resolution: {description(data)}')
    if sum((1 for label in labels if label.startswith('reso:'))) > 1:
        problems.append(f'issue has too many resolutions: {description(data)}')
    if any((label.startswith('status:') for label in labels)):
        problems.append(f'issue has a status: {description(data)}')
    num_types = sum((1 for label in labels if label.startswith('type:')))
    if num_types == 0:
        problems.append(f'issue does not have a type: {description(data)}')
    if num_types > 1:
        problems.append(f'issue has multiple types: {description(data)}')
    if any((label not in VALID_TYPES for label in labels if label.startswith('type:'))):
        problems.append(f'issue has an invalid type: {description(data)}')
    if any((label == 'TRIAGE' for label in labels)):
        problems.append(f'issue is in triage: {description(data)}')

def check_pr(data, problems):
    if False:
        while True:
            i = 10
    if data['node']['state'] != 'MERGED':
        problems.append(f'PR not merged: {description(data)}')
    labels = get_labels(data)
    if 'status: accepted' not in labels:
        problems.append(f'PR missing status: {description(data)}')
    if any((label.startswith('type:') for label in labels)):
        if 'reso: completed' not in labels:
            problems.append(f'issue PR missing resolution: {description(data)}')
        if sum((1 for label in labels if label.startswith('reso:'))) > 1:
            problems.append(f'issue PR has too many resolutions: {description(data)}')
    if sum((1 for label in labels if label.startswith('status:'))) > 1:
        problems.append(f'PR has too many statuses: {description(data)}')
    if sum((1 for label in labels if label.startswith('type:'))) > 1:
        problems.append(f'PR has multiple types: {description(data)}')
    if any((label not in VALID_TYPES for label in labels if label.startswith('type:'))):
        problems.append(f'PR has an invalid type: {description(data)}')
    if any((label == 'TRIAGE' for label in labels)):
        problems.append(f'PR is in triage: {description(data)}')

def get_milestone_items(title, token, allow_closed):
    if False:
        return 10
    " Returns the issues and PRs in the milestone with the given title,\n    otherwise None if the milestone doesn't exist.\n    "
    milestone_number = get_milestone_number(title, token, allow_closed)
    if not milestone_number:
        return None
    results = []

    def helper(kind, cursor=None):
        if False:
            print('Hello World!')
        cursor_or_null = f'"{cursor}"' if cursor else 'null'
        query = f'\n        {{\n            repository(owner: "bokeh", name: "bokeh") {{\n                milestone(number: {milestone_number}) {{\n                    {kind}(first: 100, after: {cursor_or_null}) {{\n                        edges {{\n                            node {{\n                                number\n                                title\n                                state\n                                labels(first: 20) {{\n                                    edges {{\n                                        node {{\n                                            name\n                                        }}\n                                    }}\n                                }}\n                            }}\n                        }}\n                        pageInfo {{\n                            endCursor\n                        }}\n                    }}\n                }}\n            }}\n        }}\n        '
        data = query_github(query, token)
        if not data:
            print('error: graphql query failure', file=sys.stderr)
            sys.exit(1)
        items = data['repository']['milestone'][kind]
        end_cursor = items['pageInfo']['endCursor']
        for edge in items['edges']:
            edge['kind'] = kind
            results.append(edge)
        if end_cursor:
            helper(kind, end_cursor)
    helper('issues')
    helper('pullRequests')
    return results

def check_milestone_items(items):
    if False:
        while True:
            i = 10
    problems = []
    for item in items:
        if item['kind'] == 'issues':
            check_issue(item, problems)
        elif item['kind'] == 'pullRequests':
            check_pr(item, problems)
    return problems

@click.command()
@click.argument('milestone')
@click.option('-l', '--log-level', type=click.Choice(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']), default='INFO')
@click.option('-v', '--verbose', count=True, help='Sets log level to DEBUG.')
@click.option('-c', '--check-only', default=False, is_flag=True, help='Only verify the milestone for compliance, do not output changelog section(s)')
@click.option('-a', '--allow-closed', default=False, is_flag=True, help='Allow processing of closed milestones')
def main(milestone, log_level, verbose, check_only, allow_closed):
    if False:
        print('Hello World!')
    ' Generates a bokeh changelog which includes the given milestone.\n\n    Requires that you set GH_TOKEN to your GitHub API Token. Exit code 2\n    indicates there was a verification problem whereas exit code 1 indicates a general\n    error in the script. Otherwise you can expect an exit code of 0 for success.\n    '
    log_level = 'DEBUG' if verbose else log_level
    logging.basicConfig(level=log_level)
    token = os.environ.get('GH_TOKEN', None)
    if not token:
        print('error: GH_TOKEN is not set', file=sys.stderr)
        sys.exit(1)
    items = get_milestone_items(milestone, token, allow_closed)
    if not items:
        print(f'error: no such milestone: {milestone}', file=sys.stderr)
        sys.exit(1)
    problems = check_milestone_items(items)
    for problem in problems:
        print(problem, file=sys.stderr)
    if len(problems) > 0:
        sys.exit(2)
    elif check_only:
        sys.exit(0)
    CHANGELOG = REPO_ROOT / 'docs' / 'CHANGELOG'
    with open(CHANGELOG) as f:
        old_changelog = f.read()
    out = open(CHANGELOG, mode='w')
    out.write(f'{datetime.date.today()} {milestone:>8}:\n')
    out.write('--------------------\n')
    grouping = lambda item: get_label_type(item) or 'none'
    items = sorted(items, key=grouping)
    for (group_type, group) in groupby(items, grouping):
        if group_type == 'bug':
            out.write('  * bugfixes:\n')
        elif group_type == 'feature':
            out.write('  * features:\n')
        elif group_type == 'task':
            out.write('  * tasks:\n')
        elif group_type == 'none':
            continue
        for item in group:
            out.write(f'    - {description(item)}\n')
        out.write('\n')
    out.write(old_changelog)
if __name__ == '__main__':
    main()