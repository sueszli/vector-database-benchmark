"""Get the most recent status of workflow for the current PR.

[usage]
    python get_workflow_status.py TRIGGER_PHRASE

TRIGGER_PHRASE: Code phrase that triggers workflow.
"""
import json
from os import environ
from sys import argv, exit
from time import sleep
try:
    from urllib import request
except ImportError:
    import urllib2 as request

def get_runs(trigger_phrase):
    if False:
        for i in range(10):
            print('nop')
    'Get all triggering workflow comments in the current PR.\n\n    Parameters\n    ----------\n    trigger_phrase : str\n        Code phrase that triggers workflow.\n\n    Returns\n    -------\n    pr_runs : list\n        List of comment objects sorted by the time of creation in decreasing order.\n    '
    pr_runs = []
    if environ.get('GITHUB_EVENT_NAME', '') == 'pull_request':
        pr_number = int(environ.get('GITHUB_REF').split('/')[-2])
        page = 1
        while True:
            req = request.Request(url='{}/repos/microsoft/LightGBM/issues/{}/comments?page={}&per_page=100'.format(environ.get('GITHUB_API_URL'), pr_number, page), headers={'Accept': 'application/vnd.github.v3+json'})
            url = request.urlopen(req)
            data = json.loads(url.read().decode('utf-8'))
            url.close()
            if not data:
                break
            runs_on_page = [i for i in data if i['author_association'].lower() in {'owner', 'member', 'collaborator'} and i['body'].startswith('/gha run {}'.format(trigger_phrase))]
            pr_runs.extend(runs_on_page)
            page += 1
    return pr_runs[::-1]

def get_status(runs):
    if False:
        while True:
            i = 10
    "Get the most recent status of workflow for the current PR.\n\n    Parameters\n    ----------\n    runs : list\n        List of comment objects sorted by the time of creation in decreasing order.\n\n    Returns\n    -------\n    status : str\n        The most recent status of workflow.\n        Can be 'success', 'failure' or 'in-progress'.\n    "
    status = 'success'
    for run in runs:
        body = run['body']
        if 'Status: ' in body:
            if 'Status: skipped' in body:
                continue
            if 'Status: failure' in body:
                status = 'failure'
                break
            if 'Status: success' in body:
                status = 'success'
                break
        else:
            status = 'in-progress'
            break
    return status
if __name__ == '__main__':
    trigger_phrase = argv[1]
    while True:
        status = get_status(get_runs(trigger_phrase))
        if status != 'in-progress':
            break
        sleep(60)
    if status == 'failure':
        exit(1)