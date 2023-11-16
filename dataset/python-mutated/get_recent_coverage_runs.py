from __future__ import annotations
from ansible.utils.color import stringc
import requests
import sys
import datetime
BRANCH = 'devel'
PIPELINE_ID = 20
MAX_AGE = datetime.timedelta(hours=24)
if len(sys.argv) > 1:
    BRANCH = sys.argv[1]

def get_coverage_runs():
    if False:
        i = 10
        return i + 15
    list_response = requests.get('https://dev.azure.com/ansible/ansible/_apis/pipelines/%s/runs?api-version=6.0-preview.1' % PIPELINE_ID)
    list_response.raise_for_status()
    runs = list_response.json()
    coverage_runs = []
    for run_summary in runs['value'][0:1000]:
        run_response = requests.get(run_summary['url'])
        if run_response.status_code == 500 and 'Cannot serialize type Microsoft.Azure.Pipelines.WebApi.ContainerResource' in run_response.json()['message']:
            break
        run_response.raise_for_status()
        run = run_response.json()
        if run['resources']['repositories']['self']['refName'] != 'refs/heads/%s' % BRANCH:
            continue
        if 'finishedDate' in run_summary:
            age = datetime.datetime.now() - datetime.datetime.strptime(run['finishedDate'].split('.')[0], '%Y-%m-%dT%H:%M:%S')
            if age > MAX_AGE:
                break
        artifact_response = requests.get('https://dev.azure.com/ansible/ansible/_apis/build/builds/%s/artifacts?api-version=6.0' % run['id'])
        artifact_response.raise_for_status()
        artifacts = artifact_response.json()['value']
        if any((a['name'].startswith('Coverage') for a in artifacts)):
            coverage_runs.append(run)
    return coverage_runs

def pretty_coverage_runs(runs):
    if False:
        i = 10
        return i + 15
    ended = []
    in_progress = []
    for run in runs:
        if run.get('finishedDate'):
            ended.append(run)
        else:
            in_progress.append(run)
    for run in sorted(ended, key=lambda x: x['finishedDate']):
        if run['result'] == 'succeeded':
            print('🙂 [%s] https://dev.azure.com/ansible/ansible/_build/results?buildId=%s (%s)' % (stringc('PASS', 'green'), run['id'], run['finishedDate']))
        else:
            print('😢 [%s] https://dev.azure.com/ansible/ansible/_build/results?buildId=%s (%s)' % (stringc('FAIL', 'red'), run['id'], run['finishedDate']))
    if in_progress:
        print('The following runs are ongoing:')
        for run in in_progress:
            print('🤔 [%s] https://dev.azure.com/ansible/ansible/_build/results?buildId=%s' % (stringc('FATE', 'yellow'), run['id']))

def main():
    if False:
        print('Hello World!')
    pretty_coverage_runs(get_coverage_runs())
if __name__ == '__main__':
    main()