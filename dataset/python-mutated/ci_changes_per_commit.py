QUERY_COMMITS = '\nquery ($owner: String!, $name: String!, $pullNumber: Int!, $commitsPerPage: Int!, $beforeCommit: String) {\n  repository(owner: $owner, name: $name) {\n    pullRequest(number: $pullNumber) {\n      commits(last: $commitsPerPage, before: $beforeCommit) {\n        totalCount\n        pageInfo {\n          startCursor\n          hasPreviousPage\n        }\n        nodes {\n          commit {\n            checkSuites(first: 100) {\n              nodes {\n                conclusion\n                workflowRun {\n                  workflow {\n                    name\n                  }\n                }\n                id\n              }\n              totalCount\n            }\n            oid\n          }\n        }\n      }\n    }\n  }\n}\n'
QUERY_CHECK_RUNS = '\nquery ($checkSuiteID: ID!,\n       $afterFailedRun: String, $afterIncompleteRun: String,\n       $includeFailedRuns: Boolean!, $includeIncompleteRuns: Boolean!) {\n  node(id: $checkSuiteID) {\n    ... on CheckSuite {\n      failedRuns: checkRuns(\n        first: 100\n        after: $afterFailedRun\n        filterBy: {checkType: LATEST, conclusions: [ACTION_REQUIRED, TIMED_OUT, CANCELLED, FAILURE, NEUTRAL, STARTUP_FAILURE]}\n      ) @include(if: $includeFailedRuns) {\n        nodes {\n          name\n        }\n        pageInfo {\n          endCursor\n          hasNextPage\n        }\n      }\n      incompleteRuns: checkRuns(\n        first: 100\n        after: $afterIncompleteRun\n        filterBy: {checkType: LATEST, statuses: [QUEUED, IN_PROGRESS, WAITING, PENDING, REQUESTED]}\n      ) @include(if: $includeIncompleteRuns) {\n        nodes {\n          name\n        }\n        pageInfo {\n          endCursor\n          hasNextPage\n        }\n      }\n    }\n  }\n}\n'
import os
import re
import json
import requests
query_variables_commits = {'owner': '', 'name': '', 'pullNumber': int(os.environ['PULL']), 'commitsPerPage': 20, 'beforeCommit': None}
query_variables_check_runs = {'checkSuiteID': '', 'afterFailedRun': None, 'afterIncompleteRun': None, 'includeFailedRuns': True, 'includeIncompleteRuns': True}
headers = {'Authorization': f"Bearer {os.environ['GITHUB_TOKEN']}"}

class Query:

    def __init__(self, query, variables={}, headers={}):
        if False:
            for i in range(10):
                print('nop')
        self.query = query
        self.variables = variables
        self.headers = headers

    def paginate(self, page_info, name):
        if False:
            while True:
                i = 10
        has_page = page_info['hasNextPage' if name.startswith('after') else 'hasPreviousPage']
        if has_page:
            self.variables[name] = page_info['endCursor' if name.startswith('after') else 'startCursor']
        return has_page

    def fetch(self):
        if False:
            for i in range(10):
                print('nop')
        request = requests.post('https://api.github.com/graphql', json={'query': self.query, 'variables': self.variables}, headers=self.headers)
        if request.status_code == 200:
            return request.json()
        else:
            print(request.json())
            raise Exception('Query Failed: {}'.format(request.status_code))

def set_output(name, value):
    if False:
        for i in range(10):
            print('nop')
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'at') as f:
            print(f'{name}={value}', file=f)
    else:
        print(f"Would set GitHub actions output {name} to '{value}'")

def get_commit_depth_and_check_suite(query_commits):
    if False:
        return 10
    commit_depth = 0
    while True:
        commits = query_commits.fetch()['data']['repository']['pullRequest']['commits']
        if commits['totalCount'] > 0:
            nodes = commits['nodes']
            nodes.reverse()
            if nodes[0]['commit']['oid'] == os.environ['EXCLUDE_COMMIT']:
                nodes.pop(0)
            for commit in nodes:
                commit_depth += 1
                commit = commit['commit']
                commit_sha = commit['oid']
                check_suites = commit['checkSuites']
                if check_suites['totalCount'] > 0:
                    for check_suite in check_suites['nodes']:
                        if check_suite['workflowRun']['workflow']['name'] == 'Build CI':
                            return [{'sha': commit_sha, 'depth': commit_depth}, check_suite['id'] if check_suite['conclusion'] != 'SUCCESS' else None]
        if not query_commits.paginate(commits['pageInfo'], 'beforeCommit'):
            return [None, None]

def get_bad_check_runs(query_check_runs):
    if False:
        return 10
    bad_runs = {}
    more_pages = True
    run_types = ['failed', 'incomplete']
    have_dependent_jobs = ['scheduler', 'mpy-cross', 'tests']
    while more_pages:
        check_runs = query_check_runs.fetch()['data']['node']
        more_pages = False
        for run_type in run_types:
            run_type_camel = run_type.capitalize() + 'Run'
            run_type = run_type + 'Runs'
            for check_run in check_runs[run_type]['nodes']:
                name = check_run['name']
                if any([name.startswith(job) for job in have_dependent_jobs]):
                    return {}
                if name.startswith('ports'):
                    matrix_job = name.rsplit(' (', 1)[1][:-1]
                    bad_runs.setdefault('ports', []).append(matrix_job)
                else:
                    bad_runs[name] = True
            if query_check_runs.paginate(check_runs[run_type]['pageInfo'], 'after' + run_type_camel):
                query_check_runs.variables['include' + run_type_camel] = True
                more_pages = True
    return bad_runs

def set_commit(commit):
    if False:
        i = 10
        return i + 15
    set_output('commit_sha', commit['sha'])
    set_output('commit_depth', commit['depth'])

def main():
    if False:
        i = 10
        return i + 15
    query_commits = Query(QUERY_COMMITS, query_variables_commits, headers)
    (query_commits.variables['owner'], query_commits.variables['name']) = os.environ['REPO'].split('/')
    (commit, check_suite) = get_commit_depth_and_check_suite(query_commits)
    if not check_suite:
        if commit:
            set_commit(commit)
        else:
            print('Abort: No check suite found')
        quit()
    query_check_runs = Query(QUERY_CHECK_RUNS, query_variables_check_runs, headers)
    query_check_runs.variables['checkSuiteID'] = check_suite
    check_runs = get_bad_check_runs(query_check_runs)
    if not check_runs:
        print('Abort: No check runs found')
        quit()
    set_commit(commit)
    set_output('check_runs', json.dumps(check_runs))
if __name__ == '__main__':
    main()