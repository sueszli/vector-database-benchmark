import argparse
import json
import os
import re
import sys
import time
import urllib
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

def parse_json_and_links(conn: Any) -> Tuple[Any, Dict[str, Dict[str, str]]]:
    if False:
        while True:
            i = 10
    links = {}
    if 'Link' in conn.headers:
        for elem in re.split(', *<', conn.headers['Link']):
            try:
                (url, params_) = elem.split(';', 1)
            except ValueError:
                continue
            url = urllib.parse.unquote(url.strip('<> '))
            qparams = urllib.parse.parse_qs(params_.strip(), separator=';')
            params = {k: v[0].strip('"') for (k, v) in qparams.items() if type(v) is list and len(v) > 0}
            params['url'] = url
            if 'rel' in params:
                links[params['rel']] = params
    return (json.load(conn), links)

def fetch_url(url: str, *, headers: Optional[Dict[str, str]]=None, reader: Callable[[Any], Any]=lambda x: x.read(), retries: Optional[int]=3, backoff_timeout: float=0.5) -> Any:
    if False:
        i = 10
        return i + 15
    if headers is None:
        headers = {}
    try:
        with urlopen(Request(url, headers=headers)) as conn:
            return reader(conn)
    except urllib.error.HTTPError as err:
        if isinstance(retries, (int, float)) and retries > 0:
            time.sleep(backoff_timeout)
            return fetch_url(url, headers=headers, reader=reader, retries=retries - 1, backoff_timeout=backoff_timeout)
        exception_message = ('Is github alright?', f"Recieved status code '{err.code}' when attempting to retrieve {url}:\n", f'{err.reason}\n\nheaders={err.headers}')
        raise RuntimeError(exception_message) from err

def parse_args() -> Any:
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('workflow_run_id', help='The id of the workflow run, should be GITHUB_RUN_ID')
    parser.add_argument('runner_name', help='The name of the runner to retrieve the job id, should be RUNNER_NAME')
    return parser.parse_args()

def fetch_jobs(url: str, headers: Dict[str, str]) -> List[Dict[str, str]]:
    if False:
        while True:
            i = 10
    (response, links) = fetch_url(url, headers=headers, reader=parse_json_and_links)
    jobs = response['jobs']
    assert type(jobs) is list
    while 'next' in links.keys():
        (response, links) = fetch_url(links['next']['url'], headers=headers, reader=parse_json_and_links)
        jobs.extend(response['jobs'])
    return jobs

def find_job_id_name(args: Any) -> Tuple[str, str]:
    if False:
        print('Hello World!')
    PYTORCH_REPO = os.environ.get('GITHUB_REPOSITORY', 'pytorch/pytorch')
    PYTORCH_GITHUB_API = f'https://api.github.com/repos/{PYTORCH_REPO}'
    GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
    REQUEST_HEADERS = {'Accept': 'application/vnd.github.v3+json', 'Authorization': 'token ' + GITHUB_TOKEN}
    url = f'{PYTORCH_GITHUB_API}/actions/runs/{args.workflow_run_id}/jobs?per_page=100'
    jobs = fetch_jobs(url, REQUEST_HEADERS)
    jobs.sort(key=lambda job: job['started_at'], reverse=True)
    for job in jobs:
        if job['runner_name'] == args.runner_name:
            return (job['id'], job['name'])
    raise RuntimeError(f"Can't find job id for runner {args.runner_name}")

def set_output(name: str, val: Any) -> None:
    if False:
        print('Hello World!')
    if os.getenv('GITHUB_OUTPUT'):
        with open(str(os.getenv('GITHUB_OUTPUT')), 'a') as env:
            print(f'{name}={val}', file=env)
        print(f'setting {name}={val}')
    else:
        print(f'::set-output name={name}::{val}')

def main() -> None:
    if False:
        print('Hello World!')
    args = parse_args()
    try:
        (job_id, job_name) = find_job_id_name(args)
        set_output('job-id', job_id)
        set_output('job-name', job_name)
    except Exception as e:
        print(repr(e), file=sys.stderr)
        print(f'workflow-{args.workflow_run_id}')
if __name__ == '__main__':
    main()