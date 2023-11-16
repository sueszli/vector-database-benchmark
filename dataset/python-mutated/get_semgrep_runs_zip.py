import subprocess
from typing import Any
from typing import Mapping
import click
import requests
API_ROOT = 'https://api.github.com/repos/returntocorp/semgrep-rules'
ACCEPT = 'application/vnd.github.v3+json'
HEADERS = {'accept': ACCEPT}
TIMEOUT = 30
ZIP_LOC = '/tmp/semgrep_runs_output.zip'
FILE_LOC = '/tmp/semgrep_runs_output.tar.gz'
JsonObject = Mapping[str, Any]

def err(content: str, **kwargs: Any) -> None:
    if False:
        while True:
            i = 10
    click.secho(content, err=True, **kwargs)

def _gh_get(path: str) -> JsonObject:
    if False:
        print('Hello World!')
    r = requests.get(f'{API_ROOT}/{path}', headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _get_last_rules_commit() -> str:
    if False:
        print('Hello World!')
    res = _gh_get('branches/develop')
    return res['commit']['sha']

def _get_action_run() -> JsonObject:
    if False:
        print('Hello World!')
    res = _gh_get(f'actions/runs?branch=develop&event=push')
    return next((r for r in res['workflow_runs'] if r['name'] == 'semgrep-rules-test'))

def _get_artifact_url(run_id: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    res = _gh_get(f'actions/runs/{run_id}/artifacts')
    return next((a for a in res['artifacts'] if a['name'] == 'semgrep_runs'))['archive_download_url']

def _get_runs_artifact(url: str, access_token: str) -> None:
    if False:
        while True:
            i = 10
    res = requests.get(url, headers={'Authorization': f'bearer {access_token}'}, timeout=30)
    res.raise_for_status()
    with open(ZIP_LOC, 'wb') as fd:
        fd.write(res.content)

def _unzip_artifact() -> None:
    if False:
        print('Hello World!')
    subprocess.run(['unzip', ZIP_LOC], cwd='/tmp')

@click.command()
@click.argument('access_token')
def main(access_token: str):
    if False:
        i = 10
        return i + 15
    run_id: int = _get_action_run()['id']
    url: str = _get_artifact_url(run_id)
    err(f'Downloading {url}')
    _get_runs_artifact(url, access_token)
    err(f'Download successful')
    err(f'Unzipping archive')
    _unzip_artifact()
    err(f'Done; artifact is at {FILE_LOC}')
if __name__ == '__main__':
    main()