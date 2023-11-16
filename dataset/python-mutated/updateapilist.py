"""This script is used to synthesize generated parts of this library."""
import os
import requests
from typing import List, Optional

class MissingGithubToken(ValueError):
    """Raised when the GITHUB_TOKEN environment variable is not set"""
    pass

class CloudClient:
    repo: str = None
    title: str = None
    release_level: str = None
    distribution_name: str = None

    def __init__(self, repo: dict):
        if False:
            print('Hello World!')
        self.repo = repo['repo']
        self.title = repo['name_pretty'].replace('Google ', '').replace('Cloud ', '')
        self.release_level = repo['release_level']
        self.distribution_name = repo['distribution_name']

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if self.release_level == other.release_level:
            return self.title < other.title
        return other.release_level < self.release_level

    def __repr__(self):
        if False:
            print('Hello World!')
        return repr((self.release_level, self.title))

def replace_content_in_readme(content_rows: List[str]) -> None:
    if False:
        return 10
    START_MARKER = '.. API_TABLE_START'
    END_MARKER = '.. API_TABLE_END'
    newlines = []
    repl_open = False
    with open('README.rst', 'r') as f:
        for line in f:
            if not repl_open:
                newlines.append(line)
            if line.startswith(START_MARKER):
                repl_open = True
                newlines = newlines + content_rows
            elif line.startswith(END_MARKER):
                newlines.append('\n')
                newlines.append(line)
                repl_open = False
    with open('README.rst', 'w') as f:
        for line in newlines:
            f.write(line)

def client_row(client: CloudClient) -> str:
    if False:
        return 10
    pypi_badge = f'.. |PyPI-{client.distribution_name}| image:: https://img.shields.io/pypi/v/{client.distribution_name}.svg\n     :target: https://pypi.org/project/{client.distribution_name}\n'
    content_row = [f'   * - `{client.title} <https://github.com/{client.repo}>`_\n', f'     - ' + '|' + client.release_level + f'|\n     - |PyPI-{client.distribution_name}|\n']
    return (content_row, pypi_badge)

def generate_table_contents(clients: List[CloudClient]) -> List[str]:
    if False:
        i = 10
        return i + 15
    content_rows = ['\n', '.. list-table::\n', '   :header-rows: 1\n', '\n', '   * - Client\n', '     - Release Level\n', '     - Version\n']
    pypi_links = ['\n']
    for client in clients:
        (content_row, pypi_link) = client_row(client)
        content_rows += content_row
        pypi_links.append(pypi_link)
    return content_rows + pypi_links
REPO_METADATA_URL_FORMAT = 'https://raw.githubusercontent.com/{repo_slug}/main/.repo-metadata.json'

def client_for_repo(repo_slug) -> Optional[CloudClient]:
    if False:
        for i in range(10):
            print('nop')
    url = REPO_METADATA_URL_FORMAT.format(repo_slug=repo_slug)
    response = requests.get(url)
    if response.status_code != requests.codes.ok:
        return
    return CloudClient(response.json())
REPO_EXCLUSION = ['googleapis/python-api-core', 'googleapis/python-cloud-core', 'googleapis/python-org-policy', 'googleapis/python-os-config', 'googleapis/python-access-context-manager', 'googleapis/python-api-common-protos', 'googleapis/python-test-utils']

def allowed_repo(repo) -> bool:
    if False:
        while True:
            i = 10
    return repo['full_name'].startswith('googleapis/python-') and repo['full_name'] not in REPO_EXCLUSION and (not repo['archived'])

def get_clients_batch_from_response_json(response_json) -> List[CloudClient]:
    if False:
        while True:
            i = 10
    return [client_for_repo(repo['full_name']) for repo in response_json if allowed_repo(repo)]

def all_clients() -> List[CloudClient]:
    if False:
        return 10
    clients = []
    first_request = True
    token = os.environ['GITHUB_TOKEN']
    while first_request or 'next' in response.links:
        if first_request:
            url = 'https://api.github.com/search/repositories?page=1'
            first_request = False
        else:
            url = response.links['next']['url']
        headers = {'Authorization': f'token {token}'}
        params = {'per_page': 100, 'q': 'python- in:name org:googleapis'}
        response = requests.get(url=url, params=params, headers=headers)
        repositories = response.json().get('items', [])
        if len(repositories) == 0:
            break
        clients.extend(get_clients_batch_from_response_json(repositories))
    return [client for client in clients if client]
if 'GITHUB_TOKEN' not in os.environ:
    raise MissingGithubToken('Please include a GITHUB_TOKEN env var.')
clients = sorted(all_clients())
table_contents = generate_table_contents(clients)
replace_content_in_readme(table_contents)