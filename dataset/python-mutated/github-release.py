from __future__ import annotations
import argparse
import logging
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from pprint import pformat
from typing import IO, Any, Literal, NewType, Optional
import jinja2
import requests
log = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1].resolve()
DEFAULT_REPO = 'streamlink/streamlink'
RE_CHANGELOG = re.compile('\n    ##\\sstreamlink\\s+\n    (?P<version>\\d+\\.\\d+\\.\\d+(?:-\\S+)?)\\s+\n    \\((?P<date>\\d{4}-\\d\\d-\\d\\d)\\)\\n\\n\n    (?P<changelog>.+?)\\n\\n\n    \\[Full\\schangelog]\\(\\S+\\)\\n+\n    (?=\\#\\#\\sstreamlink|$)\n', re.VERBOSE | re.DOTALL | re.IGNORECASE)
RE_CO_AUTHOR = re.compile('\n    ^\\s*Co-Authored-By:\\s+(?P<name>.+)\\s+<(?P<email>.+?@.+?)>\\s*$\n', re.VERBOSE | re.MULTILINE | re.IGNORECASE)

def get_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Create or update a GitHub release and upload release assets.\n' + 'Reads the API key from the RELEASES_API_KEY or GITHUB_TOKEN env vars.\n' + 'Performs a dry run if no API key was set.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--repo', metavar='REPOSITORY', default=getenv('GITHUB_REPOSITORY', DEFAULT_REPO), help=f'The repository name\nDefault: env.GITHUB_REPOSITORY or {DEFAULT_REPO}')
    parser.add_argument('--tag', metavar='TAG', help='The tag name\nDefault: latest tag read from current git branch')
    parser.add_argument('--template', metavar='FILE', default=ROOT / '.github' / 'release_template.md', type=Path, help='The release template file\nDefault: $GITROOT/.github/release_template.md')
    parser.add_argument('--changelog', metavar='FILE', default=ROOT / 'CHANGELOG.md', type=Path, help='The changelog file\nDefault: $GITROOT/CHANGELOG.md')
    parser.add_argument('--no-contributors', action='store_true', help="Don't generate contributors list with GitHub usernames")
    parser.add_argument('--no-shortlog', action='store_true', help="Don't generate git shortlog")
    parser.add_argument('assets', nargs='*', type=Path, help='List of asset file paths to be uploaded')
    return parser.parse_args()
Email = NewType('Email', str)

@dataclass
class Author:
    email: Email
    name: str
    commits: int = 0

class Git:

    @staticmethod
    def _output(*gitargs, **runkwargs) -> str:
        if False:
            return 10
        completedprocess = subprocess.run(['git', '--no-pager', *map(str, gitargs)], capture_output=True, check=True, **runkwargs)
        return completedprocess.stdout.decode().rstrip()

    @classmethod
    def tag(cls, ref: str='HEAD') -> str:
        if False:
            i = 10
            return i + 15
        try:
            return cls._output('describe', '--tags', '--first-parent', '--abbrev=0', ref)
        except subprocess.CalledProcessError as err:
            raise ValueError(f'Could not get tag from git:\n{err.stderr}') from err

    @classmethod
    def shortlog(cls, start: str, end: str) -> str:
        if False:
            print('Hello World!')
        try:
            return cls._output('shortlog', '--email', '--no-merges', '--pretty=%s', f'{start}...{end}')
        except subprocess.CalledProcessError as err:
            raise ValueError(f'Could not get shortlog from git:\n{err.stderr}') from err

class GitHubAPI:
    PER_PAGE = 100
    MAX_REQUESTS = 10

    def __init__(self, repo: str, tag: str):
        if False:
            return 10
        self.authenticated = False
        self.repo = repo
        self.tag = tag
        self.primary_headers = {'Accept': 'application/vnd.github.v3+json', 'User-Agent': repo}
        self._get_api_key()

    def _get_api_key(self):
        if False:
            return 10
        (github_token, releases_api_key) = (getenv('GITHUB_TOKEN'), getenv('RELEASES_API_KEY'))
        if github_token:
            self.primary_headers.update(Authorization=f'Bearer {github_token}')
        elif releases_api_key:
            self.primary_headers.update(Authorization=f'token {releases_api_key}')
            res = self.call(endpoint='/user', raise_failure=False)
            if res.status_code >= 400:
                raise ValueError('Invalid API key')
        else:
            log.info('No API key provided. Continuing with dry-run...')
            return
        self.authenticated = True

    def call(self, host: str='api.github.com', method: Literal['GET', 'POST', 'PATCH']='GET', endpoint: str='/', headers: Optional[dict[str, Any]]=None, raise_failure: bool=True, **kwargs) -> requests.Response:
        if False:
            while True:
                i = 10
        func = requests.post if method == 'POST' else requests.patch if method == 'PATCH' else requests.get
        response: requests.Response = func(f'https://{host}{endpoint}', headers={**(headers or {}), **self.primary_headers}, **kwargs)
        if raise_failure and response.status_code >= 400:
            log.debug(f'GitHub API request failed:\n{response.text}')
            raise requests.HTTPError(f'GitHub API request {method} {endpoint} returned {response.status_code}')
        return response

    @staticmethod
    def get_response_json_key(response: requests.Response, key: str) -> Any:
        if False:
            return 10
        data = response.json()
        if key not in data:
            raise KeyError(f"Missing key '{key}' in GitHub API response")
        return data[key]

    def get_id(self, response: requests.Response) -> int:
        if False:
            i = 10
            return i + 15
        return self.get_response_json_key(response, 'id')

    def get_release_id(self) -> Optional[int]:
        if False:
            print('Hello World!')
        log.debug(f'Checking for existing release in {self.repo} tagged by {self.tag}')
        response = self.call(endpoint=f'/repos/{self.repo}/releases/tags/{self.tag}', raise_failure=False)
        return None if response.status_code >= 400 else self.get_id(response)

    def create_release(self, payload: dict) -> Optional[int]:
        if False:
            while True:
                i = 10
        if not self.authenticated:
            log.info(f'dry-run: Would have created GitHub release {self.repo}#{self.tag} with:\n{pformat(payload)}')
            return
        log.info(f'Creating new GitHub release {self.repo}#{self.tag}')
        res = self.call(method='POST', endpoint=f'/repos/{self.repo}/releases', json=payload)
        log.info(f'Successfully created new GitHub release {self.repo}#{self.tag}')
        return self.get_id(res)

    def update_release(self, release_id: int, payload: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self.authenticated:
            log.info(f'dry-run: Would have updated GitHub release {self.repo}#{self.tag} with:\n{pformat(payload)}')
            return
        log.info(f'Updating existing GitHub release {self.repo}#{self.tag}')
        self.call(method='PATCH', endpoint=f'/repos/{self.repo}/releases/{release_id}', json=payload)
        log.info(f'Successfully updated existing GitHub release {self.repo}#{self.tag}')

    def create_or_update_release(self, **payload) -> int:
        if False:
            print('Hello World!')
        payload.update(tag_name=self.tag)
        release_id = self.get_release_id()
        if not release_id:
            return self.create_release(payload)
        self.update_release(release_id, payload)
        return release_id

    def upload_asset(self, release_id: int, filename: str, filehandle: IO):
        if False:
            for i in range(10):
                print('nop')
        if not self.authenticated:
            log.info(f"dry-run: Would have uploaded '{filename}' to GitHub release {self.repo}#{self.tag}")
            return
        log.info(f"Uploading '{filename}' to GitHub release {self.repo}#{self.tag}")
        self.call(host='uploads.github.com', method='POST', endpoint=f'/repos/{self.repo}/releases/{release_id}/assets', headers={'Content-Type': 'application/octet-stream'}, params={'name': filename}, data=filehandle)
        log.info(f"Successfully uploaded '{filename}' to GitHub release {self.repo}#{self.tag}")

    def get_contributors(self, start: str, end: str) -> list[Author]:
        if False:
            i = 10
            return i + 15
        log.debug(f'Getting contributors of {self.repo} in commit range {start}...{end}')
        authors: dict[Email, Author] = {}
        co_authors: list[Email] = []
        total_commits = None
        parsed_commits = 0
        page = 0
        while total_commits is None or parsed_commits < total_commits:
            page += 1
            res = self.call(endpoint=f'/repos/{self.repo}/compare/{start}...{end}', params={'page': page, 'per_page': self.PER_PAGE})
            if res.status_code != 200:
                raise requests.HTTPError(f'Status code {res.status_code} for request {res.url}')
            data: dict = res.json()
            if total_commits is None:
                total_commits = data.get('total_commits')
                if total_commits is None:
                    raise ValueError('Could not get total_commits value')
                if total_commits > self.MAX_REQUESTS * self.PER_PAGE:
                    raise ValueError('Too many commits in input range')
            commits: list[dict] = data.get('commits', [])
            parsed_commits += len(commits)
            for commitdata in commits:
                commit = commitdata.get('commit') or {}
                if len(commitdata.get('parents') or []) > 1:
                    continue
                commit_author_email = Email((commit.get('author') or {}).get('email'))
                author_name = (commitdata.get('author') or {}).get('login')
                if not commit_author_email or not author_name:
                    continue
                if commit_author_email not in authors:
                    authors[commit_author_email] = Author(commit_author_email, author_name)
                authors[commit_author_email].commits += 1
                for item in re.finditer(RE_CO_AUTHOR, commit.get('message', '')):
                    co_author_email = Email(item.group('email'))
                    if co_author_email == commit_author_email:
                        continue
                    co_authors.append(co_author_email)
        for email in co_authors:
            if email in authors:
                authors[email].commits += 1
        return sorted(sorted(authors.values(), key=lambda author: author.name, reverse=False), key=lambda author: author.commits, reverse=True)

class Release:

    def __init__(self, tag: str, template: Path, changelog: Path):
        if False:
            print('Hello World!')
        self.tag = tag
        self.template = template
        self.changelog = changelog

    @staticmethod
    def _read_file(path: Path):
        if False:
            while True:
                i = 10
        with open(path, 'r', encoding='utf-8') as fh:
            contents = fh.read()
        if not contents:
            raise IOError()
        return contents

    def _read_template(self):
        if False:
            return 10
        log.debug(f'Opening release template file: {self.template}')
        try:
            return self._read_file(self.template)
        except OSError as err:
            raise OSError('Missing release template file') from err

    def _read_changelog(self):
        if False:
            while True:
                i = 10
        log.debug(f'Opening changelog file: {self.changelog}')
        try:
            return self._read_file(self.changelog)
        except OSError as err:
            raise OSError('Missing changelog file') from err

    def _get_changelog(self) -> dict:
        if False:
            i = 10
            return i + 15
        changelog = self._read_changelog()
        log.debug('Parsing changelog file')
        for match in re.finditer(RE_CHANGELOG, changelog):
            if match.group('version') == self.tag:
                return match.groupdict()
        raise KeyError('Missing changelog for current release')

    @staticmethod
    @contextmanager
    def get_file_handles(assets: list[Path]) -> dict[str, IO]:
        if False:
            i = 10
            return i + 15
        handles = {}
        try:
            for asset in assets:
                asset = ROOT / asset
                if not asset.is_file():
                    continue
                log.info(f"Found release asset '{asset.name}'")
                handles[asset.name] = open(asset, 'rb')
            yield handles
        finally:
            for handle in handles.values():
                handle.close()

    def get_body(self, api: GitHubAPI, no_contributors: bool=False, no_shortlog: bool=False) -> str:
        if False:
            return 10
        template = self._read_template()
        jinjatemplate = jinja2.Template(template)
        changelog = self._get_changelog()
        context = dict(**changelog)
        if not no_contributors or not no_shortlog:
            prev_commit = f'{self.tag}~1'
            start = Git.tag(prev_commit)
            if not start:
                raise ValueError(f'Could not resolve tag from reference {prev_commit}')
            if not no_contributors:
                context.update(contributors=api.get_contributors(start, prev_commit))
            if not no_shortlog:
                context.update(gitshortlog=Git.shortlog(start, prev_commit))
        return jinjatemplate.render(context)

def main(args: argparse.Namespace):
    if False:
        print('Hello World!')
    tag = args.tag or Git.tag()
    if not tag:
        raise ValueError('Missing git tag')
    log.info(f'Repo: {args.repo}')
    log.info(f'Tag: {tag}')
    release = Release(tag, args.template, args.changelog)
    with release.get_file_handles(args.assets) as filehandles:
        api = GitHubAPI(args.repo, tag)
        body = release.get_body(api, args.no_contributors, args.no_shortlog)
        release_id = api.create_or_update_release(name=f'Streamlink {tag}', body=body)
        for (filename, filehandle) in filehandles.items():
            api.upload_asset(release_id, filename, filehandle)
    log.info('Done')
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(levelname)s] %(message)s')
    try:
        main(args)
    except KeyboardInterrupt:
        sys.exit(130)