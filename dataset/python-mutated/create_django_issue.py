"""
Creates an issue that generates a table for dependency checking whether
all packages support the latest Django version. "Latest" does not include
patches, only comparing major and minor version numbers.

This script handles when there are multiple Django versions that need
to keep up to date.
"""
from __future__ import annotations
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
import requests
from github import Github
if TYPE_CHECKING:
    from github.Issue import Issue
CURRENT_FILE = Path(__file__)
ROOT = CURRENT_FILE.parents[1]
REQUIREMENTS_DIR = ROOT / '{{cookiecutter.project_slug}}' / 'requirements'
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', None)
GITHUB_REPO = os.getenv('GITHUB_REPOSITORY', None)

class DjVersion(NamedTuple):
    """
    Wrapper to parse, compare and render Django versions.

    Only keeps track on (major, minor) versions, excluding patches and pre-releases.
    """
    major: int
    minor: int

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        'To render as string.'
        return f'{self.major}.{self.minor}'

    @classmethod
    def parse(cls, version_str: str) -> DjVersion:
        if False:
            i = 10
            return i + 15
        'Parse interesting values from the version string.'
        (major, minor, *_) = version_str.split('.')
        return cls(major=int(major), minor=int(minor))

    @classmethod
    def parse_to_tuple(cls, version_str: str):
        if False:
            for i in range(10):
                print('nop')
        version = cls.parse(version_str=version_str)
        return (version.major, version.minor)

def get_package_info(package: str) -> dict:
    if False:
        return 10
    'Get package metadata using PyPI API.'
    r = requests.get(f'https://pypi.org/pypi/{package}/json', allow_redirects=True)
    if not r.ok:
        print(f"Couldn't find package: {package}")
        sys.exit(1)
    return r.json()

def get_django_versions() -> Iterable[DjVersion]:
    if False:
        for i in range(10):
            print('nop')
    'List all django versions.'
    django_package_info: dict[str, Any] = get_package_info('django')
    releases = django_package_info['releases'].keys()
    for release_str in releases:
        if release_str.replace('.', '').isdigit():
            yield DjVersion.parse(release_str)

def get_name_and_version(requirements_line: str) -> tuple[str, ...]:
    if False:
        for i in range(10):
            print('nop')
    'Get the name a version of a package from a line in the requirement file.'
    (full_name, version) = requirements_line.split(' ', 1)[0].split('==')
    name_without_extras = full_name.split('[', 1)[0]
    return (name_without_extras, version)

def get_all_latest_django_versions(django_max_version: tuple[DjVersion]=None) -> tuple[DjVersion, list[DjVersion]]:
    if False:
        i = 10
        return i + 15
    '\n    Grabs all Django versions that are worthy of a GitHub issue.\n    Depends on Django versions having higher major version or minor version.\n    '
    _django_max_version = (99, 99)
    if django_max_version:
        _django_max_version = django_max_version
    print('Fetching all Django versions from PyPI')
    base_txt = REQUIREMENTS_DIR / 'base.txt'
    with base_txt.open() as f:
        for line in f.readlines():
            if 'django==' in line.lower():
                break
        else:
            print(f'django not found in {base_txt}')
            sys.exit(1)
    (_, current_version_str) = get_name_and_version(line)
    current_minor_version = DjVersion.parse(current_version_str)
    newer_versions: set[DjVersion] = set()
    for django_version in get_django_versions():
        if current_minor_version < django_version <= _django_max_version:
            newer_versions.add(django_version)
    return (current_minor_version, sorted(newer_versions, reverse=True))
_TABLE_HEADER = '\n\n## {file}.txt\n\n| Name | Version in Master | {dj_version} Compatible Version | OK |\n| ---- | :---------------: | :-----------------------------: | :-: |\n'
VITAL_BUT_UNKNOWN = ['django-environ']

class GitHubManager:

    def __init__(self, base_dj_version: DjVersion, needed_dj_versions: list[DjVersion]):
        if False:
            for i in range(10):
                print('nop')
        self.github = Github(GITHUB_TOKEN)
        self.repo = self.github.get_repo(GITHUB_REPO)
        self.base_dj_version = base_dj_version
        self.needed_dj_versions = needed_dj_versions
        self.existing_issues: dict[DjVersion, Issue] = {}
        self.requirements_files = ['base', 'local', 'production']
        self.requirements: dict[str, dict[str, tuple[str, dict]]] = {x: {} for x in self.requirements_files}

    def setup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.load_requirements()
        self.load_existing_issues()

    def load_requirements(self):
        if False:
            while True:
                i = 10
        print('Reading requirements')
        for requirements_file in self.requirements_files:
            with (REQUIREMENTS_DIR / f'{requirements_file}.txt').open() as f:
                for line in f.readlines():
                    if '==' in line and (not line.startswith('{%')) and (not line.startswith('    #')) and (not line.startswith('#')) and (not line.startswith(' ')):
                        (name, version) = get_name_and_version(line)
                        self.requirements[requirements_file][name] = (version, get_package_info(name))

    def load_existing_issues(self):
        if False:
            print('Hello World!')
        'Closes the issue if the base Django version is greater than needed'
        print('Load existing issues from GitHub')
        qualifiers = {'repo': GITHUB_REPO, 'author': 'app/github-actions', 'state': 'open', 'is': 'issue', 'in': 'title'}
        issues = list(self.github.search_issues('[Django Update]', 'created', 'desc', **qualifiers))
        print(f'Found {len(issues)} issues matching search')
        for issue in issues:
            matches = re.match('\\[Update Django] Django (\\d+.\\d+)$', issue.title)
            if not matches:
                continue
            issue_version = DjVersion.parse(matches.group(1))
            if self.base_dj_version >= issue_version:
                self.close_issue(issue)
            else:
                self.existing_issues[issue_version] = issue

    def get_compatibility(self, package_name: str, package_info: dict, needed_dj_version: DjVersion):
        if False:
            return 10
        "\n        Verify compatibility via setup.py classifiers. If Django is not in the\n        classifiers, then default compatibility is n/a and OK is ✅.\n\n        If it's a package that's vital but known to not be updated often, we give it\n        a ❓. If a package has ❓ or 🕒, then we allow manual update. Automatic updates\n         only include ❌ and ✅.\n        "
        if (issue := self.existing_issues.get(needed_dj_version)):
            if (index := issue.body.find(package_name)):
                (name, _current, prev_compat, ok) = (s.strip() for s in issue.body[index:].split('|', 4)[:4])
                if ok in ('✅', '❓', '🕒'):
                    return (prev_compat, ok)
        if package_name in VITAL_BUT_UNKNOWN:
            return ('', '❓')
        supported_dj_versions: list[DjVersion] = []
        for classifier in package_info['info']['classifiers']:
            tokens = classifier.split(' ')
            if len(tokens) >= 5 and tokens[2].lower() == 'django':
                version = DjVersion.parse(tokens[4])
                if len(version) == 2:
                    supported_dj_versions.append(version)
        if supported_dj_versions:
            if any((v >= needed_dj_version for v in supported_dj_versions)):
                return (package_info['info']['version'], '✅')
            else:
                return ('', '❌')
        return ('n/a', '✅')
    HOME_PAGE_URL_KEYS = ['home_page', 'project_url', 'docs_url', 'package_url', 'release_url', 'bugtrack_url']

    def _get_md_home_page_url(self, package_info: dict):
        if False:
            print('Hello World!')
        urls = [package_info['info'].get(url_key) for url_key in self.HOME_PAGE_URL_KEYS]
        try:
            return f'[{{}}]({next((item for item in urls if item))})'
        except StopIteration:
            return '{}'

    def generate_markdown(self, needed_dj_version: DjVersion):
        if False:
            i = 10
            return i + 15
        requirements = f'{needed_dj_version} requirements tables\n\n'
        for _file in self.requirements_files:
            requirements += _TABLE_HEADER.format_map({'file': _file, 'dj_version': needed_dj_version})
            for (package_name, (version, info)) in self.requirements[_file].items():
                (compat_version, icon) = self.get_compatibility(package_name, info, needed_dj_version)
                requirements += f'| {self._get_md_home_page_url(info).format(package_name)} | {version.strip()} | {compat_version.strip()} | {icon} |\n'
        return requirements

    def create_or_edit_issue(self, needed_dj_version: DjVersion, description: str):
        if False:
            while True:
                i = 10
        if (issue := self.existing_issues.get(needed_dj_version)):
            print(f'Editing issue #{issue.number} for Django {needed_dj_version}')
            issue.edit(body=description)
        else:
            print(f'Creating new issue for Django {needed_dj_version}')
            issue = self.repo.create_issue(f'[Update Django] Django {needed_dj_version}', description)
            issue.add_to_labels(f'django{needed_dj_version}')

    @staticmethod
    def close_issue(issue: Issue):
        if False:
            for i in range(10):
                print('nop')
        issue.edit(state='closed')
        print(f'Closed issue {issue.title} (ID: [{issue.id}]({issue.url}))')

    def generate(self):
        if False:
            while True:
                i = 10
        for version in self.needed_dj_versions:
            print(f'Handling GitHub issue for Django {version}')
            md_content = self.generate_markdown(version)
            print(f'Generated markdown:\n\n{md_content}')
            self.create_or_edit_issue(version, md_content)

def main(django_max_version=None) -> None:
    if False:
        while True:
            i = 10
    (current_dj, latest_djs) = get_all_latest_django_versions(django_max_version=django_max_version)
    manager = GitHubManager(current_dj, latest_djs)
    manager.setup()
    if not latest_djs:
        print('No new Django versions to update. Exiting...')
        sys.exit(0)
    manager.generate()
if __name__ == '__main__':
    if GITHUB_REPO is None:
        raise RuntimeError('No github repo, please set the environment variable GITHUB_REPOSITORY')
    max_version = None
    last_arg = sys.argv[-1]
    if CURRENT_FILE.name not in last_arg:
        max_version = DjVersion.parse_to_tuple(version_str=last_arg)
    main(django_max_version=max_version)