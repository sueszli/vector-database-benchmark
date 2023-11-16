"""Internal tool to update the changelog."""
import json
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from rich.prompt import Prompt

@dataclass(frozen=True)
class Change:
    """Capture the data of a git commit."""
    commit_hash: str
    prefix: str
    message: str
    author: str
    author_login: str

def main(changelog_path: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Create a changelog.\n\n    Args:\n        changelog_path: The location of the CHANGELOG file\n    '
    changelog = get_changelog(changelog_path)
    git_tag = get_most_recent_git_tag()
    (changes, changes_with_author) = get_formatted_changes(git_tag)
    if changes == '':
        print('No changes')
        return
    new_version = version_bump(git_tag)
    new_version = get_version_interactive(new_version, changes)
    adjust_version_py(new_version)
    today = datetime.now(tz=timezone.utc)
    header = f'## Version {new_version}, {today:%Y-%m-%d}\n'
    url = f'https://github.com/py-pdf/pypdf/compare/{git_tag}...{new_version}'
    trailer = f'\n[Full Changelog]({url})\n\n'
    new_entry = header + changes + trailer
    print(new_entry)
    write_commit_msg_file(new_version, changes_with_author + trailer)
    write_release_msg_file(new_version, changes_with_author + trailer, today)
    if new_entry in changelog:
        print('Changelog is already up-to-date!')
        return
    new_changelog = '# CHANGELOG\n\n' + new_entry + strip_header(changelog)
    write_changelog(new_changelog, changelog_path)
    print_instructions(new_version)

def print_instructions(new_version: str) -> None:
    if False:
        return 10
    'Print release instructions.'
    print('=' * 80)
    print(f"☑  _version.py was adjusted to '{new_version}'")
    print('☑  CHANGELOG.md was adjusted')
    print('')
    print('Now run:')
    print('  git commit -eF RELEASE_COMMIT_MSG.md')
    print(f'  git tag -s {new_version} -eF RELEASE_TAG_MSG.md')
    print('  git push')
    print('  git push --tags')

def adjust_version_py(version: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Adjust the __version__ string.'
    with open('pypdf/_version.py', 'w') as fp:
        fp.write(f'__version__ = "{version}"\n')

def get_version_interactive(new_version: str, changes: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the new __version__ interactively.'
    print('The changes are:')
    print(changes)
    orig = new_version
    new_version = Prompt.ask('New semantic version', default=orig)
    while not is_semantic_version(new_version):
        new_version = Prompt.ask('That was not a semantic version. Please enter a semantic version', default=orig)
    return new_version

def is_semantic_version(version: str) -> bool:
    if False:
        return 10
    'Check if the given version is a semantic version.'
    if version.count('.') != 2:
        return False
    try:
        return bool([int(part) for part in version.split('.')])
    except Exception:
        return False

def write_commit_msg_file(new_version: str, commit_changes: str) -> None:
    if False:
        return 10
    '\n    Write a file that can be used as a commit message.\n\n    Like this:\n\n        git commit -eF RELEASE_COMMIT_MSG.md && git push\n    '
    with open('RELEASE_COMMIT_MSG.md', 'w') as fp:
        fp.write(f'REL: {new_version}\n\n')
        fp.write("## What's new\n")
        fp.write(commit_changes)

def write_release_msg_file(new_version: str, commit_changes: str, today: datetime) -> None:
    if False:
        return 10
    '\n    Write a file that can be used as a git tag message.\n\n    Like this:\n\n        git tag -eF RELEASE_TAG_MSG.md && git push\n    '
    with open('RELEASE_TAG_MSG.md', 'w') as fp:
        fp.write(f'Version {new_version}, {today:%Y-%m-%d}\n\n')
        fp.write("## What's new\n")
        fp.write(commit_changes)

def strip_header(md: str) -> str:
    if False:
        print('Hello World!')
    "Remove the 'CHANGELOG' header."
    return md.lstrip('# CHANGELOG').lstrip()

def version_bump(git_tag: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Increase the patch version of the git tag by one.\n\n    Args:\n        git_tag: Old version tag\n\n    Returns:\n        The new version where the patch version is bumped.\n    '
    (major, minor, patch) = git_tag.split('.')
    return f'{major}.{minor}.{int(patch) + 1}'

def get_changelog(changelog_path: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Read the changelog.\n\n    Args:\n        changelog_path: Path to the CHANGELOG file\n\n    Returns:\n        Data of the CHANGELOG\n    '
    with open(changelog_path) as fh:
        changelog = fh.read()
    return changelog

def write_changelog(new_changelog: str, changelog_path: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Write the changelog.\n\n    Args:\n        new_changelog: Contents of the new CHANGELOG\n        changelog_path: Path where the CHANGELOG file is\n    '
    with open(changelog_path, 'w') as fh:
        fh.write(new_changelog)

def get_formatted_changes(git_tag: str) -> Tuple[str, str]:
    if False:
        print('Hello World!')
    '\n    Format the changes done since the last tag.\n\n    Args:\n        git_tag: the reference tag\n\n    Returns:\n        Changes done since git_tag\n    '
    commits = get_git_commits_since_tag(git_tag)
    grouped = {}
    for commit in commits:
        if commit.prefix not in grouped:
            grouped[commit.prefix] = []
        grouped[commit.prefix].append({'msg': commit.message, 'author': commit.author_login})
    order = ['SEC', 'DEP', 'ENH', 'PI', 'BUG', 'ROB', 'DOC', 'DEV', 'CI', 'MAINT', 'TST', 'STY']
    abbrev2long = {'SEC': 'Security', 'DEP': 'Deprecations', 'ENH': 'New Features', 'BUG': 'Bug Fixes', 'ROB': 'Robustness', 'DOC': 'Documentation', 'DEV': 'Developer Experience', 'CI': 'Continuous Integration', 'MAINT': 'Maintenance', 'TST': 'Testing', 'STY': 'Code Style', 'PI': 'Performance Improvements'}
    output = ''
    output_with_user = ''
    for prefix in order:
        if prefix not in grouped:
            continue
        tmp = f'\n### {abbrev2long[prefix]} ({prefix})\n'
        output += tmp
        output_with_user += tmp
        for commit in grouped[prefix]:
            output += f"- {commit['msg']}\n"
            output_with_user += f"- {commit['msg']} by @{commit['author']}\n"
        del grouped[prefix]
    if grouped:
        output += '\n### Other\n'
        for prefix in grouped:
            output += f'- {prefix}: {grouped[prefix]}\n'
    return (output, output_with_user)

def get_most_recent_git_tag() -> str:
    if False:
        i = 10
        return i + 15
    '\n    Get the git tag most recently created.\n\n    Returns:\n        Most recently created git tag.\n    '
    git_tag = str(subprocess.check_output(['git', 'describe', '--abbrev=0'], stderr=subprocess.STDOUT)).strip("'b\\n")
    return git_tag

def get_author_mapping(line_count: int) -> Dict[str, str]:
    if False:
        i = 10
        return i + 15
    '\n    Get the authors for each commit.\n\n    Args:\n        line_count: Number of lines from Git log output. Used for determining how\n            many commits to fetch.\n\n    Returns:\n        A mapping of long commit hashes to author login handles.\n    '
    per_page = min(line_count, 100)
    page = 1
    mapping: Dict[str, str] = {}
    for _ in range(0, line_count, per_page):
        with urllib.request.urlopen(f'https://api.github.com/repos/py-pdf/pypdf/commits?per_page={per_page}&page={page}') as response:
            commits = json.loads(response.read())
        page += 1
        for commit in commits:
            mapping[commit['sha']] = commit['author']['login']
    return mapping

def get_git_commits_since_tag(git_tag: str) -> List[Change]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get all commits since the last tag.\n\n    Args:\n        git_tag: Reference tag from which the changes to the current commit are\n            fetched.\n\n    Returns:\n        List of all changes since git_tag.\n    '
    commits = str(subprocess.check_output(['git', '--no-pager', 'log', f'{git_tag}..HEAD', '--pretty=format:"%H:::%s:::%aN"'], stderr=subprocess.STDOUT)).strip("'b\\n")
    lines = commits.split('\\n')
    authors = get_author_mapping(len(lines))
    return [parse_commit_line(line, authors) for line in lines if line != '']

def parse_commit_line(line: str, authors: Dict[str, str]) -> Change:
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse the first line of a git commit message.\n\n    Args:\n        line: The first line of a git commit message.\n\n    Returns:\n        The parsed Change object\n\n    Raises:\n        ValueError: The commit line is not well-structured\n    '
    parts = line.split(':::')
    if len(parts) != 3:
        raise ValueError(f"Invalid commit line: '{line}'")
    (commit_hash, rest, author) = parts
    if ':' in rest:
        (prefix, message) = rest.split(':', 1)
    else:
        prefix = ''
        message = rest
    message.strip()
    commit_hash = commit_hash.strip('"')
    if author.endswith('"'):
        author = author[:-1]
    author_login = authors[commit_hash]
    prefix = prefix.strip()
    if prefix == 'DOCS':
        prefix = 'DOC'
    return Change(commit_hash=commit_hash, prefix=prefix, message=message, author=author, author_login=author_login)
if __name__ == '__main__':
    main('CHANGELOG.md')