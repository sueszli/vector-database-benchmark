import json
import os
from pathlib import Path
from github import Github
from github.NamedUser import NamedUser
from jinja2 import Template
CURRENT_FILE = Path(__file__)
ROOT = CURRENT_FILE.parents[1]
BOT_LOGINS = ['pyup-bot']
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', None)
GITHUB_REPO = os.getenv('GITHUB_REPOSITORY', None)

def main() -> None:
    if False:
        while True:
            i = 10
    '\n    Script entry point.\n\n    1. Fetch recent contributors from the Github API\n    2. Add missing ones to the JSON file\n    3. Generate Markdown from JSON file\n    '
    recent_authors = set(iter_recent_authors())
    contrib_file = ContributorsJSONFile()
    for author in recent_authors:
        print(f'Checking if {author.login} should be added')
        if author.login not in contrib_file:
            contrib_file.add_contributor(author)
            print(f'Added {author.login} to contributors')
    contrib_file.save()
    write_md_file(contrib_file.content)

def iter_recent_authors():
    if False:
        while True:
            i = 10
    '\n    Fetch users who opened recently merged pull requests.\n\n    Use Github API to fetch recent authors rather than\n    git CLI to work with Github usernames.\n    '
    repo = Github(login_or_token=GITHUB_TOKEN, per_page=5).get_repo(GITHUB_REPO)
    recent_pulls = repo.get_pulls(state='closed', sort='updated', direction='desc').get_page(0)
    for pull in recent_pulls:
        if pull.merged and pull.user.type == 'User' and (pull.user.login not in BOT_LOGINS):
            yield pull.user

class ContributorsJSONFile:
    """Helper to interact with the JSON file."""
    file_path = ROOT / '.github' / 'contributors.json'
    content = None

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        'Read initial content.'
        self.content = json.loads(self.file_path.read_text())

    def __contains__(self, github_login: str):
        if False:
            print('Hello World!')
        'Provide a nice API to do: `username in file`.'
        return any((github_login.lower() == contrib['github_login'].lower() for contrib in self.content))

    def add_contributor(self, user: NamedUser):
        if False:
            return 10
        'Append the contributor data we care about at the end.'
        contributor_data = {'name': user.name or user.login, 'github_login': user.login, 'twitter_username': user.twitter_username or ''}
        self.content.append(contributor_data)

    def save(self):
        if False:
            print('Hello World!')
        'Write the file to disk with indentation.'
        text_content = json.dumps(self.content, indent=2, ensure_ascii=False)
        self.file_path.write_text(text_content)

def write_md_file(contributors):
    if False:
        while True:
            i = 10
    'Generate markdown file from Jinja template.'
    contributors_template = ROOT / '.github' / 'CONTRIBUTORS-template.md'
    template = Template(contributors_template.read_text(), autoescape=True)
    core_contributors = [c for c in contributors if c.get('is_core', False)]
    other_contributors = (c for c in contributors if not c.get('is_core', False))
    other_contributors = sorted(other_contributors, key=lambda c: c['name'].lower())
    content = template.render(core_contributors=core_contributors, other_contributors=other_contributors)
    file_path = ROOT / 'CONTRIBUTORS.md'
    file_path.write_text(content)
if __name__ == '__main__':
    if GITHUB_REPO is None:
        raise RuntimeError('No github repo, please set the environment variable GITHUB_REPOSITORY')
    main()