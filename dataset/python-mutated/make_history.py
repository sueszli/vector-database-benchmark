from __future__ import annotations as _annotations
import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
import requests

def main():
    if False:
        print('Hello World!')
    root_dir = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', help='print preview of release notes to terminal without saving to HISTORY.md')
    args = parser.parse_args()
    if args.preview:
        new_version = args.preview
    else:
        version_file = root_dir / 'pydantic' / 'version.py'
        new_version = re.search("VERSION = '(.*)'", version_file.read_text()).group(1)
    history_path = root_dir / 'HISTORY.md'
    history_content = history_path.read_text()
    if f'## v{new_version} (' in history_content:
        print(f'WARNING: v{new_version} already in history, stopping')
        sys.exit(1)
    title = f'v{new_version} ({date.today():%Y-%m-%d})'
    notes = get_notes(new_version)
    new_chunk = f'## {title}\n\n[GitHub release](https://github.com/pydantic/pydantic/releases/tag/v{new_version})\n\n{notes}\n\n'
    if args.preview:
        print(new_chunk)
        return
    history = new_chunk + history_content
    history_path.write_text(history)
    print(f'\nSUCCESS: added "{title}" section to {history_path.relative_to(root_dir)}')

def get_notes(new_version: str) -> str:
    if False:
        print('Hello World!')
    last_tag = get_last_tag()
    auth_token = get_gh_auth_token()
    data = {'target_committish': 'main', 'previous_tag_name': last_tag, 'tag_name': f'v{new_version}'}
    response = requests.post('https://api.github.com/repos/pydantic/pydantic/releases/generate-notes', headers={'Accept': 'application/vnd.github+json', 'Authorization': f'Bearer {auth_token}', 'x-github-api-version': '2022-11-28'}, data=json.dumps(data))
    response.raise_for_status()
    body = response.json()['body']
    body = body.replace('<!-- Release notes generated using configuration in .github/release.yml at main -->\n\n', '')
    body = re.sub(pattern='^(#+ .+?)$', repl='#\\1\\n', string=body, flags=re.MULTILINE)
    body = re.sub(pattern='([^\n])(\n#+ .+?\n)', repl='\\1\\n\\2', string=body)
    body = re.sub(pattern='https://github.com/pydantic/pydantic/pull/(\\d+)', repl='[#\\1](https://github.com/pydantic/pydantic/pull/\\1)', string=body)
    body = re.sub(pattern='\\*\\*Full Changelog\\*\\*: https://.*$', repl='', string=body)
    return body.strip()

def get_last_tag():
    if False:
        i = 10
        return i + 15
    return run('git', 'describe', '--tags', '--abbrev=0')

def get_gh_auth_token():
    if False:
        while True:
            i = 10
    return run('gh', 'auth', 'token')

def run(*args: str) -> str:
    if False:
        return 10
    p = subprocess.run(args, stdout=subprocess.PIPE, check=True, encoding='utf-8')
    return p.stdout.strip()
if __name__ == '__main__':
    main()