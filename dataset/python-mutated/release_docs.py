import os
import re
import sys
import base64
import argparse
import requests
VERSION_VALIDATOR = re.compile('^[0-9]+\\.[0-9]+$')

class ReadmeAuth(requests.auth.AuthBase):

    def __call__(self, r):
        if False:
            return 10
        r.headers['authorization'] = f'Basic {readme_token()}'
        return r

def readme_token():
    if False:
        for i in range(10):
            print('nop')
    api_key = os.getenv('RDME_API_KEY', None)
    if not api_key:
        raise Exception('RDME_API_KEY env var is not set')
    api_key = f'{api_key}:'
    return base64.b64encode(api_key.encode('utf-8')).decode('utf-8')

def get_versions():
    if False:
        return 10
    '\n    Return all versions currently published in Readme.io.\n    '
    url = 'https://dash.readme.com/api/v1/version'
    res = requests.get(url, auth=ReadmeAuth(), timeout=30)
    res.raise_for_status()
    return [v['version'] for v in res.json()]

def create_new_unstable(current, new):
    if False:
        return 10
    '\n    Create new version by copying current.\n\n    :param current: Existing current unstable version\n    :param new: Non existing new unstable version\n    '
    url = 'https://dash.readme.com/api/v1/version/'
    payload = {'is_beta': False, 'version': new, 'from': current, 'is_hidden': False, 'is_stable': False}
    res = requests.post(url, json=payload, auth=ReadmeAuth(), timeout=30)
    res.raise_for_status()

def promote_unstable_to_stable(unstable, stable):
    if False:
        for i in range(10):
            print('nop')
    '\n    Rename the current unstable to stable and set it as stable.\n\n    :param unstable: Existing unstable version\n    :param stable: Non existing new stable version\n    '
    url = f'https://dash.readme.com/api/v1/version/{unstable}'
    payload = {'is_beta': False, 'version': stable, 'from': unstable, 'is_hidden': False, 'is_stable': True}
    res = requests.put(url, json=payload, auth=ReadmeAuth(), timeout=30)
    res.raise_for_status()

def calculate_new_unstable(version):
    if False:
        i = 10
        return i + 15
    (major, minor) = version.split('.')
    return f'{major}.{int(minor) + 1}-unstable'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--new-version', help='The new minor version that is being released (e.g. 1.9).', required=True)
    args = parser.parse_args()
    if VERSION_VALIDATOR.match(args.new_version) is None:
        sys.exit('Version must be formatted like so <major>.<minor>')
    new_stable = f'{args.new_version}'
    new_unstable = calculate_new_unstable(args.new_version)
    versions = get_versions()
    new_stable_is_published = new_stable in versions
    new_unstable_is_published = new_unstable in versions
    if new_stable_is_published and new_unstable_is_published:
        print(f'Both new version {new_stable} and {new_unstable} are already published.')
        sys.exit(0)
    elif new_stable_is_published or new_unstable_is_published:
        sys.exit(f'Either version {new_stable} or {new_unstable} are already published. Too risky to proceed.')
    current_unstable = f'{new_stable}-unstable'
    if current_unstable not in versions:
        sys.exit(f"Can't find version {current_unstable} to promote to {new_stable}")
    create_new_unstable(current_unstable, new_unstable)
    promote_unstable_to_stable(current_unstable, new_stable)