import os
import sys
import requests
USAGE = f'Usage: python {sys.argv[0]} [--help] | version_being_released (e.g., v0.19.1)]'

def get_prerelease_status(version_being_released, token):
    if False:
        return 10
    url = f'https://api.github.com/repos/feast-dev/feast/releases/tags/v{version_being_released}'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.github.v3+json', 'Authorization': f'Bearer {token}'}
    response = requests.request('GET', url, headers=headers)
    response_json = response.json()
    return (bool(response_json['prerelease']), response_json['id'])

def set_prerelease_status(release_id, status, token):
    if False:
        print('Hello World!')
    url = f'https://api.github.com/repos/feast-dev/feast/releases/{release_id}'
    payload = {'prerelease': status}
    headers = {'Content-Type': 'application/json', 'Accept': 'application/vnd.github.v3+json', 'Authorization': f'Bearer {token}'}
    requests.request('PATCH', url, json=payload, headers=headers)

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    args = sys.argv[1:]
    if not args or len(args) != 1:
        raise SystemExit(USAGE)
    version_being_released = args[0].strip()
    print(f'Disabling prerelease status for {version_being_released}')
    token = os.getenv('GITHUB_TOKEN', default=None)
    if token is None:
        raise OSError('GITHUB_TOKEN environmental variable is not set')
    (is_prerelease, release_id) = get_prerelease_status(version_being_released, token)
    if is_prerelease:
        set_prerelease_status(release_id, False, token)
    else:
        print(f'{version_being_released} is not a pre-release, exiting.')
        exit(0)
    (is_prerelease, release_id) = get_prerelease_status(version_being_released, token)
    if is_prerelease:
        import warnings
        warnings.warn(f'Failed to unset prerelease status for {version_being_released} release id {release_id}')
    else:
        print(f'Successfully unset prerelease status for {version_being_released} release id {release_id}')
if __name__ == '__main__':
    main()