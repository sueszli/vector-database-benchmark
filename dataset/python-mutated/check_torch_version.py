"""
Ensures currently installed torch version is the newest allowed.
"""
from typing import Tuple, cast

def main():
    if False:
        print('Hello World!')
    current_torch_version = _get_current_installed_torch_version()
    latest_torch_version = _get_latest_torch_version()
    torch_version_upper_limit = _get_torch_version_upper_limit()
    if current_torch_version < latest_torch_version < torch_version_upper_limit:
        raise RuntimeError(f'current torch version {current_torch_version} is behind latest allowed torch version {latest_torch_version}')
    print('All good!')

def _get_current_installed_torch_version() -> Tuple[str, str, str]:
    if False:
        return 10
    import torch
    version = tuple(torch.version.__version__.split('.'))
    assert len(version) == 3, f"Bad parsed version '{version}'"
    return cast(Tuple[str, str, str], version)

def _get_latest_torch_version() -> Tuple[str, str, str]:
    if False:
        for i in range(10):
            print('nop')
    import requests
    r = requests.get('https://api.github.com/repos/pytorch/pytorch/tags')
    assert r.ok
    for tag_data in r.json():
        tag = tag_data['name']
        if tag.startswith('v') and '-rc' not in tag:
            version = tuple(tag[1:].split('.'))
            assert len(version) == 3, f"Bad parsed version '{version}'"
            break
    else:
        raise RuntimeError('could not find latest stable release tag')
    return cast(Tuple[str, str, str], version)

def _get_torch_version_upper_limit() -> Tuple[str, str, str]:
    if False:
        for i in range(10):
            print('nop')
    with open('constraints.txt') as f:
        for line in f:
            if 'torch<' in line:
                version = tuple(line.split('<')[1].strip().split('.'))
                assert len(version) == 3, f"Bad parsed version '{version}'"
                break
        else:
            raise RuntimeError('could not find torch version spec in constraints.txt')
    return cast(Tuple[str, str, str], version)
if __name__ == '__main__':
    main()