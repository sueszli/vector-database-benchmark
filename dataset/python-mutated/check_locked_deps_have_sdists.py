import sys
from pathlib import Path
from typing import Dict, List
import tomli

def main() -> None:
    if False:
        i = 10
        return i + 15
    lockfile_path = Path(__file__).parent.parent.joinpath('poetry.lock')
    with open(lockfile_path, 'rb') as lockfile:
        lockfile_content = tomli.load(lockfile)
    packages_to_assets: Dict[str, List[Dict[str, str]]] = {package['name']: package['files'] for package in lockfile_content['package']}
    success = True
    for (package_name, assets) in packages_to_assets.items():
        has_sdist = any((asset['file'].endswith('.tar.gz') for asset in assets))
        if not has_sdist:
            success = False
            print(f'Locked package {package_name!r} does not have a source distribution!', file=sys.stderr)
    if not success:
        print('\nThere were some problems with the Poetry lockfile (poetry.lock).', file=sys.stderr)
        sys.exit(1)
    print(f'Poetry lockfile OK. {len(packages_to_assets)} locked packages checked.', file=sys.stderr)
if __name__ == '__main__':
    main()