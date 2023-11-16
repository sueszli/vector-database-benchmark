"""Audit the licenses of all our frontend dependencies (as defined by our
`yarn.lock` file). If any dependency has an unacceptable license, print it
out and exit with an error code. If all dependencies have acceptable licenses,
exit normally.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn, Set, Tuple, cast
from typing_extensions import TypeAlias
PackageInfo: TypeAlias = Tuple[str, str, str, str, str, str]
SCRIPT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR_LIB = SCRIPT_DIR.parent / 'frontend/lib'
FRONTEND_DIR_APP = SCRIPT_DIR.parent / 'frontend/app'
ACCEPTABLE_LICENSES = {'MIT', 'Apache-2.0', 'Apache-2.0 WITH LLVM-exception', '0BSD', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC', 'CC0-1.0', 'CC-BY-3.0', 'CC-BY-4.0', 'Python-2.0', 'Zlib', 'Unlicense', 'WTFPL', '(MIT OR Apache-2.0)', '(MPL-2.0 OR Apache-2.0)', '(MIT OR CC0-1.0)', '(Apache-2.0 OR MPL-1.1)', '(BSD-3-Clause OR GPL-2.0)', '(MIT AND BSD-3-Clause)', '(MIT AND Zlib)', '(WTFPL OR MIT)', '(AFL-2.1 OR BSD-3-Clause)'}
PACKAGE_EXCEPTIONS: Set[PackageInfo] = {('@mapbox/jsonlint-lines-primitives', '2.0.2', 'UNKNOWN', 'git://github.com/mapbox/jsonlint.git', 'http://zaa.ch', 'Zach Carter'), ('flatbuffers', '23.5.26', 'SEE LICENSE IN LICENSE', 'git+https://github.com/google/flatbuffers.git', 'https://google.github.io/flatbuffers/', 'The FlatBuffers project'), ('mapbox-gl', '1.13.3', 'SEE LICENSE IN LICENSE.txt', 'git://github.com/mapbox/mapbox-gl-js.git', 'Unknown', 'Unknown'), ('mapbox-gl', '1.10.1', 'SEE LICENSE IN LICENSE.txt', 'git://github.com/mapbox/mapbox-gl-js.git', 'Unknown', 'Unknown'), ('cartocolor', '4.0.2', 'UNKNOWN', 'https://github.com/cartodb/cartocolor', 'http://carto.com/', 'Unknown'), ('colorbrewer', '1.0.0', 'Apache*', 'https://github.com/saikocat/colorbrewer', 'http://colorbrewer2.org/', 'Cynthia Brewer')}

def get_license_type(package: PackageInfo) -> str:
    if False:
        i = 10
        return i + 15
    'Return the license type string for a dependency entry.'
    return package[2]

def check_licenses(licenses) -> NoReturn:
    if False:
        return 10
    licenses_json = json.loads(licenses[len(licenses) - 1])
    assert licenses_json['type'] == 'table'
    packages = [cast(PackageInfo, tuple(package)) for package in licenses_json['data']['body']]
    unused_exceptions = PACKAGE_EXCEPTIONS.difference(set(packages))
    if len(unused_exceptions) > 0:
        for exception in sorted(list(unused_exceptions)):
            print(f'Unused package exception, please remove: {exception}')
    bad_packages = [package for package in packages if get_license_type(package) not in ACCEPTABLE_LICENSES and package not in PACKAGE_EXCEPTIONS and ('workspace-aggregator' not in package[0])]
    if len(bad_packages) > 0:
        for package in bad_packages:
            print(f"Unacceptable license: '{get_license_type(package)}' (in {package})")
        print(f'{len(bad_packages)} unacceptable licenses')
        sys.exit(1)
    print(f'No unacceptable licenses')
    sys.exit(0)

def main() -> NoReturn:
    if False:
        i = 10
        return i + 15
    licenses_output = subprocess.check_output(['yarn', 'licenses', 'list', '--json', '--production', '--ignore-platform'], cwd=str(FRONTEND_DIR_LIB)).decode().splitlines()
    licenses_output = licenses_output + subprocess.check_output(['yarn', 'licenses', 'list', '--json', '--production', '--ignore-platform'], cwd=str(FRONTEND_DIR_APP)).decode().splitlines()
    check_licenses(licenses_output)
if __name__ == '__main__':
    main()