import requests
import pathlib
import re
_VERSIONS_URL = 'https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json'

def parse_version(v):
    if False:
        return 10
    return tuple((int(part) for part in re.split('\\W', v)[:3]))

def get_github_python_versions():
    if False:
        i = 10
        return i + 15
    versions_json = requests.get(_VERSIONS_URL).json()
    raw_versions = [v['version'] for v in versions_json]
    versions = []
    for version_str in raw_versions:
        if '-' in version_str:
            continue
        (major, minor, patch) = parse_version(version_str)
        if major == 3 and minor < 5:
            continue
        elif major == 2 and minor < 7:
            continue
        versions.append(version_str)
    return versions
if __name__ == '__main__':
    versions = sorted(get_github_python_versions(), key=parse_version)
    build_yml = pathlib.Path(__file__).parent.parent / '.github' / 'workflows' / 'build.yml'
    transformed = []
    for line in open(build_yml):
        if line.startswith('        python-version: ['):
            newversions = f"        python-version: [{', '.join((v for v in versions))}]\n"
            if newversions != line:
                print('Adding new versions')
                print('Old:', line)
                print('New:', newversions)
            line = newversions
        transformed.append(line)
    exclusions = []
    for v in versions:
        if v.startswith('3.11'):
            exclusions.append('          - os: macos-latest\n')
            exclusions.append(f'            python-version: {v}\n')
    test_wheels = transformed.index('  test-wheels:\n')
    first_line = transformed.index('        exclude:\n', test_wheels)
    last_line = transformed.index('\n', first_line)
    transformed = transformed[:first_line + 1] + exclusions + transformed[last_line:]
    with open(build_yml, 'w') as o:
        o.write(''.join(transformed))