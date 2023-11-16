"""Retrieve collection detail."""
from __future__ import annotations
import json
import os
import re
import sys
import yaml
NUMERIC_IDENTIFIER = '(?:0|[1-9][0-9]*)'
ALPHANUMERIC_IDENTIFIER = '(?:[0-9]*[a-zA-Z-][a-zA-Z0-9-]*)'
PRE_RELEASE_IDENTIFIER = '(?:' + NUMERIC_IDENTIFIER + '|' + ALPHANUMERIC_IDENTIFIER + ')'
BUILD_IDENTIFIER = '[a-zA-Z0-9-]+'
VERSION_CORE = NUMERIC_IDENTIFIER + '\\.' + NUMERIC_IDENTIFIER + '\\.' + NUMERIC_IDENTIFIER
PRE_RELEASE = '(?:-' + PRE_RELEASE_IDENTIFIER + '(?:\\.' + PRE_RELEASE_IDENTIFIER + ')*)?'
BUILD = '(?:\\+' + BUILD_IDENTIFIER + '(?:\\.' + BUILD_IDENTIFIER + ')*)?'
SEMVER_REGULAR_EXPRESSION = '^' + VERSION_CORE + PRE_RELEASE + BUILD + '$'

def validate_version(version):
    if False:
        print('Hello World!')
    'Raise exception if the provided version is not None or a valid semantic version.'
    if version is None:
        return
    if not re.match(SEMVER_REGULAR_EXPRESSION, version):
        raise Exception('Invalid version number "{0}". Collection version numbers must follow semantic versioning (https://semver.org/).'.format(version))

def read_manifest_json(collection_path):
    if False:
        print('Hello World!')
    'Return collection information from the MANIFEST.json file.'
    manifest_path = os.path.join(collection_path, 'MANIFEST.json')
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
        collection_info = manifest.get('collection_info') or {}
        result = dict(version=collection_info.get('version'))
        validate_version(result['version'])
    except Exception as ex:
        raise Exception('{0}: {1}'.format(os.path.basename(manifest_path), ex)) from None
    return result

def read_galaxy_yml(collection_path):
    if False:
        return 10
    'Return collection information from the galaxy.yml file.'
    galaxy_path = os.path.join(collection_path, 'galaxy.yml')
    if not os.path.exists(galaxy_path):
        return None
    try:
        with open(galaxy_path, encoding='utf-8') as galaxy_file:
            galaxy = yaml.safe_load(galaxy_file)
        result = dict(version=galaxy.get('version'))
        validate_version(result['version'])
    except Exception as ex:
        raise Exception('{0}: {1}'.format(os.path.basename(galaxy_path), ex)) from None
    return result

def main():
    if False:
        return 10
    'Retrieve collection detail.'
    collection_path = sys.argv[1]
    try:
        result = read_manifest_json(collection_path) or read_galaxy_yml(collection_path) or {}
    except Exception as ex:
        result = dict(error='{0}'.format(ex))
    print(json.dumps(result))
if __name__ == '__main__':
    main()