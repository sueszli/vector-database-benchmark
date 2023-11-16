from __future__ import annotations
import os
import sys
from pathlib import Path
import re2
from packaging.version import parse as parse_version
PROJECT_SOURCE_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DB_FILE = PROJECT_SOURCE_ROOT_DIR / 'airflow' / 'utils' / 'db.py'
MIGRATION_PATH = PROJECT_SOURCE_ROOT_DIR / 'airflow' / 'migrations' / 'versions'
sys.path.insert(0, str(Path(__file__).parent.resolve()))

def revision_heads_map():
    if False:
        for i in range(10):
            print('nop')
    rh_map = {}
    pattern = 'revision = "[a-fA-F0-9]+"'
    airflow_version_pattern = 'airflow_version = "\\d+\\.\\d+\\.\\d+"'
    filenames = os.listdir(MIGRATION_PATH)

    def sorting_key(filen):
        if False:
            return 10
        prefix = filen.split('_')[0]
        return int(prefix) if prefix.isdigit() else 0
    sorted_filenames = sorted(filenames, key=sorting_key)
    for filename in sorted_filenames:
        if not filename.endswith('.py'):
            continue
        with open(os.path.join(MIGRATION_PATH, filename)) as file:
            content = file.read()
            revision_match = re2.search(pattern, content)
            airflow_version_match = re2.search(airflow_version_pattern, content)
            if revision_match and airflow_version_match:
                revision = revision_match.group(0).split('"')[1]
                version = airflow_version_match.group(0).split('"')[1]
                if parse_version(version) >= parse_version('2.0.0'):
                    rh_map[version] = revision
    return rh_map
if __name__ == '__main__':
    with open(DB_FILE) as file:
        content = file.read()
    pattern = '_REVISION_HEADS_MAP = {[^}]+\\}'
    match = re2.search(pattern, content)
    if not match:
        print(f'_REVISION_HEADS_MAP not found in {DB_FILE}. If this has been removed intentionally, please update scripts/ci/pre_commit/pre_commit_version_heads_map.py')
        sys.exit(1)
    existing_revision_heads_map = match.group(0)
    rh_map = revision_heads_map()
    updated_revision_heads_map = '_REVISION_HEADS_MAP = {\n'
    for (k, v) in rh_map.items():
        updated_revision_heads_map += f'    "{k}": "{v}",\n'
    updated_revision_heads_map += '}'
    if existing_revision_heads_map != updated_revision_heads_map:
        new_content = content.replace(existing_revision_heads_map, updated_revision_heads_map)
        with open(DB_FILE, 'w') as file:
            file.write(new_content)
        print('_REVISION_HEADS_MAP updated in db.py. Please commit the changes.')
        sys.exit(1)