from __future__ import annotations
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from common_precommit_utils import AIRFLOW_SOURCES_ROOT_PATH, read_airflow_version

def update_version(pattern: re.Pattern, v: str, file_path: Path):
    if False:
        print('Hello World!')
    print(f'Checking {pattern} in {file_path}')
    with file_path.open('r+') as f:
        file_content = f.read()
        if not pattern.search(file_content):
            raise Exception(f"Pattern {pattern!r} doesn't found in {file_path!r} file")
        new_content = pattern.sub(f'\\g<1>{v}\\g<2>', file_content)
        if file_content == new_content:
            return
        print('    Updated.')
        f.seek(0)
        f.truncate()
        f.write(new_content)
REPLACEMENTS = {'^(FROM apache\\/airflow:).*($)': 'docs/docker-stack/docker-examples/extending/*/Dockerfile', '(apache\\/airflow:)[^-]*(\\-)': 'docs/docker-stack/entrypoint.rst', '(`apache/airflow:(?:slim-)?)[0-9].*?((?:-pythonX.Y)?`)': 'docs/docker-stack/README.md', '(\\(Assuming Airflow version `).*(`\\))': 'docs/docker-stack/README.md'}
if __name__ == '__main__':
    version = read_airflow_version()
    print(f'Current version: {version}')
    for (regexp, p) in REPLACEMENTS.items():
        text_pattern = re.compile(regexp, flags=re.MULTILINE)
        files = list(AIRFLOW_SOURCES_ROOT_PATH.glob(p))
        if not files:
            print(f'ERROR! No files matched on {p}')
        for file in files:
            update_version(text_pattern, version, file)