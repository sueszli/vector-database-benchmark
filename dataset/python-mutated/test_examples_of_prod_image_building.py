from __future__ import annotations
import glob
import os
import re
from functools import lru_cache
from pathlib import Path
import pytest
import requests
from docker_tests.command_utils import run_command
from docker_tests.constants import SOURCE_ROOT
DOCKER_EXAMPLES_DIR = SOURCE_ROOT / 'docs' / 'docker-stack' / 'docker-examples'

@lru_cache(maxsize=None)
def get_latest_airflow_version_released():
    if False:
        return 10
    response = requests.get('https://pypi.org/pypi/apache-airflow/json')
    response.raise_for_status()
    return response.json()['info']['version']

@pytest.mark.skipif(os.environ.get('CI') == 'true', reason='Skipping the script builds on CI! They take very long time to build.')
@pytest.mark.parametrize('script_file', glob.glob(f'{DOCKER_EXAMPLES_DIR}/**/*.sh', recursive=True))
def test_shell_script_example(script_file):
    if False:
        for i in range(10):
            print('nop')
    run_command(['bash', script_file])

@pytest.mark.parametrize('dockerfile', glob.glob(f'{DOCKER_EXAMPLES_DIR}/**/Dockerfile', recursive=True))
def test_dockerfile_example(dockerfile):
    if False:
        i = 10
        return i + 15
    rel_dockerfile_path = Path(dockerfile).relative_to(DOCKER_EXAMPLES_DIR)
    image_name = str(rel_dockerfile_path).lower().replace('/', '-')
    content = Path(dockerfile).read_text()
    latest_released_version: str = get_latest_airflow_version_released()
    new_content = re.sub('FROM apache/airflow:.*', f'FROM apache/airflow:{latest_released_version}', content)
    try:
        run_command(['docker', 'build', '.', '--tag', image_name, '-f', '-'], cwd=str(Path(dockerfile).parent), input=new_content.encode())
    finally:
        run_command(['docker', 'rmi', '--force', image_name])