from __future__ import annotations
import os
from docker_tests.command_utils import run_command
DEFAULT_PYTHON_MAJOR_MINOR_VERSION = '3.8'
docker_image = os.environ.get('DOCKER_IMAGE', f'ghcr.io/apache/airflow/main/prod/python{DEFAULT_PYTHON_MAJOR_MINOR_VERSION}:latest')
print('Using docker image: ', docker_image)

def run_bash_in_docker(bash_script, **kwargs):
    if False:
        i = 10
        return i + 15
    docker_command = ['docker', 'run', '--rm', '-e', 'COLUMNS=180', '--entrypoint', '/bin/bash', docker_image, '-c', bash_script]
    return run_command(docker_command, **kwargs)

def run_python_in_docker(python_script, **kwargs):
    if False:
        while True:
            i = 10
    docker_command = ['docker', 'run', '--rm', '-e', 'COLUMNS=180', '-e', 'PYTHONDONTWRITEBYTECODE=true', docker_image, 'python', '-c', python_script]
    return run_command(docker_command, **kwargs)

def display_dependency_conflict_message():
    if False:
        while True:
            i = 10
    print("\n***** Beginning of the instructions ****\n\nThe image did not pass 'pip check' verification. This means that there are some conflicting dependencies\nin the image.\n\nIt can mean one of those:\n\n1) The main is currently broken (other PRs will fail with the same error)\n2) You changed some dependencies in setup.py or setup.cfg and they are conflicting.\n\n\n\nIn case 1) - apologies for the trouble.Please let committers know and they will fix it. You might\nbe asked to rebase to the latest main after the problem is fixed.\n\nIn case 2) - Follow the steps below:\n\n* try to build CI and then PROD image locally with breeze, adding --upgrade-to-newer-dependencies flag\n  (repeat it for all python versions)\n\nCI image:\n\n     breeze ci-image build --upgrade-to-newer-dependencies --python 3.8\n\nProduction image:\n\n     breeze ci-image build --production-image --upgrade-to-newer-dependencies --python 3.8\n\n* You will see error messages there telling which requirements are conflicting and which packages caused the\n  conflict. Add the limitation that caused the conflict to EAGER_UPGRADE_ADDITIONAL_REQUIREMENTS\n  variable in Dockerfile.ci. Note that the limitations might be different for Dockerfile.ci and Dockerfile\n  because not all packages are installed by default in the PROD Dockerfile. So you might find that you\n  only need to add the limitation to the Dockerfile.ci\n\n***** End of the instructions ****\n")