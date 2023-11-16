from __future__ import annotations
import subprocess
from docker_tests.command_utils import run_command
from docker_tests.docker_tests_utils import display_dependency_conflict_message, docker_image

class TestPythonPackages:

    def test_pip_dependencies_conflict(self):
        if False:
            while True:
                i = 10
        try:
            run_command(['docker', 'run', '--rm', '--entrypoint', '/bin/bash', docker_image, '-c', 'pip check'])
        except subprocess.CalledProcessError as ex:
            display_dependency_conflict_message()
            raise ex