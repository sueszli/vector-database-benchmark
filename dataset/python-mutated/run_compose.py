import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List
PROG = Path(__file__).name
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module. To run this script, run the ./{PROG} command')
E2E_DIR = Path(__file__).resolve().parent
ROOT_DIR = E2E_DIR.parent
IN_CONTAINER_HOME = Path('/home/circleci/repo')

def is_relative_to(path: Path, *other):
    if False:
        while True:
            i = 10
    'Return True if the path is relative to another path or False.\n\n    This function is backported from Python 3.9 - Path.is_relative_to.\n    '
    try:
        path.relative_to(*other)
        return True
    except ValueError:
        return False

def display_usage():
    if False:
        for i in range(10):
            print('nop')
    print(textwrap.dedent(f'    usage: {PROG} [-h] [ARGS ...]\n\n    Runs the compose environment for E2E tests\n\n    If additional arguments are passed, it will be executed as a command\n    in the environment.\n\n    If no additional arguments are passed, the bash console will be started.\n\n    The script automatically enters the corresponding directory in the container,\n    so you can safely pass relatively paths as script arguments.\n\n    example:\n\n    To run a single test, run command:\n    ./{PROG} ../scripts/run_e2e_tests.py -u ./specs/st_code.spec.js\n\n    positional arguments:\n      ARGS  sequence of program arguments\n\n    optional arguments:\n      -h, --help    show this help message and exit    '))

def parse_args() -> List[str]:
    if False:
        return 10
    if len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help'):
        display_usage()
        sys.exit(0)
    return sys.argv[1:]

def get_container_cwd():
    if False:
        return 10
    cwd_path = Path(os.getcwd())
    if not is_relative_to(cwd_path, ROOT_DIR):
        print(textwrap.dedent(f'You must be in your repository directory to run this command.\nTo go to the repository, run command:\n    cd {str(ROOT_DIR)}'), file=sys.stderr)
        sys.exit(1)
    return str(IN_CONTAINER_HOME / cwd_path.relative_to(ROOT_DIR))

def main():
    if False:
        for i in range(10):
            print('nop')
    subprocess_args = parse_args()
    (ROOT_DIR / 'frontend' / 'test_results').mkdir(parents=True, exist_ok=True)
    in_container_working_directory = get_container_cwd()
    compose_file = str(E2E_DIR / 'docker-compose.yml')
    docker_compose_args = ['docker-compose', f'--file={compose_file}', 'run', '--rm', '--name=streamlit_e2e_tests', f'--workdir={in_container_working_directory}', 'streamlit_e2e_tests', *subprocess_args]
    try:
        subprocess.run(docker_compose_args, check=True)
    except subprocess.CalledProcessError as ex:
        sys.exit(ex.returncode)
main()