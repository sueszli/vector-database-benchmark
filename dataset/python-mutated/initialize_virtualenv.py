from __future__ import annotations
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
if __name__ not in ('__main__', '__mp_main__'):
    raise SystemExit(f'This file is intended to be executed as an executable program. You cannot use it as a module.To run this script, run the ./{__file__} command')

def clean_up_airflow_home(airflow_home: Path):
    if False:
        while True:
            i = 10
    if airflow_home.exists():
        print(f'Removing {airflow_home}')
        shutil.rmtree(airflow_home, ignore_errors=True)

def check_if_in_virtualenv() -> bool:
    if False:
        while True:
            i = 10
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def check_for_package_extras() -> str:
    if False:
        i = 10
        return i + 15
    "\n    check if the user provided any extra packages to install.\n    defaults to package 'devel'.\n    "
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            print('Provide extras as 1 argument like: "devel,google,snowflake"')
            sys.exit(1)
        return sys.argv[1]
    return 'devel'

def pip_install_requirements() -> int:
    if False:
        while True:
            i = 10
    '\n    install the requirements of the current python version.\n    return 0 if success, anything else is an error.\n    '
    extras = check_for_package_extras()
    print(f'''\nInstalling requirements.\n\nAirflow is installed with "{extras}" extra.\n\n----------------------------------------------------------------------------------------\n\nIMPORTANT NOTE ABOUT EXTRAS !!!\n\nYou can specify extras as single coma-separated parameter to install. For example\n\n* google,amazon,microsoft.azure\n* devel_all\n\nNote that "devel_all" installs all possible dependencies and we have > 600 of them,\nwhich might not be possible to install cleanly on your host because of lack of\nsystem packages. It's easier to install extras one-by-one as needed.\n\n----------------------------------------------------------------------------------------\n\n''')
    version = get_python_version()
    constraint = f'https://raw.githubusercontent.com/apache/airflow/constraints-main/constraints-source-providers-{version}.txt'
    pip_install_command = ['pip', 'install', '-e', f'.[{extras}]', '--constraint', constraint]
    quoted_command = ' '.join([shlex.quote(parameter) for parameter in pip_install_command])
    print()
    print(f'Running command: \n   {quoted_command}\n')
    e = subprocess.run(pip_install_command)
    return e.returncode

def get_python_version() -> str:
    if False:
        while True:
            i = 10
    '\n    return the version of python we are running.\n    '
    major = sys.version_info[0]
    minor = sys.version_info[1]
    return f'{major}.{minor}'

def main():
    if False:
        while True:
            i = 10
    '\n    Setup local virtual environment.\n    '
    airflow_home_dir = Path(os.environ.get('AIRFLOW_HOME', Path.home() / 'airflow'))
    airflow_sources = Path(__file__).resolve().parents[2]
    if not check_if_in_virtualenv():
        print("Local virtual environment not activated.\nPlease create and activate it first. (for example using 'python3 -m venv venv && source venv/bin/activate')")
        sys.exit(1)
    print('Initializing environment...')
    print(f'This will remove the folder {airflow_home_dir} and reset all the databases!')
    response = input('Are you sure? (y/N/q)')
    if response != 'y':
        sys.exit(2)
    print(f'\nWiping and recreating {airflow_home_dir}')
    if airflow_home_dir == airflow_sources:
        print('AIRFLOW_HOME and Source code are in the same path')
        print(f'When running this script it will delete all files in path {airflow_home_dir} to clear dynamic files like config/logs/db')
        print('Please move the airflow source code elsewhere to avoid deletion')
        sys.exit(3)
    clean_up_airflow_home(airflow_home_dir)
    return_code = pip_install_requirements()
    if return_code != 0:
        print('To solve persisting issues with the installation, you might need the prerequisites installed on your system.\n Try running the command below and rerun virtualenv installation\n')
        os_type = sys.platform
        if os_type == 'darwin':
            print('brew install sqlite mysql postgresql openssl')
            print('export LDFLAGS="-L/usr/local/opt/openssl/lib"')
            print('export CPPFLAGS="-I/usr/local/opt/openssl/include"')
        else:
            print('sudo apt install build-essential python3-dev libsqlite3-dev openssl sqlite default-libmysqlclient-dev libmysqlclient-dev postgresql')
        sys.exit(4)
    print('\nResetting AIRFLOW sqlite database...')
    env = os.environ.copy()
    env['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
    env['AIRFLOW__CORE__UNIT_TEST_MODE'] = 'False'
    env['AIRFLOW__DATABASE__SQL_ALCHEMY_POOL_ENABLED'] = 'False'
    env['AIRFLOW__CORE__DAGS_FOLDER'] = f'{airflow_sources}/empty'
    env['AIRFLOW__CORE__PLUGINS_FOLDER'] = f'{airflow_sources}/empty'
    subprocess.run(['airflow', 'db', 'reset', '--yes'], env=env)
    print('\nResetting AIRFLOW sqlite unit test database...')
    env = os.environ.copy()
    env['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'True'
    env['AIRFLOW__CORE__UNIT_TEST_MODE'] = 'False'
    env['AIRFLOW__DATABASE__SQL_ALCHEMY_POOL_ENABLED'] = 'False'
    env['AIRFLOW__CORE__DAGS_FOLDER'] = f'{airflow_sources}/empty'
    env['AIRFLOW__CORE__PLUGINS_FOLDER'] = f'{airflow_sources}/empty'
    subprocess.run(['airflow', 'db', 'reset', '--yes'], env=env)
    print('\nInitialization of environment complete! Go ahead and develop Airflow!')
if __name__ == '__main__':
    main()