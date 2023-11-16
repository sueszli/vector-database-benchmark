"""
Execute code
"""
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--requirements-only', help='Only install requirements without executing the code.', action='store_true')
parser.add_argument('--code-only', help='Only execute code without installing requirements.', action='store_true')
(args, unknown) = parser.parse_known_args()
if unknown:
    log.info('Unknown arguments ' + str(unknown))
start_time = time.time()
log.info('Execute Code...')

def call(command):
    if False:
        for i in range(10):
            print('nop')
    log.info('Executing: ' + command)
    return subprocess.call(command, shell=True)
RESOURCES_PATH = os.getenv('RESOURCES_PATH', '/resources')
WORKSPACE_HOME = os.getenv('WORKSPACE_HOME', '/workspace')
CONDA_ROOT = os.getenv('CONDA_ROOT', '/opt/conda')
EXECUTE_CODE = os.getenv('EXECUTE_CODE', ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else None)
code_path = None
if not EXECUTE_CODE:
    log.info('EXECUTE_CODE env variable is not set.')
    sys.exit(1)
if os.path.exists(EXECUTE_CODE):
    code_path = EXECUTE_CODE
elif EXECUTE_CODE.lower().startswith(('git+', 'svn+', 'hg+', 'bzr+')):
    try:
        from pip._internal.vcs import vcs
        vcs_url = EXECUTE_CODE
        (vc_type, _) = vcs_url.split('+', 1)
        import atexit
        import shutil
        import tempfile
        code_path = tempfile.mkdtemp()

        def cleanup():
            if False:
                while True:
                    i = 10
            shutil.rmtree(code_path)
        atexit.register(cleanup)
        vcs.get_backend(vc_type).export(code_path, url=vcs_url)
        from pip._internal.models.link import Link
        subdir = Link(vcs_url).subdirectory_fragment
        if subdir:
            code_path = os.path.join(code_path, subdir.lstrip('/'))
    except Exception as ex:
        log.exception('Failed to clone repository via pip internal.')
if not code_path or not os.path.exists(code_path):
    log.info('No code artifacts could be found for ' + EXECUTE_CODE)
    sys.exit(1)
main_script = code_path
if os.path.isfile(code_path):
    code_path = os.path.dirname(os.path.realpath(code_path))
if os.path.isdir(code_path):
    pip_runtime = 'pip'
    python_runtime = 'python'
    bash_runtime = '/bin/bash'
    if not args.code_only:
        log.info('Searching requirements at path ' + main_script)
        conda_env_path = os.path.join(code_path, 'environment.yml')
        if os.path.isfile(conda_env_path):
            conda_env_name = 'conda-env'
            log.info('Installing conda environment from ' + conda_env_path)
            if call('conda env create -n ' + conda_env_name + ' -f ' + conda_env_path) == 0:
                pip_runtime = CONDA_ROOT + '/envs/' + conda_env_name + '/bin/pip'
                python_runtime = CONDA_ROOT + '/envs/' + conda_env_name + '/bin/python'
                bash_runtime = 'PATH=' + CONDA_ROOT + '/envs/' + conda_env_name + '/bin/:$PATH /bin/bash'
            else:
                log.info('Failed to install conda env from ' + conda_env_path)
        setup_path = os.path.join(code_path, 'setup.sh')
        if os.path.isfile(setup_path):
            log.info('Running setup from ' + setup_path)
            if call(bash_runtime + ' ' + setup_path) != 0:
                log.info('Failed to run setup.sh from ' + setup_path)
        requirements_path = os.path.join(code_path, 'requirements.txt')
        if os.path.isfile(requirements_path):
            log.info('Installing requirements from ' + requirements_path)
            if call(pip_runtime + ' install --no-cache-dir -r ' + requirements_path) != 0:
                log.info('Failed to install requirements.txt from ' + requirements_path)
    if args.requirements_only:
        log.info('Finished installing requirements. Code execution is deactivated.')
        sys.exit(0)
    log.info('Executing python code at path ' + main_script)
    exit_code = call(python_runtime + ' "' + main_script + '"')
    if exit_code > 0:
        log.info('Execution failed with exit code: ' + str(exit_code))
        if os.path.isdir(main_script):
            log.info('Please make sure that there is a main module (e.g. __main__.py) at this path: ' + main_script)
    else:
        log.info('Code execution finished successfully.')
    log.info('Elapsed time: ' + str(timedelta(seconds=time.time() - start_time)))
    sys.exit(exit_code)
log.info('Something went wrong. This code should have never been reached.')
sys.exit(1)