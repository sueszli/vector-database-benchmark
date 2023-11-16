"""
Main Workspace Run Script
"""
import logging
import math
import os
import sys
from subprocess import call
from urllib.parse import quote
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)
log.info('Starting...')

def set_env_variable(env_variable: str, value: str, ignore_if_set: bool=False):
    if False:
        return 10
    if ignore_if_set and os.getenv(env_variable, None):
        return
    call('export ' + env_variable + '="' + value + '"', shell=True)
    os.environ[env_variable] = value
ENV_JUPYTERHUB_SERVICE_PREFIX = os.getenv('JUPYTERHUB_SERVICE_PREFIX', None)
ENV_NAME_WORKSPACE_BASE_URL = 'WORKSPACE_BASE_URL'
base_url = os.getenv(ENV_NAME_WORKSPACE_BASE_URL, '')
if ENV_JUPYTERHUB_SERVICE_PREFIX:
    base_url = ENV_JUPYTERHUB_SERVICE_PREFIX
if not base_url.startswith('/'):
    base_url = '/' + base_url
base_url = base_url.rstrip('/').strip()
base_url = quote(base_url, safe='/%')
set_env_variable(ENV_NAME_WORKSPACE_BASE_URL, base_url)
ENV_MAX_NUM_THREADS = os.getenv('MAX_NUM_THREADS', None)
if ENV_MAX_NUM_THREADS:
    if ENV_MAX_NUM_THREADS.lower() == 'auto':
        ENV_MAX_NUM_THREADS = str(math.ceil(os.cpu_count()))
        try:
            cpu_count = math.ceil(int(os.popen('cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us').read().replace('\n', '')) / 100000)
            if cpu_count > 0 and cpu_count < os.cpu_count():
                ENV_MAX_NUM_THREADS = str(cpu_count)
        except:
            pass
        if not ENV_MAX_NUM_THREADS or not ENV_MAX_NUM_THREADS.isnumeric() or ENV_MAX_NUM_THREADS == '0':
            ENV_MAX_NUM_THREADS = '4'
        if int(ENV_MAX_NUM_THREADS) > 8:
            ENV_MAX_NUM_THREADS = str(int(ENV_MAX_NUM_THREADS) - 1)
        if int(ENV_MAX_NUM_THREADS) > 32:
            ENV_MAX_NUM_THREADS = '32'
    set_env_variable('OMP_NUM_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('OPENBLAS_NUM_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('MKL_NUM_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('VECLIB_MAXIMUM_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('NUMEXPR_NUM_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('NUMEXPR_MAX_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('NUMBA_NUM_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('SPARK_WORKER_CORES', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('BLIS_NUM_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
    set_env_variable('TBB_NUM_THREADS', ENV_MAX_NUM_THREADS, ignore_if_set=True)
ENV_RESOURCES_PATH = os.getenv('RESOURCES_PATH', '/resources')
ENV_WORKSPACE_HOME = os.getenv('WORKSPACE_HOME', '/workspace')
script_arguments = ' ' + ' '.join(sys.argv[1:])
EXECUTE_CODE = os.getenv('EXECUTE_CODE', None)
if EXECUTE_CODE:
    sys.exit(call('cd ' + ENV_WORKSPACE_HOME + ' && python ' + ENV_RESOURCES_PATH + '/scripts/execute_code.py' + script_arguments, shell=True))
sys.exit(call('python ' + ENV_RESOURCES_PATH + '/scripts/run_workspace.py' + script_arguments, shell=True))