"""
Test execute code functionality
"""
import subprocess
import os
import sys
import logging
logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

def call(command):
    if False:
        for i in range(10):
            print('nop')
    print('Executing: ' + command)
    return subprocess.call(command, shell=True)
ENV_RESOURCES_PATH = os.getenv('RESOURCES_PATH', '/resources')
exit_code = call(ENV_RESOURCES_PATH + '/scripts/execute_code.py ' + ENV_RESOURCES_PATH + '/tests/ml-job/')
if exit_code == 0:
    print('Code execution test successfull.')
else:
    print('Code execution test failed.')