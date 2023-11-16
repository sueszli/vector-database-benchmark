import os
import sys

def version_check(python_env, major_python_version):
    if False:
        print('Hello World!')
    '\n        These are various tests to test the Python container image.\n        This file will be distributed via --py-files in the e2e tests.\n    '
    env_version = os.environ.get('PYSPARK_PYTHON', 'python3')
    print('Python runtime version check is: ' + str(sys.version_info[0] == major_python_version))
    print('Python environment version check is: ' + str(env_version == python_env))