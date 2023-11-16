import os
import sys
import filecmp
import tempfile
import subprocess
SPARK_HOME = os.environ.get('SPARK_HOME', os.getcwd())

def fail(msg):
    if False:
        i = 10
        return i + 15
    print(msg)
    sys.exit(-1)

def run_cmd(cmd):
    if False:
        print('Hello World!')
    print(f'RUN: {cmd}')
    if isinstance(cmd, list):
        return subprocess.check_output(cmd).decode('utf-8')
    else:
        return subprocess.check_output(cmd.split(' ')).decode('utf-8')

def check_connect_protos():
    if False:
        i = 10
        return i + 15
    print('Start checking the generated codes in pyspark-connect.')
    with tempfile.TemporaryDirectory() as tmp:
        run_cmd(f'{SPARK_HOME}/dev/connect-gen-protos.sh {tmp}')
        result = filecmp.dircmp(f'{SPARK_HOME}/python/pyspark/sql/connect/proto/', tmp, ignore=['__init__.py', '__pycache__'])
        success = True
        if len(result.left_only) > 0:
            print(f'Unexpected files: {result.left_only}')
            success = False
        if len(result.right_only) > 0:
            print(f'Missing files: {result.right_only}')
            success = False
        if len(result.funny_files) > 0:
            print(f'Incomparable files: {result.funny_files}')
            success = False
        if len(result.diff_files) > 0:
            print(f'Different files: {result.diff_files}')
            success = False
        if success:
            print('Finish checking the generated codes in pyspark-connect: SUCCESS')
        else:
            fail("Generated files for pyspark-connect are out of sync! If you have touched files under connector/connect/common/src/main/protobuf/, please run ./dev/connect-gen-protos.sh. If you haven't touched any file above, please rebase your PR against main branch.")
check_connect_protos()