"""
run_pytest.py is the script submitted to Azure ML that runs pytest.
pytest runs all tests in the specified test folder unless parameters
are set otherwise.
"""
import argparse
import subprocess
import logging
import os
import sys
from azureml.core import Run

def create_arg_parser():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('--testfolder', '-f', action='store', default='./tests/unit', help='Folder where tests are located')
    parser.add_argument('--num', action='store', default='99', help='test num')
    parser.add_argument('--testmarkers', '-m', action='store', default='not notebooks and not spark and not gpu', help='Specify test markers for test selection')
    parser.add_argument('--xmlname', '-j', action='store', default='reports/test-unit.xml', help='Test results')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    logger = logging.getLogger('submit_azureml_pytest.py')
    args = create_arg_parser()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.debug('junit_xml {}'.format(args.xmlname))
    run = Run.get_context()
    '\n    This is an example of a working subprocess.run for a unit test run:\n    subprocess.run(["pytest", "tests/unit",\n                    "-m", "not notebooks and not spark and not gpu",\n                    "--junitxml=reports/test-unit.xml"])\n    '
    logger.debug('args.junitxml {}'.format(args.xmlname))
    logger.debug('junit= --junitxml={}'.format(args.xmlname))
    pytest_cmd = ['pytest', '--durations=100', '--ignore=contrib', args.testfolder, '-m', args.testmarkers, '--junitxml={}'.format(args.xmlname)]
    logger.info('pytest run:{}'.format(' '.join(pytest_cmd)))
    subprocess.run(pytest_cmd)
    logger.debug('os.listdir files {}'.format(os.listdir('.')))
    name_of_upload = 'reports'
    path_on_disk = './reports'
    run.upload_folder(name_of_upload, path_on_disk)