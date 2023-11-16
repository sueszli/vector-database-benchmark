import os
import re
import numpy as np
import sys
import pytest
abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, '..', '..', '..', '..', 'Tutorials', 'CNTK_103A_MNIST_DataLoader.ipynb')
datadir = os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Image', 'DataSets', 'MNIST')
reWeekly = re.compile('^weekly\\b', re.IGNORECASE)
notebook_deviceIdsToRun = [-1]

@pytest.fixture(scope='module')
def clean_data(device_id):
    if False:
        return 10
    if device_id in notebook_deviceIdsToRun:
        import subprocess
        args = ['git', 'clean', '-fdx', datadir]
        subprocess.check_call(args)

@pytest.mark.skipif(not reWeekly.search(os.environ.get('TEST_TAG')), reason='only runs as part of the weekly tests')
def test_cntk_103a_mnist_dataloader_noErrors(clean_data, nb):
    if False:
        for i in range(10):
            print('nop')
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell for output in cell['outputs'] if output.output_type == 'error']
    assert errors == []
    assert os.path.exists(os.path.join(datadir, 'Test-28x28_cntk_text.txt'))
    assert os.path.exists(os.path.join(datadir, 'Train-28x28_cntk_text.txt'))