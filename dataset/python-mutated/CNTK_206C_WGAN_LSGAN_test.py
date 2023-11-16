import os
import sys
import pytest
import re
import numpy as np
abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, '..', '..', '..', '..', 'Tutorials', 'CNTK_206C_WGAN_LSGAN.ipynb')
notebook_deviceIdsToRun = [0]
notebook_timeoutSeconds = 900

def test_cntk_206C_wgan_lsgan_noErrors(nb):
    if False:
        i = 10
        return i + 15
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell for output in cell['outputs'] if output.output_type == 'error']
    assert errors == []