import os
import sys
import pytest
import re
import numpy as np
from _cntk_py import force_deterministic_algorithms
force_deterministic_algorithms()
abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, '..', '..', '..', '..', 'Tutorials', 'CNTK_301_Image_Recognition_with_Deep_Transfer_Learning.ipynb')
notebook_timeoutSeconds = 900
notebook_deviceIdsToRun = [0]

def test_CNTK_301_Image_Recognition_with_Deep_Transfer_Learning_noErrors(nb):
    if False:
        i = 10
        return i + 15
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell for output in cell['outputs'] if output.output_type == 'error']
    print(errors)
    assert errors == []