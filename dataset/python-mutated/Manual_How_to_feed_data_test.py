import os
import sys
import pytest
import re
import numpy as np
abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, '..', '..', '..', '..', 'Manual', 'Manual_How_to_feed_data.ipynb')

def test_manual_how_to_feed_data_noErrors(nb):
    if False:
        print('Hello World!')
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell for output in cell['outputs'] if output.output_type == 'error']
    assert errors == []