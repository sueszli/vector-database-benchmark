import os
import re
import sys
import pytest
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
from nb_helper import get_output_stream_from_cell
abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, '..', '..', '..', '..', 'Tutorials', 'CNTK_203_Reinforcement_Learning_Basics.ipynb')
notebook_timeoutSeconds = 450

def test_cntk_203_reinforcement_learning_basics_noErrors(nb):
    if False:
        while True:
            i = 10
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell for output in cell['outputs'] if output.output_type == 'error']
    print(errors)
    assert errors == []

def test_cntk_203_reinforcement_learning_basics_tasks_are_solved(nb):
    if False:
        for i in range(10):
            print('nop')
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    testCells = [cell for cell in nb.cells if re.search('Task solved in[ :]', get_output_stream_from_cell(cell))]
    assert len(testCells) == 2