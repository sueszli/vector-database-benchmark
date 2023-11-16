import os
import sys
import pytest
import re
import numpy
abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, '..', '..', '..', '..', 'Tutorials', 'CNTK_303_Deep_Structured_Semantic_Modeling_with_LSTM_Networks.ipynb')
notebook_timeoutSeconds = 600
expectedEvalErrorByDeviceId = {-1: [0.04, 0.04], 0: [0.04, 0.04]}

def test_cntk_303_deep_structured_semantic_modeling_with_lstm_networks_noErrors(nb):
    if False:
        while True:
            i = 10
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell for output in cell['outputs'] if output.output_type == 'error']
    print(errors)
    assert errors == []

def test_cntk_303_deep_structured_semantic_modeling_with_lstm_networks_trainerror(nb):
    if False:
        return 10
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    metrics = []
    for cell in nb.cells:
        try:
            if cell.cell_type == 'code':
                m = re.search('Finished Evaluation.* metric = (?P<metric>\\d+\\.\\d+)%', cell.outputs[0]['text'])
                if m:
                    metrics.append(float(m.group('metric')))
        except IndexError:
            pass
        except KeyError:
            pass
    expectedMetrics = [0.03]
    assert numpy.allclose(expectedMetrics, metrics, atol=0.02)