import os
import sys
import pytest
import re
import numpy as np
abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, '..', '..', '..', '..', 'Tutorials', 'CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb')

def test_cntk_103d_mnist_convolutionalneuralnetwork_noErrors(nb):
    if False:
        while True:
            i = 10
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    errors = [output for cell in nb.cells if 'outputs' in cell for output in cell['outputs'] if output.output_type == 'error']
    assert errors == []
notebook_timeoutSeconds = 1500
expectedEvalErrorByDeviceId = {-1: [1.35, 1.05], 0: [1.35, 1.05]}

def test_cntk_103d_mnist_convolutionalneuralnetwork_trainerror(nb, device_id):
    if False:
        i = 10
        return i + 15
    if os.getenv('OS') == 'Windows_NT' and sys.version_info[0] == 2:
        pytest.skip('tests with Python 2.7 on Windows are not stable in the CI environment. ')
    metrics = []
    for cell in nb.cells:
        try:
            if cell.cell_type == 'code':
                m = re.search('Average test error: (?P<metric>\\d+\\.\\d+)%', cell.outputs[0]['text'])
                if m:
                    metrics.append(float(m.group('metric')))
        except IndexError:
            pass
        except KeyError:
            pass
    assert np.allclose(expectedEvalErrorByDeviceId[device_id], metrics, atol=0.4)