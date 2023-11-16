import os, sys
import numpy as np
import shutil
from cntk.device import try_set_default_device
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Image', 'Classification', 'MLP', 'Python'))
from SimpleMNIST import simple_mnist
TOLERANCE_ABSOLUTE = 0.1

def test_simple_mnist_error(device_id):
    if False:
        print('Hello World!')
    abs_path = os.path.dirname(os.path.abspath(__file__))
    tb_logdir = os.path.join(abs_path, 'simple_mnist_test_log')
    if os.path.exists(tb_logdir):
        shutil.rmtree(tb_logdir)
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))
    test_error = simple_mnist(tb_logdir)
    expected_test_error = 0.09
    assert np.allclose(test_error, expected_test_error, atol=TOLERANCE_ABSOLUTE)
    tb_files = 0
    for tb_file in os.listdir(tb_logdir):
        assert tb_file.startswith('events.out.tfevents')
        tb_files += 1
    assert tb_files == 1