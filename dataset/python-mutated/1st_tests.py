from __future__ import print_function
import os, sys
import numpy as np
from cntk import DeviceDescriptor
from cntk import placeholder
from cntk.layers import *
from cntk.internal.utils import *
from cntk.logging import *
from cntk.ops import splice
from cntk.cntk_py import reset_random_seed
from cntk.device import try_set_default_device
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, '..', '..', '..', '..', 'Examples', '1stSteps'))

def test_1st_steps_functional(device_id):
    if False:
        print('Hello World!')
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))
    reset_random_seed(0)
    from LogisticRegression_FunctionalAPI import final_loss, final_metric, final_samples, test_metric
    assert np.allclose(final_loss, 0.344399, atol=1e-05)
    assert np.allclose(final_metric, 0.1258, atol=0.0001)
    assert np.allclose(test_metric, 0.0811, atol=0.0001)
    assert final_samples == 20000

def test_1st_steps_graph(device_id):
    if False:
        while True:
            i = 10
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))
    reset_random_seed(0)
    from LogisticRegression_GraphAPI import trainer, evaluator, X_test, Y_test, data, label_one_hot
    assert np.allclose(trainer.previous_minibatch_loss_average, 0.1233455091714859, atol=1e-05)
    assert trainer.previous_minibatch_sample_count == 32
    i = 0
    x = X_test[i:i + 32]
    y = Y_test[i:i + 32]
    metric = evaluator.test_minibatch({data: x, label_one_hot: y})
    assert np.allclose(metric, 0.0625, atol=1e-05)

def test_1st_steps_mnist(device_id):
    if False:
        while True:
            i = 10
    from cntk.ops.tests.ops_test_utils import cntk_device
    cntk_py.force_deterministic_algorithms()
    cntk_py.set_fixed_random_seed(1)
    try_set_default_device(cntk_device(device_id))
    reset_random_seed(0)
    from MNIST_Complex_Training import final_loss, final_metric, final_samples, test_metric
    print(final_loss, final_metric, final_samples, test_metric)
    assert np.allclose(final_loss, 0.00906, atol=1e-05)
    assert np.allclose(final_metric, 0.0027, atol=0.001)
    assert np.allclose(test_metric, 0.0063, atol=0.001)
    assert final_samples == 54000
if __name__ == '__main__':
    test_1st_steps_mnist(0)
    test_1st_steps_functional(0)
    test_1st_steps_graph(0)