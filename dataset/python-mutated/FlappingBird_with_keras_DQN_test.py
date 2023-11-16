import numpy as np
import os
import platform
import sys
import pytest
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['KERAS_BACKEND'] = 'cntk'
from cntk.device import try_set_default_device, gpu
abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'ReinforcementLearning', 'FlappingBirdWithKeras')
sys.path.append(example_dir)
current_dir = os.getcwd()
os.chdir(example_dir)
linux_only = pytest.mark.skipif(sys.platform == 'win32', reason='temporarily disable these two tests on Windows due to an issue introduced by adding onnx to our CI.')

@linux_only
def test_FlappingBird_with_keras_DQN_noerror(device_id):
    if False:
        for i in range(10):
            print('nop')
    if platform.system() != 'Windows':
        pytest.skip('Test only runs on Windows, pygame video device requirement constraint')
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))
    sys.path.append(example_dir)
    current_dir = os.getcwd()
    os.chdir(example_dir)
    import FlappingBird_with_keras_DQN as fbgame
    model = fbgame.buildmodel()
    args = {'mode': 'Run'}
    res = fbgame.trainNetwork(model, args, internal_testing=True)
    np.testing.assert_array_equal(res, 0, err_msg='Error in running Flapping Bird example', verbose=True)
    args = {'mode': 'Train'}
    res = fbgame.trainNetwork(model, args, internal_testing=True)
    np.testing.assert_array_equal(res, 0, err_msg='Error in testing Flapping Bird example', verbose=True)
    os.chdir(current_dir)
    print('Done')