import os, sys
import numpy as np
import subprocess
import time
import re
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device, gpu
from cntk.ops.tests.ops_test_utils import cntk_device
import pytest
abs_path = os.path.dirname(os.path.abspath(__file__))
convnet_path = os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Image', 'Classification', 'ConvNet')
sys.path.append(os.path.join(convnet_path, 'Python'))
bs_scripts_path = os.path.join(convnet_path, 'BrainScript')
sys.path.append(bs_scripts_path)
from ConvNet_CIFAR10 import convnet_cifar10
TOLERANCE_ABSOLUTE = 0.2

def test_convnet_cifar_error(device_id):
    if False:
        return 10
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))
    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        dataset_path = os.path.join(extPath, 'Image', 'CIFAR', 'v0')
    else:
        dataset_path = os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Image', 'DataSets', 'CIFAR-10')
    error = convnet_cifar10(data_path=dataset_path, epoch_size=2000, minibatch_size=32, max_epochs=10)
    expected_error = 0.7
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)

def test_check_percentages_after_restarting_training(device_id):
    if False:
        i = 10
        return i + 15
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))
    configFile = 'ConvNet_CIFAR10.cntk'
    timeout_seconds = 60
    cntkPath = 'cntk'
    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        dataset_path = os.path.join(extPath, 'Image', 'CIFAR', 'v0')
    else:
        dataset_path = os.path.join(abs_path, '..', '..', '..', '..', 'Examples', 'Image', 'DataSets', 'CIFAR-10')
    cntkPath = os.environ['TEST_CNTK_BINARY']
    if sys.platform == 'win32':
        p = subprocess.Popen(['cygpath', '-aw', os.environ['TEST_CNTK_BINARY']], stdout=subprocess.PIPE)
        out = p.communicate()[0]
        cntkPath = out.decode(sys.getdefaultencoding()).strip()
    cmdStr = cntkPath + ' configFile=' + os.path.join(bs_scripts_path, configFile) + ' makeMode=true dataDir=' + dataset_path + ' TrainConvNet=[SGD=[maxEpochs=12]]'
    p = subprocess.Popen(cmdStr.split(' '), stdout=subprocess.PIPE)
    time.sleep(timeout_seconds)
    if p.poll() == 0:
        return
    p.kill()
    out = subprocess.check_output(cmdStr.split(' '), stderr=subprocess.STDOUT)
    all_percentages = re.findall('.* Epoch\\[ *\\d+ of \\d+]-Minibatch\\[ *\\d+- *\\d+, *(\\d+\\.\\d+)\\%\\].*', out.decode('utf-8'))
    expected_percentages = set(['14.29', '28.57', '57.14', '42.86', '71.43', '85.71', '100.00'])
    assert set(all_percentages) == expected_percentages