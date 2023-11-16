from unittest import mock
import pytest
from lightning.fabric.utilities import device_parser
from lightning.fabric.utilities.exceptions import MisconfigurationException
_PRETEND_N_OF_GPUS = 16

@pytest.mark.parametrize(('devices', 'expected_root_gpu'), [pytest.param(None, None, id='No gpus, expect gpu root device to be None'), pytest.param([0], 0, id='Oth gpu, expect gpu root device to be 0.'), pytest.param([1], 1, id='1st gpu, expect gpu root device to be 1.'), pytest.param([3], 3, id='3rd gpu, expect gpu root device to be 3.'), pytest.param([1, 2], 1, id='[1, 2] gpus, expect gpu root device to be 1.')])
def test_determine_root_gpu_device(devices, expected_root_gpu):
    if False:
        i = 10
        return i + 15
    assert device_parser._determine_root_gpu_device(devices) == expected_root_gpu

@pytest.mark.parametrize(('devices', 'expected_gpu_ids'), [(0, None), ([], None), (1, [0]), (3, [0, 1, 2]), pytest.param(-1, list(range(_PRETEND_N_OF_GPUS)), id='-1 - use all gpus'), ([0], [0]), ([1, 3], [1, 3]), ((1, 3), [1, 3]), ('0', None), ('3', [0, 1, 2]), ('1, 3', [1, 3]), ('2,', [2]), pytest.param('-1', list(range(_PRETEND_N_OF_GPUS)), id="'-1' - use all gpus")])
@mock.patch('lightning.fabric.accelerators.cuda.num_cuda_devices', return_value=_PRETEND_N_OF_GPUS)
def test_parse_gpu_ids(_, devices, expected_gpu_ids):
    if False:
        print('Hello World!')
    assert device_parser._parse_gpu_ids(devices, include_cuda=True) == expected_gpu_ids

@pytest.mark.parametrize('devices', [0.1, -2, False, [-1], [None], ['0'], [0, 0]])
@mock.patch('lightning.fabric.accelerators.cuda.num_cuda_devices', return_value=_PRETEND_N_OF_GPUS)
def test_parse_gpu_fail_on_unsupported_inputs(_, devices):
    if False:
        while True:
            i = 10
    with pytest.raises((TypeError, MisconfigurationException)):
        device_parser._parse_gpu_ids(devices, include_cuda=True)

@pytest.mark.parametrize('devices', [[1, 2, 19], -1, '-1'])
@mock.patch('lightning.fabric.accelerators.cuda.num_cuda_devices', return_value=0)
def test_parse_gpu_fail_on_non_existent_id(_, devices):
    if False:
        while True:
            i = 10
    with pytest.raises((TypeError, MisconfigurationException)):
        device_parser._parse_gpu_ids(devices, include_cuda=True)

@mock.patch('lightning.fabric.accelerators.cuda.num_cuda_devices', return_value=_PRETEND_N_OF_GPUS)
def test_parse_gpu_fail_on_non_existent_id_2(_):
    if False:
        while True:
            i = 10
    with pytest.raises((TypeError, MisconfigurationException)):
        device_parser._parse_gpu_ids([1, 2, 19], include_cuda=True)

@pytest.mark.parametrize('devices', [-1, '-1'])
@mock.patch('lightning.fabric.accelerators.cuda.num_cuda_devices', return_value=0)
def test_parse_gpu_returns_none_when_no_devices_are_available(_, devices):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(MisconfigurationException):
        device_parser._parse_gpu_ids(devices, include_cuda=True)