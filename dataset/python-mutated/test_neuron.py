import os
import sys
import subprocess
import pytest
from unittest.mock import patch
import ray
from ray._private.accelerators import NeuronAcceleratorManager

def test_user_configured_more_than_visible(monkeypatch, call_ray_stop_only):
    if False:
        while True:
            i = 10
    monkeypatch.setenv('NEURON_RT_VISIBLE_CORES', '0,1,2')
    with pytest.raises(ValueError):
        ray.init(resources={'neuron_cores': 4})

@patch('ray._private.accelerators.NeuronAcceleratorManager.get_current_node_num_accelerators', return_value=4)
def test_auto_detected_more_than_visible(mock_get_num_accelerators, monkeypatch, shutdown_only):
    if False:
        i = 10
        return i + 15
    monkeypatch.setenv('NEURON_RT_VISIBLE_CORES', '0,1,2')
    ray.init()
    mock_get_num_accelerators.called
    assert ray.available_resources()['neuron_cores'] == 3

@patch('ray._private.accelerators.NeuronAcceleratorManager.get_current_node_num_accelerators', return_value=2)
def test_auto_detect_resources(mock_get_num_accelerators, shutdown_only):
    if False:
        print('Hello World!')
    ray.init()
    mock_get_num_accelerators.called
    assert ray.available_resources()['neuron_cores'] == 2

@patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b'[{"neuron_device":0,"bdf":"00:1e.0","connected_to":null,"nc_count":2,"memory_size":34359738368,"neuron_processes":[]}]'))
@patch('os.path.isdir', return_value=True)
@patch('sys.platform', 'linux')
def test_get_neuron_core_count_single_device(mock_isdir, mock_subprocess):
    if False:
        i = 10
        return i + 15
    assert NeuronAcceleratorManager.get_current_node_num_accelerators() == 2

@patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b'[{"neuron_device":0,"bdf":"00:1e.0","connected_to":null,"nc_count":2,"memory_size":34359738368,"neuron_processes":[]},{"neuron_device":1,"bdf":"00:1f.0","connected_to":null,"nc_count":2,"memory_size":34359738368,"neuron_processes":[]}]'))
@patch('os.path.isdir', return_value=True)
@patch('sys.platform', 'linux')
def test_get_neuron_core_count_multiple_devices(mock_isdir, mock_subprocess):
    if False:
        i = 10
        return i + 15
    assert NeuronAcceleratorManager.get_current_node_num_accelerators() == 4

@patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=b'AccessDenied'))
@patch('os.path.isdir', return_value=True)
@patch('sys.platform', 'linux')
def test_get_neuron_core_count_failure_with_error(mock_isdir, mock_subprocess):
    if False:
        while True:
            i = 10
    assert NeuronAcceleratorManager.get_current_node_num_accelerators() == 0

@patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b'[{}]'))
@patch('os.path.isdir', return_value=True)
@patch('sys.platform', 'linux')
def test_get_neuron_core_count_failure_with_empty_results(mock_isdir, mock_subprocess):
    if False:
        for i in range(10):
            print('nop')
    assert NeuronAcceleratorManager.get_current_node_num_accelerators() == 0
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))