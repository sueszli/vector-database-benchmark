import subprocess
import pytest
from .test_common import _assert_eq

def test_mpi_adam():
    if False:
        while True:
            i = 10
    'Test RunningMeanStd object for MPI'
    pytest.skip()
    return_code = subprocess.call(['mpirun', '--allow-run-as-root', '-np', '2', 'python', '-m', 'stable_baselines.common.mpi_adam'])
    _assert_eq(return_code, 0)

def test_mpi_adam_ppo1():
    if False:
        return 10
    'Running test for ppo1'
    pytest.skip()
    return_code = subprocess.call(['mpirun', '--allow-run-as-root', '-np', '2', 'python', '-m', 'stable_baselines.ppo1.experiments.train_cartpole'])
    _assert_eq(return_code, 0)