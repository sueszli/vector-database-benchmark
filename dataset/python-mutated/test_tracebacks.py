import pytest
import ray
from ray.train import ScalingConfig
from ray.air._internal.util import StartTraceback, skip_exceptions
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.tune import Tuner

@pytest.fixture
def ray_start_2_cpus():
    if False:
        i = 10
        return i + 15
    address_info = ray.init(num_cpus=2)
    yield address_info
    ray.shutdown()

def _failing_recursive(levels: int=0, start_traceback: int=-1):
    if False:
        i = 10
        return i + 15
    if levels > 0:
        if start_traceback == 0:
            try:
                _failing_recursive(levels=levels - 1, start_traceback=start_traceback - 1)
            except Exception as e:
                raise StartTraceback from e
        else:
            _failing_recursive(levels=levels - 1, start_traceback=start_traceback - 1)
    else:
        raise RuntimeError('Failing')

@pytest.mark.parametrize('levels', [4, 5, 6, 7, 8, 9, 10])
def test_short_traceback(levels):
    if False:
        while True:
            i = 10
    start_traceback = 3
    with pytest.raises(StartTraceback) as exc_info:
        _failing_recursive(levels=levels, start_traceback=start_traceback)
    exc = skip_exceptions(exc_info.value)
    tb = exc.__traceback__
    i = 0
    while tb:
        i += 1
        tb = tb.tb_next
    assert i == levels - start_traceback + 1

def test_traceback_tuner(ray_start_2_cpus):
    if False:
        while True:
            i = 10
    "Ensure that the Tuner's stack trace is not too long."

    def failing(config):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('Error')
    tuner = Tuner(failing)
    results = tuner.fit()
    assert len(str(results[0].error).split('\n')) <= 12

def test_traceback_trainer(ray_start_2_cpus):
    if False:
        while True:
            i = 10
    "Ensure that the Trainer's stack trace is not too long."

    def failing(config):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('Error')
    trainer = DataParallelTrainer(failing, scaling_config=ScalingConfig(num_workers=1))
    with pytest.raises(RuntimeError) as exc_info:
        trainer.fit()
    assert len(str(exc_info.value).split('\n')) <= 13
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-x', __file__]))