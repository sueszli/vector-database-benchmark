import pytest
import sys
from ray import tune

def test_checkpoint_dir_deprecation():
    if False:
        for i in range(10):
            print('nop')

    def train_fn(config, checkpoint_dir=None):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DeprecationWarning, match='.*checkpoint_dir.*'):
        tune.run(train_fn)

    def train_fn(config, reporter):
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(DeprecationWarning, match='.*reporter.*'):
        tune.run(train_fn)

    def train_fn(config):
        if False:
            for i in range(10):
                print('nop')
        tune.report(test=1)
    with pytest.raises(DeprecationWarning, match='.*tune\\.report.*'):
        tune.run(train_fn, fail_fast='raise')

    def train_fn(config):
        if False:
            for i in range(10):
                print('nop')
        with tune.checkpoint_dir(step=1) as _:
            pass
    with pytest.raises(DeprecationWarning, match='.*tune\\.checkpoint_dir.*'):
        tune.run(train_fn, fail_fast='raise')
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))