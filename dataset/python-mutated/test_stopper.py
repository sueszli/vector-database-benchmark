import pickle
from freezegun import freeze_time
from ray.tune.stopper import TimeoutStopper

def test_timeout_stopper_timeout():
    if False:
        while True:
            i = 10
    with freeze_time() as frozen:
        stopper = TimeoutStopper(timeout=60)
        assert not stopper.stop_all()
        frozen.tick(40)
        assert not stopper.stop_all()
        frozen.tick(22)
        assert stopper.stop_all()

def test_timeout_stopper_recover_before_timeout():
    if False:
        while True:
            i = 10
    ' "If checkpointed before timeout, should continue where we left.'
    with freeze_time() as frozen:
        stopper = TimeoutStopper(timeout=60)
        assert not stopper.stop_all()
        frozen.tick(40)
        assert not stopper.stop_all()
        checkpoint = pickle.dumps(stopper)
        frozen.tick(200)
        stopper = pickle.loads(checkpoint)
        assert not stopper.stop_all()
        frozen.tick(10)
        assert not stopper.stop_all()
        frozen.tick(12)
        assert stopper.stop_all()

def test_timeout_stopper_recover_after_timeout():
    if False:
        print('Hello World!')
    ' "If checkpointed after timeout, should still stop after recover.'
    with freeze_time() as frozen:
        stopper = TimeoutStopper(timeout=60)
        assert not stopper.stop_all()
        frozen.tick(62)
        assert stopper.stop_all()
        checkpoint = pickle.dumps(stopper)
        frozen.tick(200)
        stopper = pickle.loads(checkpoint)
        assert stopper.stop_all()
        frozen.tick(10)
        assert stopper.stop_all()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__] + sys.argv[1:]))