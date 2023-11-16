import io
import unittest
import chainer
from chainer import testing

class DummyTrainer(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.elapsed_time = 0

class TestTimeTrigger(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.trigger = chainer.training.triggers.TimeTrigger(1)
        self.trainer = DummyTrainer()

    def test_call(self):
        if False:
            while True:
                i = 10
        assert not self.trigger(self.trainer)
        self.trainer.elapsed_time = 0.9
        assert not self.trigger(self.trainer)
        self.trainer.elapsed_time = 1.2
        assert self.trigger(self.trainer)
        self.trainer.elapsed_time = 1.3
        assert not self.trigger(self.trainer)
        self.trainer.elapsed_time = 2.1
        assert self.trigger(self.trainer)

    def test_resume(self):
        if False:
            while True:
                i = 10
        self.trainer.elapsed_time = 1.2
        self.trigger(self.trainer)
        assert self.trigger._next_time == 2.0
        f = io.BytesIO()
        chainer.serializers.save_npz(f, self.trigger)
        trigger = chainer.training.triggers.TimeTrigger(1)
        chainer.serializers.load_npz(io.BytesIO(f.getvalue()), trigger)
        assert trigger._next_time == 2.0
testing.run_module(__name__, __file__)