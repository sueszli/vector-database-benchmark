import os
import shutil
import tempfile
import unittest
import warnings
import numpy
import chainer
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer import training

class Model(chainer.Chain):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(Model, self).__init__()
        with self.init_scope():
            self.l = links.Linear(1, 3)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.l(x)

class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, values):
        if False:
            return 10
        self.values = values

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.values)

    def get_example(self, i):
        if False:
            return 10
        return (numpy.array([self.values[i]], numpy.float32), numpy.int32(i % 2))

class TestFailOnNonNumber(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.n_data = 4
        self.n_epochs = 3
        self.model = Model()
        self.classifier = links.Classifier(self.model)
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.classifier)
        self.dataset = Dataset([i for i in range(self.n_data)])
        self.iterator = chainer.iterators.SerialIterator(self.dataset, 1, shuffle=False)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.temp_dir)

    def prepare(self, dirname='test', device=None):
        if False:
            i = 10
            return i + 15
        outdir = os.path.join(self.temp_dir, dirname)
        self.updater = training.updaters.StandardUpdater(self.iterator, self.optimizer, device=device)
        self.trainer = training.Trainer(self.updater, (self.n_epochs, 'epoch'), out=outdir)
        self.trainer.extend(training.extensions.FailOnNonNumber())

    def test_trainer(self):
        if False:
            for i in range(10):
                print('nop')
        self.prepare(dirname='test_trainer')
        self.trainer.run()

    def test_nan(self):
        if False:
            i = 10
            return i + 15
        self.prepare(dirname='test_nan')
        self.model.l.W.array[1, 0] = numpy.nan
        with self.assertRaises(RuntimeError):
            self.trainer.run(show_loop_exception_msg=False)

    def test_inf(self):
        if False:
            print('Hello World!')
        self.prepare(dirname='test_inf')
        self.model.l.W.array[2, 0] = numpy.inf
        with warnings.catch_warnings(), self.assertRaises(RuntimeError):
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            self.trainer.run(show_loop_exception_msg=False)

    @attr.gpu
    def test_trainer_gpu(self):
        if False:
            print('Hello World!')
        self.prepare(dirname='test_trainer_gpu', device=0)
        self.trainer.run()

    @attr.gpu
    def test_nan_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.prepare(dirname='test_nan_gpu', device=0)
        self.model.l.W.array[:] = numpy.nan
        with self.assertRaises(RuntimeError):
            self.trainer.run(show_loop_exception_msg=False)

    @attr.gpu
    def test_inf_gpu(self):
        if False:
            print('Hello World!')
        self.prepare(dirname='test_inf_gpu', device=0)
        self.model.l.W.array[:] = numpy.inf
        with self.assertRaises(RuntimeError):
            self.trainer.run(show_loop_exception_msg=False)
testing.run_module(__name__, __file__)