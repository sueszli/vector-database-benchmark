"""Tests for base_delegate."""
import os
from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import base
from tensorflow.python.trackable import base_delegate

class Inner(base.Trackable):

    def __init__(self, v):
        if False:
            return 10
        self.v = v
        self._track_trackable(v, 'v')

    def _copy_trackable_to_cpu(self, object_map):
        if False:
            for i in range(10):
                print('nop')
        if self not in object_map:
            object_map[self] = Inner(self.v)
        self.v._copy_trackable_to_cpu(object_map)

class Wrapper(base_delegate.DelegatingTrackableMixin, base.Trackable):

    def __init__(self, inner):
        if False:
            print('Hello World!')
        self.inner = inner
        super(Wrapper, self).__init__(inner)

    @property
    def v(self):
        if False:
            print('Hello World!')
        return self.inner.v

@test_util.run_all_in_graph_and_eager_modes
class BaseDelegateTest(parameterized.TestCase, test.TestCase):

    @parameterized.named_parameters(('_enable_async_ckpt', True), ('_disable_async_ckpt', False))
    def test_checkpoint(self, enable_async_ckpt):
        if False:
            while True:
                i = 10
        a = Wrapper(Inner(variables_lib.Variable(15.0)))
        b = Wrapper(Inner(variables_lib.Variable(-15.0)))
        self.evaluate([a.v.initializer, b.v.initializer])
        test_dir = self.get_temp_dir()
        prefix = os.path.join(test_dir, 'ckpt')
        ckpt = util.Checkpoint(a=a, b=b)
        ckpt_options = checkpoint_options.CheckpointOptions(experimental_enable_async_checkpoint=enable_async_ckpt)
        prefix_tensor = ckpt.save(prefix, options=ckpt_options)
        self.assertEqual([15, -15], self.evaluate([a.v, b.v]))
        self.evaluate(a.v.assign(-3))
        self.evaluate(b.v.assign(12))
        self.assertEqual([-3, 12], self.evaluate([a.v, b.v]))
        ckpt2 = util.Checkpoint(a=a.inner, b=b.inner)
        if enable_async_ckpt:
            ckpt.sync()
        ckpt2.restore(prefix_tensor).assert_consumed().run_restore_ops()
        self.assertEqual([15, -15], self.evaluate([a.v, b.v]))

    def test_saved_model(self):
        if False:
            i = 10
            return i + 15
        a = Wrapper(Inner(variables_lib.Variable(-15.0)))
        self.evaluate([a.v.initializer])
        self.assertEqual([-15], self.evaluate([a.v]))
        test_dir = self.get_temp_dir()
        saved_model_path = os.path.join(test_dir, 'saved_model')
        save.save(a, saved_model_path)
        loaded = load.load(saved_model_path)
        self.evaluate([loaded.v.initializer])
        self.assertEqual([-15], self.evaluate([loaded.v]))
if __name__ == '__main__':
    test.main()