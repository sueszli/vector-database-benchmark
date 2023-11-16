"""Tests for loading SavedModels with optimizers."""
from tensorflow.python.eager import test
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load

class LoadOptimizerTest(test.TestCase):

    def test_load_optimizer_without_keras(self):
        if False:
            i = 10
            return i + 15
        save_path = test.test_src_dir_path('cc/saved_model/testdata/OptimizerSlotVariableModule')
        loaded = load.load(save_path)
        self.assertIsInstance(loaded.opt.get_slot(loaded.v, 'v'), variables.Variable)
if __name__ == '__main__':
    test.main()