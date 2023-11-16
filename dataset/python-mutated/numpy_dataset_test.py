"""Tests for numpy_dataset."""
import numpy as np
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variable_v1

class InitVarFromNumpyTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_creating_var_with_numpy_arrays(self):
        if False:
            return 10
        with self.cached_session() as session:
            x = np.asarray(np.random.random((64, 3)), dtype=np.float32)
            initial = np.zeros_like(x)
            var_x = variable_v1.VariableV1(initial)
            numpy_dataset.init_var_from_numpy(var_x, x, session)
            val = self.evaluate(var_x.value())
            self.assertAllEqual(x, val)
if __name__ == '__main__':
    test.main()