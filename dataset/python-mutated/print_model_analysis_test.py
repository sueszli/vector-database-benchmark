"""print_model_analysis test."""
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
TEST_OPTIONS = {'max_depth': 10000, 'min_bytes': 0, 'min_micros': 0, 'min_params': 0, 'min_float_ops': 0, 'order_by': 'name', 'account_type_regexes': ['.*'], 'start_name_regexes': ['.*'], 'trim_name_regexes': [], 'show_name_regexes': ['.*'], 'hide_name_regexes': [], 'account_displayed_op_only': True, 'select': ['params'], 'output': 'stdout'}

class PrintModelAnalysisTest(test.TestCase):

    def _BuildSmallModel(self):
        if False:
            while True:
                i = 10
        image = array_ops.zeros([2, 6, 6, 3])
        kernel = variable_scope.get_variable('DW', [6, 6, 3, 6], dtypes.float32, initializer=init_ops.random_normal_initializer(stddev=0.001))
        x = nn_ops.conv2d(image, kernel, [1, 2, 2, 1], padding='SAME')
        return x
if __name__ == '__main__':
    test.main()