"""A simple smoke test that runs these examples for 1 training iteraton."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd
from six.moves import StringIO
import iris_data
import custom_estimator
import premade_estimator
FOUR_LINES = '\n'.join(['1,52.40, 2823,152,2', '164, 99.80,176.60,66.20,1', '176,2824, 136,3.19,0', '2,177.30,66.30, 53.10,1'])

def four_lines_data():
    if False:
        i = 10
        return i + 15
    text = StringIO(FOUR_LINES)
    df = pd.read_csv(text, names=iris_data.CSV_COLUMN_NAMES)
    xy = (df, df.pop('Species'))
    return (xy, xy)

class RegressionTest(tf.test.TestCase):
    """Test the regression examples in this directory."""

    @tf.test.mock.patch.dict(premade_estimator.__dict__, {'load_data': four_lines_data})
    def test_premade_estimator(self):
        if False:
            while True:
                i = 10
        premade_estimator.main([None, '--train_steps=1'])

    @tf.test.mock.patch.dict(custom_estimator.__dict__, {'load_data': four_lines_data})
    def test_custom_estimator(self):
        if False:
            i = 10
            return i + 15
        custom_estimator.main([None, '--train_steps=1'])
if __name__ == '__main__':
    tf.test.main()