"""Tests for training_util."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training_util

class GlobalStepTest(test.TestCase):

    def _assert_global_step(self, global_step, expected_dtype=dtypes.int64):
        if False:
            print('Hello World!')
        self.assertEqual('%s:0' % ops.GraphKeys.GLOBAL_STEP, global_step.name)
        self.assertEqual(expected_dtype, global_step.dtype.base_dtype)
        self.assertEqual([], global_step.get_shape().as_list())

    def test_invalid_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default() as g:
            self.assertIsNone(training_util.get_global_step())
            variable_v1.VariableV1(0.0, trainable=False, dtype=dtypes.float32, name=ops.GraphKeys.GLOBAL_STEP, collections=[ops.GraphKeys.GLOBAL_STEP])
            self.assertRaisesRegex(TypeError, 'does not have integer type', training_util.get_global_step)
        self.assertRaisesRegex(TypeError, 'does not have integer type', training_util.get_global_step, g)

    def test_invalid_shape(self):
        if False:
            return 10
        with ops.Graph().as_default() as g:
            self.assertIsNone(training_util.get_global_step())
            variable_v1.VariableV1([0], trainable=False, dtype=dtypes.int32, name=ops.GraphKeys.GLOBAL_STEP, collections=[ops.GraphKeys.GLOBAL_STEP])
            self.assertRaisesRegex(TypeError, 'not scalar', training_util.get_global_step)
        self.assertRaisesRegex(TypeError, 'not scalar', training_util.get_global_step, g)

    def test_create_global_step(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(training_util.get_global_step())
        with ops.Graph().as_default() as g:
            global_step = training_util.create_global_step()
            self._assert_global_step(global_step)
            self.assertRaisesRegex(ValueError, 'already exists', training_util.create_global_step)
            self.assertRaisesRegex(ValueError, 'already exists', training_util.create_global_step, g)
            self._assert_global_step(training_util.create_global_step(ops.Graph()))

    def test_get_global_step(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default() as g:
            self.assertIsNone(training_util.get_global_step())
            variable_v1.VariableV1(0, trainable=False, dtype=dtypes.int32, name=ops.GraphKeys.GLOBAL_STEP, collections=[ops.GraphKeys.GLOBAL_STEP])
            self._assert_global_step(training_util.get_global_step(), expected_dtype=dtypes.int32)
        self._assert_global_step(training_util.get_global_step(g), expected_dtype=dtypes.int32)

    def test_get_or_create_global_step(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default() as g:
            self.assertIsNone(training_util.get_global_step())
            self._assert_global_step(training_util.get_or_create_global_step())
            self._assert_global_step(training_util.get_or_create_global_step(g))

class GlobalStepReadTest(test.TestCase):

    def test_global_step_read_is_none_if_there_is_no_global_step(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            self.assertIsNone(training_util._get_or_create_global_step_read())
            training_util.create_global_step()
            self.assertIsNotNone(training_util._get_or_create_global_step_read())

    def test_reads_from_cache(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            training_util.create_global_step()
            first = training_util._get_or_create_global_step_read()
            second = training_util._get_or_create_global_step_read()
            self.assertEqual(first, second)

    def test_reads_before_increments(self):
        if False:
            return 10
        with ops.Graph().as_default():
            training_util.create_global_step()
            read_tensor = training_util._get_or_create_global_step_read()
            inc_op = training_util._increment_global_step(1)
            inc_three_op = training_util._increment_global_step(3)
            with monitored_session.MonitoredTrainingSession() as sess:
                (read_value, _) = sess.run([read_tensor, inc_op])
                self.assertEqual(0, read_value)
                (read_value, _) = sess.run([read_tensor, inc_three_op])
                self.assertEqual(1, read_value)
                read_value = sess.run(read_tensor)
                self.assertEqual(4, read_value)
if __name__ == '__main__':
    test.main()