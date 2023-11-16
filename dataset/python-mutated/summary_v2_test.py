"""Tests for the API surface of the V1 tf.summary ops when TF2 is enabled.

V1 summary ops will invoke V2 TensorBoard summary ops in eager mode.
"""
from tensorboard.summary import v2 as summary_v2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.training import training_util

class SummaryV2Test(test.TestCase):

    @test_util.run_v2_only
    def test_scalar_summary_v2__w_writer(self):
        if False:
            return 10
        'Tests scalar v2 invocation with a v2 writer.'
        with test.mock.patch.object(summary_v2, 'scalar', autospec=True) as mock_scalar_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default(step=1):
                i = constant_op.constant(2.5)
                tensor = summary_lib.scalar('float', i)
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_scalar_v2.assert_called_once_with('float', data=i, step=1)

    @test_util.run_v2_only
    def test_scalar_summary_v2__wo_writer(self):
        if False:
            while True:
                i = 10
        'Tests scalar v2 invocation with no writer.'
        with self.assertWarnsRegex(UserWarning, 'default summary writer not found'):
            with test.mock.patch.object(summary_v2, 'scalar', autospec=True) as mock_scalar_v2:
                summary_lib.scalar('float', constant_op.constant(2.5))
        mock_scalar_v2.assert_not_called()

    @test_util.run_v2_only
    def test_scalar_summary_v2__global_step_not_set(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests scalar v2 invocation when global step is not set.'
        with self.assertWarnsRegex(UserWarning, 'global step not set'):
            with test.mock.patch.object(summary_v2, 'scalar', autospec=True) as mock_scalar_v2:
                with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default():
                    summary_lib.scalar('float', constant_op.constant(2.5))
        mock_scalar_v2.assert_not_called()

    @test_util.run_v2_only
    def test_scalar_summary_v2__family(self):
        if False:
            print('Hello World!')
        'Tests `family` arg handling when scalar v2 is invoked.'
        with test.mock.patch.object(summary_v2, 'scalar', autospec=True) as mock_scalar_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default(step=1):
                tensor = summary_lib.scalar('float', constant_op.constant(2.5), family='otter')
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_scalar_v2.assert_called_once_with('otter/otter/float', data=constant_op.constant(2.5), step=1)

    @test_util.run_v2_only
    def test_scalar_summary_v2__family_w_outer_scope(self):
        if False:
            i = 10
            return i + 15
        'Tests `family` arg handling when there is an outer scope.'
        with test.mock.patch.object(summary_v2, 'scalar', autospec=True) as mock_scalar_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default(step=1):
                with ops.name_scope_v2('sea'):
                    tensor = summary_lib.scalar('float', constant_op.constant(3.5), family='crabnet')
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_scalar_v2.assert_called_once_with('crabnet/sea/crabnet/float', data=constant_op.constant(3.5), step=1)

    @test_util.run_v2_only
    def test_scalar_summary_v2__v1_set_step(self):
        if False:
            return 10
        'Tests scalar v2 invocation when v1 step is set.'
        global_step = training_util.create_global_step()
        global_step.assign(1024)
        with test.mock.patch.object(summary_v2, 'scalar', autospec=True) as mock_scalar_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default():
                i = constant_op.constant(2.5)
                tensor = summary_lib.scalar('float', i)
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_scalar_v2.assert_called_once_with('float', data=i, step=1024)

    @test_util.run_v2_only
    def test_image_summary_v2(self):
        if False:
            i = 10
            return i + 15
        'Tests image v2 invocation.'
        with test.mock.patch.object(summary_v2, 'image', autospec=True) as mock_image_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default(step=2):
                i = array_ops.ones((5, 4, 4, 3))
                with ops.name_scope_v2('outer'):
                    tensor = summary_lib.image('image', i, max_outputs=3, family='family')
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_image_v2.assert_called_once_with('family/outer/family/image', data=i, step=2, max_outputs=3)

    @test_util.run_v2_only
    def test_histogram_summary_v2(self):
        if False:
            while True:
                i = 10
        'Tests histogram v2 invocation.'
        with test.mock.patch.object(summary_v2, 'histogram', autospec=True) as mock_histogram_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default(step=3):
                i = array_ops.ones((1024,))
                tensor = summary_lib.histogram('histogram', i, family='family')
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_histogram_v2.assert_called_once_with('family/family/histogram', data=i, step=3)

    @test_util.run_v2_only
    def test_audio_summary_v2(self):
        if False:
            i = 10
            return i + 15
        'Tests audio v2 invocation.'
        with test.mock.patch.object(summary_v2, 'audio', autospec=True) as mock_audio_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default(step=10):
                i = array_ops.ones((5, 3, 4))
                with ops.name_scope_v2('dolphin'):
                    tensor = summary_lib.audio('wave', i, 0.2, max_outputs=3)
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_audio_v2.assert_called_once_with('dolphin/wave', data=i, sample_rate=0.2, step=10, max_outputs=3)

    @test_util.run_v2_only
    def test_audio_summary_v2__2d_tensor(self):
        if False:
            while True:
                i = 10
        'Tests audio v2 invocation with 2-D tensor input.'
        with test.mock.patch.object(summary_v2, 'audio', autospec=True) as mock_audio_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default(step=11):
                input_2d = array_ops.ones((5, 3))
                tensor = summary_lib.audio('wave', input_2d, 0.2, max_outputs=3)
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_audio_v2.assert_called_once_with('wave', data=test.mock.ANY, sample_rate=0.2, step=11, max_outputs=3)
        input_3d = array_ops.ones((5, 3, 1))
        self.assertAllEqual(mock_audio_v2.call_args[1]['data'], input_3d)

    @test_util.run_v2_only
    def test_text_summary_v2(self):
        if False:
            return 10
        'Tests text v2 invocation.'
        with test.mock.patch.object(summary_v2, 'text', autospec=True) as mock_text_v2:
            with summary_ops_v2.create_summary_file_writer(self.get_temp_dir()).as_default(step=22):
                i = constant_op.constant('lorem ipsum', dtype=dtypes.string)
                tensor = summary_lib.text('text', i)
        self.assertEqual(tensor.numpy(), b'')
        self.assertEqual(tensor.dtype, dtypes.string)
        mock_text_v2.assert_called_once_with('text', data=i, step=22)
if __name__ == '__main__':
    test.main()