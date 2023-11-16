"""Tests for calibration_builder."""
import numpy as np
from scipy import interpolate
import tensorflow as tf
from object_detection.builders import calibration_builder
from object_detection.protos import calibration_pb2

class CalibrationBuilderTest(tf.test.TestCase):

    def test_tf_linear_interp1d_map(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests TF linear interpolation mapping to a single number.'
        with self.test_session() as sess:
            tf_x = tf.constant([0.0, 0.5, 1.0])
            tf_y = tf.constant([0.5, 0.5, 0.5])
            new_x = tf.constant([0.0, 0.25, 0.5, 0.75, 1.0])
            tf_map_outputs = calibration_builder._tf_linear_interp1d(new_x, tf_x, tf_y)
            tf_map_outputs_np = sess.run([tf_map_outputs])
        self.assertAllClose(tf_map_outputs_np, [[0.5, 0.5, 0.5, 0.5, 0.5]])

    def test_tf_linear_interp1d_interpolate(self):
        if False:
            while True:
                i = 10
        'Tests TF 1d linear interpolation not mapping to a single number.'
        with self.test_session() as sess:
            tf_x = tf.constant([0.0, 0.5, 1.0])
            tf_y = tf.constant([0.6, 0.7, 1.0])
            new_x = tf.constant([0.0, 0.25, 0.5, 0.75, 1.0])
            tf_interpolate_outputs = calibration_builder._tf_linear_interp1d(new_x, tf_x, tf_y)
            tf_interpolate_outputs_np = sess.run([tf_interpolate_outputs])
        self.assertAllClose(tf_interpolate_outputs_np, [[0.6, 0.65, 0.7, 0.85, 1.0]])

    @staticmethod
    def _get_scipy_interp1d(new_x, x, y):
        if False:
            return 10
        'Helper performing 1d linear interpolation using SciPy.'
        interpolation1d_fn = interpolate.interp1d(x, y)
        return interpolation1d_fn(new_x)

    def _get_tf_interp1d(self, new_x, x, y):
        if False:
            while True:
                i = 10
        'Helper performing 1d linear interpolation using Tensorflow.'
        with self.test_session() as sess:
            tf_interp_outputs = calibration_builder._tf_linear_interp1d(tf.convert_to_tensor(new_x, dtype=tf.float32), tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32))
            np_tf_interp_outputs = sess.run(tf_interp_outputs)
        return np_tf_interp_outputs

    def test_tf_linear_interp1d_against_scipy_map(self):
        if False:
            i = 10
            return i + 15
        'Tests parity of TF linear interpolation with SciPy for simple mapping.'
        length = 10
        np_x = np.linspace(0, 1, length)
        np_y_map = np.repeat(0.5, length)
        test_data_np = np.linspace(0, 1, length * 10)
        scipy_map_outputs = self._get_scipy_interp1d(test_data_np, np_x, np_y_map)
        np_tf_map_outputs = self._get_tf_interp1d(test_data_np, np_x, np_y_map)
        self.assertAllClose(scipy_map_outputs, np_tf_map_outputs)

    def test_tf_linear_interp1d_against_scipy_interpolate(self):
        if False:
            return 10
        'Tests parity of TF linear interpolation with SciPy.'
        length = 10
        np_x = np.linspace(0, 1, length)
        np_y_interp = np.linspace(0.5, 1, length)
        test_data_np = np.linspace(0, 1, length * 10)
        scipy_interp_outputs = self._get_scipy_interp1d(test_data_np, np_x, np_y_interp)
        np_tf_interp_outputs = self._get_tf_interp1d(test_data_np, np_x, np_y_interp)
        self.assertAllClose(scipy_interp_outputs, np_tf_interp_outputs)

    @staticmethod
    def _add_function_approximation_to_calibration_proto(calibration_proto, x_array, y_array, class_id):
        if False:
            print('Hello World!')
        'Adds a function approximation to calibration proto for a class id.'
        if class_id is not None:
            function_approximation = calibration_proto.class_id_function_approximations.class_id_xy_pairs_map[class_id]
        else:
            function_approximation = calibration_proto.function_approximation.x_y_pairs
        for (x, y) in zip(x_array, y_array):
            x_y_pair_message = function_approximation.x_y_pair.add()
            x_y_pair_message.x = x
            x_y_pair_message.y = y

    def test_class_agnostic_function_approximation(self):
        if False:
            print('Hello World!')
        'Tests that calibration produces correct class-agnostic values.'
        class_agnostic_x = np.asarray([0.0, 0.5, 1.0])
        class_agnostic_y = np.asarray([0.0, 0.25, 0.75])
        calibration_config = calibration_pb2.CalibrationConfig()
        self._add_function_approximation_to_calibration_proto(calibration_config, class_agnostic_x, class_agnostic_y, class_id=None)
        od_graph = tf.Graph()
        with self.test_session(graph=od_graph) as sess:
            calibration_fn = calibration_builder.build(calibration_config)
            class_predictions_with_background = tf.constant([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]], [[0.6, 0.7, 0.8], [0.9, 1.0, 1.0]]], dtype=tf.float32)
            calibrated_scores = calibration_fn(class_predictions_with_background)
            calibrated_scores_np = sess.run(calibrated_scores)
        self.assertAllClose(calibrated_scores_np, [[[0.05, 0.1, 0.15], [0.2, 0.25, 0.0]], [[0.35, 0.45, 0.55], [0.65, 0.75, 0.75]]])

    def test_multiclass_function_approximations(self):
        if False:
            i = 10
            return i + 15
        'Tests that calibration produces correct multiclass values.'
        class_0_x = np.asarray([0.0, 0.5, 1.0])
        class_0_y = np.asarray([0.5, 0.5, 0.5])
        calibration_config = calibration_pb2.CalibrationConfig()
        self._add_function_approximation_to_calibration_proto(calibration_config, class_0_x, class_0_y, class_id=0)
        class_1_x = np.asarray([0.0, 0.2, 1.0])
        class_1_y = np.asarray([0.0, 0.6, 1.0])
        self._add_function_approximation_to_calibration_proto(calibration_config, class_1_x, class_1_y, class_id=1)
        od_graph = tf.Graph()
        with self.test_session(graph=od_graph) as sess:
            calibration_fn = calibration_builder.build(calibration_config)
            class_predictions_with_background = tf.constant([[[0.1, 0.2], [0.9, 0.1]], [[0.6, 0.4], [0.08, 0.92]]], dtype=tf.float32)
            calibrated_scores = calibration_fn(class_predictions_with_background)
            calibrated_scores_np = sess.run(calibrated_scores)
        self.assertAllClose(calibrated_scores_np, [[[0.5, 0.6], [0.5, 0.3]], [[0.5, 0.7], [0.5, 0.96]]])

    def test_temperature_scaling(self):
        if False:
            while True:
                i = 10
        'Tests that calibration produces correct temperature scaling values.'
        calibration_config = calibration_pb2.CalibrationConfig()
        calibration_config.temperature_scaling_calibration.scaler = 2.0
        od_graph = tf.Graph()
        with self.test_session(graph=od_graph) as sess:
            calibration_fn = calibration_builder.build(calibration_config)
            class_predictions_with_background = tf.constant([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]], [[0.6, 0.7, 0.8], [0.9, 1.0, 1.0]]], dtype=tf.float32)
            calibrated_scores = calibration_fn(class_predictions_with_background)
            calibrated_scores_np = sess.run(calibrated_scores)
        self.assertAllClose(calibrated_scores_np, [[[0.05, 0.1, 0.15], [0.2, 0.25, 0.0]], [[0.3, 0.35, 0.4], [0.45, 0.5, 0.5]]])

    def test_temperature_scaling_incorrect_value_error(self):
        if False:
            print('Hello World!')
        calibration_config = calibration_pb2.CalibrationConfig()
        calibration_config.temperature_scaling_calibration.scaler = 0
        calibration_fn = calibration_builder.build(calibration_config)
        class_predictions_with_background = tf.constant([[[0.1, 0.2, 0.3]]], dtype=tf.float32)
        with self.assertRaises(ValueError):
            calibration_fn(class_predictions_with_background)

    def test_skips_class_when_calibration_parameters_not_present(self):
        if False:
            while True:
                i = 10
        'Tests that graph fails when parameters not present for all classes.'
        class_0_x = np.asarray([0.0, 0.5, 1.0])
        class_0_y = np.asarray([0.5, 0.5, 0.5])
        calibration_config = calibration_pb2.CalibrationConfig()
        self._add_function_approximation_to_calibration_proto(calibration_config, class_0_x, class_0_y, class_id=0)
        od_graph = tf.Graph()
        with self.test_session(graph=od_graph) as sess:
            calibration_fn = calibration_builder.build(calibration_config)
            class_predictions_with_background = tf.constant([[[0.1, 0.2], [0.9, 0.1]], [[0.6, 0.4], [0.08, 0.92]]], dtype=tf.float32)
            calibrated_scores = calibration_fn(class_predictions_with_background)
            calibrated_scores_np = sess.run(calibrated_scores)
        self.assertAllClose(calibrated_scores_np, [[[0.5, 0.2], [0.5, 0.1]], [[0.5, 0.4], [0.5, 0.92]]])
if __name__ == '__main__':
    tf.test.main()