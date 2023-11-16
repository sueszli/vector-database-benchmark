"""Tests for object_detection.predictors.heads.keypoint_head."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import keypoint_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case

class MaskRCNNKeypointHeadTest(test_case.TestCase):

    def _build_arg_scope_with_hyperparams(self, op_type=hyperparams_pb2.Hyperparams.FC):
        if False:
            for i in range(10):
                print('nop')
        hyperparams = hyperparams_pb2.Hyperparams()
        hyperparams_text_proto = '\n      activation: NONE\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(hyperparams_text_proto, hyperparams)
        hyperparams.op = op_type
        return hyperparams_builder.build(hyperparams, is_training=True)

    def test_prediction_size(self):
        if False:
            return 10
        keypoint_prediction_head = keypoint_head.MaskRCNNKeypointHead(conv_hyperparams_fn=self._build_arg_scope_with_hyperparams())
        roi_pooled_features = tf.random_uniform([64, 14, 14, 1024], minval=-2.0, maxval=2.0, dtype=tf.float32)
        prediction = keypoint_prediction_head.predict(features=roi_pooled_features, num_predictions_per_location=1)
        self.assertAllEqual([64, 1, 17, 56, 56], prediction.get_shape().as_list())
if __name__ == '__main__':
    tf.test.main()