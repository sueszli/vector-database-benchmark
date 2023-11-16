"""Tests for object_detection.predictors.rfcn_box_predictor."""
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors import rfcn_box_predictor as box_predictor
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case

class RfcnBoxPredictorTest(test_case.TestCase):

    def _build_arg_scope_with_conv_hyperparams(self):
        if False:
            for i in range(10):
                print('nop')
        conv_hyperparams = hyperparams_pb2.Hyperparams()
        conv_hyperparams_text_proto = '\n      regularizer {\n        l2_regularizer {\n        }\n      }\n      initializer {\n        truncated_normal_initializer {\n        }\n      }\n    '
        text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
        return hyperparams_builder.build(conv_hyperparams, is_training=True)

    def test_get_correct_box_encoding_and_class_prediction_shapes(self):
        if False:
            i = 10
            return i + 15

        def graph_fn(image_features, proposal_boxes):
            if False:
                for i in range(10):
                    print('nop')
            rfcn_box_predictor = box_predictor.RfcnBoxPredictor(is_training=False, num_classes=2, conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(), num_spatial_bins=[3, 3], depth=4, crop_size=[12, 12], box_code_size=4)
            box_predictions = rfcn_box_predictor.predict([image_features], num_predictions_per_location=[1], scope='BoxPredictor', proposal_boxes=proposal_boxes)
            box_encodings = tf.concat(box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
            class_predictions_with_background = tf.concat(box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND], axis=1)
            return (box_encodings, class_predictions_with_background)
        image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
        proposal_boxes = np.random.rand(4, 2, 4).astype(np.float32)
        (box_encodings, class_predictions_with_background) = self.execute(graph_fn, [image_features, proposal_boxes])
        self.assertAllEqual(box_encodings.shape, [8, 1, 2, 4])
        self.assertAllEqual(class_predictions_with_background.shape, [8, 1, 3])
if __name__ == '__main__':
    tf.test.main()