"""Tests for google3.third_party.tensorflow_models.object_detection.builders.target_assigner_builder."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import target_assigner_builder
from object_detection.core import target_assigner
from object_detection.protos import target_assigner_pb2

class TargetAssignerBuilderTest(tf.test.TestCase):

    def test_build_a_target_assigner(self):
        if False:
            i = 10
            return i + 15
        target_assigner_text_proto = '\n      matcher {\n        argmax_matcher {matched_threshold: 0.5}\n      }\n      similarity_calculator {\n        iou_similarity {}\n      }\n      box_coder {\n        faster_rcnn_box_coder {}\n      }\n    '
        target_assigner_proto = target_assigner_pb2.TargetAssigner()
        text_format.Merge(target_assigner_text_proto, target_assigner_proto)
        target_assigner_instance = target_assigner_builder.build(target_assigner_proto)
        self.assertIsInstance(target_assigner_instance, target_assigner.TargetAssigner)
if __name__ == '__main__':
    tf.test.main()