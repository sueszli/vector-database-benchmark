"""Tests for losses_builder."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import losses_builder
from object_detection.core import losses
from object_detection.protos import losses_pb2
from object_detection.utils import ops

class LocalizationLossBuilderTest(tf.test.TestCase):

    def test_build_weighted_l2_localization_loss(self):
        if False:
            for i in range(10):
                print('nop')
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, localization_loss, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(localization_loss, losses.WeightedL2LocalizationLoss))

    def test_build_weighted_smooth_l1_localization_loss_default_delta(self):
        if False:
            i = 10
            return i + 15
        losses_text_proto = '\n      localization_loss {\n        weighted_smooth_l1 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, localization_loss, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(localization_loss, losses.WeightedSmoothL1LocalizationLoss))
        self.assertAlmostEqual(localization_loss._delta, 1.0)

    def test_build_weighted_smooth_l1_localization_loss_non_default_delta(self):
        if False:
            print('Hello World!')
        losses_text_proto = '\n      localization_loss {\n        weighted_smooth_l1 {\n          delta: 0.1\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, localization_loss, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(localization_loss, losses.WeightedSmoothL1LocalizationLoss))
        self.assertAlmostEqual(localization_loss._delta, 0.1)

    def test_build_weighted_iou_localization_loss(self):
        if False:
            while True:
                i = 10
        losses_text_proto = '\n      localization_loss {\n        weighted_iou {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, localization_loss, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(localization_loss, losses.WeightedIOULocalizationLoss))

    def test_anchorwise_output(self):
        if False:
            i = 10
            return i + 15
        losses_text_proto = '\n      localization_loss {\n        weighted_smooth_l1 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, localization_loss, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(localization_loss, losses.WeightedSmoothL1LocalizationLoss))
        predictions = tf.constant([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]])
        targets = tf.constant([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]])
        weights = tf.constant([[1.0, 1.0]])
        loss = localization_loss(predictions, targets, weights=weights)
        self.assertEqual(loss.shape, [1, 2])

    def test_raise_error_on_empty_localization_config(self):
        if False:
            while True:
                i = 10
        losses_text_proto = '\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        with self.assertRaises(ValueError):
            losses_builder._build_localization_loss(losses_proto)

class ClassificationLossBuilderTest(tf.test.TestCase):

    def test_build_weighted_sigmoid_classification_loss(self):
        if False:
            i = 10
            return i + 15
        losses_text_proto = '\n      classification_loss {\n        weighted_sigmoid {\n        }\n      }\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, _, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSigmoidClassificationLoss))

    def test_build_weighted_sigmoid_focal_classification_loss(self):
        if False:
            while True:
                i = 10
        losses_text_proto = '\n      classification_loss {\n        weighted_sigmoid_focal {\n        }\n      }\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, _, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.SigmoidFocalClassificationLoss))
        self.assertAlmostEqual(classification_loss._alpha, None)
        self.assertAlmostEqual(classification_loss._gamma, 2.0)

    def test_build_weighted_sigmoid_focal_loss_non_default(self):
        if False:
            while True:
                i = 10
        losses_text_proto = '\n      classification_loss {\n        weighted_sigmoid_focal {\n          alpha: 0.25\n          gamma: 3.0\n        }\n      }\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, _, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.SigmoidFocalClassificationLoss))
        self.assertAlmostEqual(classification_loss._alpha, 0.25)
        self.assertAlmostEqual(classification_loss._gamma, 3.0)

    def test_build_weighted_softmax_classification_loss(self):
        if False:
            print('Hello World!')
        losses_text_proto = '\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, _, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationLoss))

    def test_build_weighted_logits_softmax_classification_loss(self):
        if False:
            print('Hello World!')
        losses_text_proto = '\n      classification_loss {\n        weighted_logits_softmax {\n        }\n      }\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, _, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationAgainstLogitsLoss))

    def test_build_weighted_softmax_classification_loss_with_logit_scale(self):
        if False:
            i = 10
            return i + 15
        losses_text_proto = '\n      classification_loss {\n        weighted_softmax {\n          logit_scale: 2.0\n        }\n      }\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, _, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationLoss))

    def test_build_bootstrapped_sigmoid_classification_loss(self):
        if False:
            print('Hello World!')
        losses_text_proto = '\n      classification_loss {\n        bootstrapped_sigmoid {\n          alpha: 0.5\n        }\n      }\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, _, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.BootstrappedSigmoidClassificationLoss))

    def test_anchorwise_output(self):
        if False:
            for i in range(10):
                print('nop')
        losses_text_proto = '\n      classification_loss {\n        weighted_sigmoid {\n          anchorwise_output: true\n        }\n      }\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, _, _, _, _, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSigmoidClassificationLoss))
        predictions = tf.constant([[[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]]])
        targets = tf.constant([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        weights = tf.constant([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
        loss = classification_loss(predictions, targets, weights=weights)
        self.assertEqual(loss.shape, [1, 2, 3])

    def test_raise_error_on_empty_config(self):
        if False:
            while True:
                i = 10
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        with self.assertRaises(ValueError):
            losses_builder.build(losses_proto)

class HardExampleMinerBuilderTest(tf.test.TestCase):

    def test_do_not_build_hard_example_miner_by_default(self):
        if False:
            for i in range(10):
                print('nop')
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, _, _, _, hard_example_miner, _, _) = losses_builder.build(losses_proto)
        self.assertEqual(hard_example_miner, None)

    def test_build_hard_example_miner_for_classification_loss(self):
        if False:
            for i in range(10):
                print('nop')
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n      hard_example_miner {\n        loss_type: CLASSIFICATION\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, _, _, _, hard_example_miner, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
        self.assertEqual(hard_example_miner._loss_type, 'cls')

    def test_build_hard_example_miner_for_localization_loss(self):
        if False:
            for i in range(10):
                print('nop')
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n      hard_example_miner {\n        loss_type: LOCALIZATION\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, _, _, _, hard_example_miner, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
        self.assertEqual(hard_example_miner._loss_type, 'loc')

    def test_build_hard_example_miner_with_non_default_values(self):
        if False:
            for i in range(10):
                print('nop')
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n      hard_example_miner {\n        num_hard_examples: 32\n        iou_threshold: 0.5\n        loss_type: LOCALIZATION\n        max_negatives_per_positive: 10\n        min_negatives_per_image: 3\n      }\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (_, _, _, _, hard_example_miner, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
        self.assertEqual(hard_example_miner._num_hard_examples, 32)
        self.assertAlmostEqual(hard_example_miner._iou_threshold, 0.5)
        self.assertEqual(hard_example_miner._max_negatives_per_positive, 10)
        self.assertEqual(hard_example_miner._min_negatives_per_image, 3)

class LossBuilderTest(tf.test.TestCase):

    def test_build_all_loss_parameters(self):
        if False:
            return 10
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n      hard_example_miner {\n      }\n      classification_weight: 0.8\n      localization_weight: 0.2\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, localization_loss, classification_weight, localization_weight, hard_example_miner, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationLoss))
        self.assertTrue(isinstance(localization_loss, losses.WeightedL2LocalizationLoss))
        self.assertAlmostEqual(classification_weight, 0.8)
        self.assertAlmostEqual(localization_weight, 0.2)

    def test_build_expected_sampling(self):
        if False:
            print('Hello World!')
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n      hard_example_miner {\n      }\n      classification_weight: 0.8\n      localization_weight: 0.2\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, localization_loss, classification_weight, localization_weight, hard_example_miner, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationLoss))
        self.assertTrue(isinstance(localization_loss, losses.WeightedL2LocalizationLoss))
        self.assertAlmostEqual(classification_weight, 0.8)
        self.assertAlmostEqual(localization_weight, 0.2)

    def test_build_reweighting_unmatched_anchors(self):
        if False:
            while True:
                i = 10
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_softmax {\n        }\n      }\n      hard_example_miner {\n      }\n      classification_weight: 0.8\n      localization_weight: 0.2\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        (classification_loss, localization_loss, classification_weight, localization_weight, hard_example_miner, _, _) = losses_builder.build(losses_proto)
        self.assertTrue(isinstance(hard_example_miner, losses.HardExampleMiner))
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationLoss))
        self.assertTrue(isinstance(localization_loss, losses.WeightedL2LocalizationLoss))
        self.assertAlmostEqual(classification_weight, 0.8)
        self.assertAlmostEqual(localization_weight, 0.2)

    def test_raise_error_when_both_focal_loss_and_hard_example_miner(self):
        if False:
            for i in range(10):
                print('nop')
        losses_text_proto = '\n      localization_loss {\n        weighted_l2 {\n        }\n      }\n      classification_loss {\n        weighted_sigmoid_focal {\n        }\n      }\n      hard_example_miner {\n      }\n      classification_weight: 0.8\n      localization_weight: 0.2\n    '
        losses_proto = losses_pb2.Loss()
        text_format.Merge(losses_text_proto, losses_proto)
        with self.assertRaises(ValueError):
            losses_builder.build(losses_proto)

class FasterRcnnClassificationLossBuilderTest(tf.test.TestCase):

    def test_build_sigmoid_loss(self):
        if False:
            print('Hello World!')
        losses_text_proto = '\n      weighted_sigmoid {\n      }\n    '
        losses_proto = losses_pb2.ClassificationLoss()
        text_format.Merge(losses_text_proto, losses_proto)
        classification_loss = losses_builder.build_faster_rcnn_classification_loss(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSigmoidClassificationLoss))

    def test_build_softmax_loss(self):
        if False:
            while True:
                i = 10
        losses_text_proto = '\n      weighted_softmax {\n      }\n    '
        losses_proto = losses_pb2.ClassificationLoss()
        text_format.Merge(losses_text_proto, losses_proto)
        classification_loss = losses_builder.build_faster_rcnn_classification_loss(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationLoss))

    def test_build_logits_softmax_loss(self):
        if False:
            return 10
        losses_text_proto = '\n      weighted_logits_softmax {\n      }\n    '
        losses_proto = losses_pb2.ClassificationLoss()
        text_format.Merge(losses_text_proto, losses_proto)
        classification_loss = losses_builder.build_faster_rcnn_classification_loss(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationAgainstLogitsLoss))

    def test_build_sigmoid_focal_loss(self):
        if False:
            i = 10
            return i + 15
        losses_text_proto = '\n      weighted_sigmoid_focal {\n      }\n    '
        losses_proto = losses_pb2.ClassificationLoss()
        text_format.Merge(losses_text_proto, losses_proto)
        classification_loss = losses_builder.build_faster_rcnn_classification_loss(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.SigmoidFocalClassificationLoss))

    def test_build_softmax_loss_by_default(self):
        if False:
            for i in range(10):
                print('nop')
        losses_text_proto = '\n    '
        losses_proto = losses_pb2.ClassificationLoss()
        text_format.Merge(losses_text_proto, losses_proto)
        classification_loss = losses_builder.build_faster_rcnn_classification_loss(losses_proto)
        self.assertTrue(isinstance(classification_loss, losses.WeightedSoftmaxClassificationLoss))
if __name__ == '__main__':
    tf.test.main()