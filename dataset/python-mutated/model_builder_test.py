"""Tests for object_detection.models.model_builder."""
from absl.testing import parameterized
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.meta_architectures import rfcn_meta_arch
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import ssd_resnet_v1_fpn_feature_extractor as ssd_resnet_v1_fpn
from object_detection.protos import hyperparams_pb2
from object_detection.protos import losses_pb2
from object_detection.protos import model_pb2

class ModelBuilderTest(tf.test.TestCase, parameterized.TestCase):

    def create_model(self, model_config, is_training=True):
        if False:
            return 10
        'Builds a DetectionModel based on the model config.\n\n    Args:\n      model_config: A model.proto object containing the config for the desired\n        DetectionModel.\n      is_training: True if this model is being built for training purposes.\n\n    Returns:\n      DetectionModel based on the config.\n    '
        return model_builder.build(model_config, is_training=is_training)

    def create_default_ssd_model_proto(self):
        if False:
            i = 10
            return i + 15
        'Creates a DetectionModel proto with ssd model fields populated.'
        model_text_proto = "\n      ssd {\n        feature_extractor {\n          type: 'ssd_inception_v2'\n          conv_hyperparams {\n            regularizer {\n                l2_regularizer {\n                }\n              }\n              initializer {\n                truncated_normal_initializer {\n                }\n              }\n          }\n          override_base_feature_extractor_hyperparams: true\n        }\n        box_coder {\n          faster_rcnn_box_coder {\n          }\n        }\n        matcher {\n          argmax_matcher {\n          }\n        }\n        similarity_calculator {\n          iou_similarity {\n          }\n        }\n        anchor_generator {\n          ssd_anchor_generator {\n            aspect_ratios: 1.0\n          }\n        }\n        image_resizer {\n          fixed_shape_resizer {\n            height: 320\n            width: 320\n          }\n        }\n        box_predictor {\n          convolutional_box_predictor {\n            conv_hyperparams {\n              regularizer {\n                l2_regularizer {\n                }\n              }\n              initializer {\n                truncated_normal_initializer {\n                }\n              }\n            }\n          }\n        }\n        loss {\n          classification_loss {\n            weighted_softmax {\n            }\n          }\n          localization_loss {\n            weighted_smooth_l1 {\n            }\n          }\n        }\n      }"
        model_proto = model_pb2.DetectionModel()
        text_format.Merge(model_text_proto, model_proto)
        return model_proto

    def create_default_faster_rcnn_model_proto(self):
        if False:
            i = 10
            return i + 15
        'Creates a DetectionModel proto with FasterRCNN model fields populated.'
        model_text_proto = "\n      faster_rcnn {\n        inplace_batchnorm_update: false\n        num_classes: 3\n        image_resizer {\n          keep_aspect_ratio_resizer {\n            min_dimension: 600\n            max_dimension: 1024\n          }\n        }\n        feature_extractor {\n          type: 'faster_rcnn_resnet101'\n        }\n        first_stage_anchor_generator {\n          grid_anchor_generator {\n            scales: [0.25, 0.5, 1.0, 2.0]\n            aspect_ratios: [0.5, 1.0, 2.0]\n            height_stride: 16\n            width_stride: 16\n          }\n        }\n        first_stage_box_predictor_conv_hyperparams {\n          regularizer {\n            l2_regularizer {\n            }\n          }\n          initializer {\n            truncated_normal_initializer {\n            }\n          }\n        }\n        initial_crop_size: 14\n        maxpool_kernel_size: 2\n        maxpool_stride: 2\n        second_stage_box_predictor {\n          mask_rcnn_box_predictor {\n            conv_hyperparams {\n              regularizer {\n                l2_regularizer {\n                }\n              }\n              initializer {\n                truncated_normal_initializer {\n                }\n              }\n            }\n            fc_hyperparams {\n              op: FC\n              regularizer {\n                l2_regularizer {\n                }\n              }\n              initializer {\n                truncated_normal_initializer {\n                }\n              }\n            }\n          }\n        }\n        second_stage_post_processing {\n          batch_non_max_suppression {\n            score_threshold: 0.01\n            iou_threshold: 0.6\n            max_detections_per_class: 100\n            max_total_detections: 300\n          }\n          score_converter: SOFTMAX\n        }\n      }"
        model_proto = model_pb2.DetectionModel()
        text_format.Merge(model_text_proto, model_proto)
        return model_proto

    def test_create_ssd_models_from_config(self):
        if False:
            print('Hello World!')
        model_proto = self.create_default_ssd_model_proto()
        ssd_feature_extractor_map = {}
        ssd_feature_extractor_map.update(model_builder.SSD_FEATURE_EXTRACTOR_CLASS_MAP)
        ssd_feature_extractor_map.update(model_builder.SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP)
        for (extractor_type, extractor_class) in ssd_feature_extractor_map.items():
            model_proto.ssd.feature_extractor.type = extractor_type
            model = model_builder.build(model_proto, is_training=True)
            self.assertIsInstance(model, ssd_meta_arch.SSDMetaArch)
            self.assertIsInstance(model._feature_extractor, extractor_class)

    def test_create_ssd_fpn_model_from_config(self):
        if False:
            i = 10
            return i + 15
        model_proto = self.create_default_ssd_model_proto()
        model_proto.ssd.feature_extractor.type = 'ssd_resnet101_v1_fpn'
        model_proto.ssd.feature_extractor.fpn.min_level = 3
        model_proto.ssd.feature_extractor.fpn.max_level = 7
        model = model_builder.build(model_proto, is_training=True)
        self.assertIsInstance(model._feature_extractor, ssd_resnet_v1_fpn.SSDResnet101V1FpnFeatureExtractor)
        self.assertEqual(model._feature_extractor._fpn_min_level, 3)
        self.assertEqual(model._feature_extractor._fpn_max_level, 7)

    @parameterized.named_parameters({'testcase_name': 'mask_rcnn_with_matmul', 'use_matmul_crop_and_resize': False, 'enable_mask_prediction': True}, {'testcase_name': 'mask_rcnn_without_matmul', 'use_matmul_crop_and_resize': True, 'enable_mask_prediction': True}, {'testcase_name': 'faster_rcnn_with_matmul', 'use_matmul_crop_and_resize': False, 'enable_mask_prediction': False}, {'testcase_name': 'faster_rcnn_without_matmul', 'use_matmul_crop_and_resize': True, 'enable_mask_prediction': False})
    def test_create_faster_rcnn_models_from_config(self, use_matmul_crop_and_resize, enable_mask_prediction):
        if False:
            return 10
        model_proto = self.create_default_faster_rcnn_model_proto()
        faster_rcnn_config = model_proto.faster_rcnn
        faster_rcnn_config.use_matmul_crop_and_resize = use_matmul_crop_and_resize
        if enable_mask_prediction:
            faster_rcnn_config.second_stage_mask_prediction_loss_weight = 3.0
            mask_predictor_config = faster_rcnn_config.second_stage_box_predictor.mask_rcnn_box_predictor
            mask_predictor_config.predict_instance_masks = True
        for (extractor_type, extractor_class) in model_builder.FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP.items():
            faster_rcnn_config.feature_extractor.type = extractor_type
            model = model_builder.build(model_proto, is_training=True)
            self.assertIsInstance(model, faster_rcnn_meta_arch.FasterRCNNMetaArch)
            self.assertIsInstance(model._feature_extractor, extractor_class)
            if enable_mask_prediction:
                self.assertAlmostEqual(model._second_stage_mask_loss_weight, 3.0)

    def test_create_faster_rcnn_model_from_config_with_example_miner(self):
        if False:
            for i in range(10):
                print('nop')
        model_proto = self.create_default_faster_rcnn_model_proto()
        model_proto.faster_rcnn.hard_example_miner.num_hard_examples = 64
        model = model_builder.build(model_proto, is_training=True)
        self.assertIsNotNone(model._hard_example_miner)

    def test_create_rfcn_model_from_config(self):
        if False:
            for i in range(10):
                print('nop')
        model_proto = self.create_default_faster_rcnn_model_proto()
        rfcn_predictor_config = model_proto.faster_rcnn.second_stage_box_predictor.rfcn_box_predictor
        rfcn_predictor_config.conv_hyperparams.op = hyperparams_pb2.Hyperparams.CONV
        for (extractor_type, extractor_class) in model_builder.FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP.items():
            model_proto.faster_rcnn.feature_extractor.type = extractor_type
            model = model_builder.build(model_proto, is_training=True)
            self.assertIsInstance(model, rfcn_meta_arch.RFCNMetaArch)
            self.assertIsInstance(model._feature_extractor, extractor_class)

    def test_invalid_model_config_proto(self):
        if False:
            i = 10
            return i + 15
        model_proto = ''
        with self.assertRaisesRegexp(ValueError, 'model_config not of type model_pb2.DetectionModel.'):
            model_builder.build(model_proto, is_training=True)

    def test_unknown_meta_architecture(self):
        if False:
            for i in range(10):
                print('nop')
        model_proto = model_pb2.DetectionModel()
        with self.assertRaisesRegexp(ValueError, 'Unknown meta architecture'):
            model_builder.build(model_proto, is_training=True)

    def test_unknown_ssd_feature_extractor(self):
        if False:
            while True:
                i = 10
        model_proto = self.create_default_ssd_model_proto()
        model_proto.ssd.feature_extractor.type = 'unknown_feature_extractor'
        with self.assertRaisesRegexp(ValueError, 'Unknown ssd feature_extractor'):
            model_builder.build(model_proto, is_training=True)

    def test_unknown_faster_rcnn_feature_extractor(self):
        if False:
            i = 10
            return i + 15
        model_proto = self.create_default_faster_rcnn_model_proto()
        model_proto.faster_rcnn.feature_extractor.type = 'unknown_feature_extractor'
        with self.assertRaisesRegexp(ValueError, 'Unknown Faster R-CNN feature_extractor'):
            model_builder.build(model_proto, is_training=True)

    def test_invalid_first_stage_nms_iou_threshold(self):
        if False:
            print('Hello World!')
        model_proto = self.create_default_faster_rcnn_model_proto()
        model_proto.faster_rcnn.first_stage_nms_iou_threshold = 1.1
        with self.assertRaisesRegexp(ValueError, 'iou_threshold not in \\[0, 1\\.0\\]'):
            model_builder.build(model_proto, is_training=True)
        model_proto.faster_rcnn.first_stage_nms_iou_threshold = -0.1
        with self.assertRaisesRegexp(ValueError, 'iou_threshold not in \\[0, 1\\.0\\]'):
            model_builder.build(model_proto, is_training=True)

    def test_invalid_second_stage_batch_size(self):
        if False:
            while True:
                i = 10
        model_proto = self.create_default_faster_rcnn_model_proto()
        model_proto.faster_rcnn.first_stage_max_proposals = 1
        model_proto.faster_rcnn.second_stage_batch_size = 2
        with self.assertRaisesRegexp(ValueError, 'second_stage_batch_size should be no greater than first_stage_max_proposals.'):
            model_builder.build(model_proto, is_training=True)

    def test_invalid_faster_rcnn_batchnorm_update(self):
        if False:
            return 10
        model_proto = self.create_default_faster_rcnn_model_proto()
        model_proto.faster_rcnn.inplace_batchnorm_update = True
        with self.assertRaisesRegexp(ValueError, 'inplace batchnorm updates not supported'):
            model_builder.build(model_proto, is_training=True)

    def test_create_experimental_model(self):
        if False:
            print('Hello World!')
        model_text_proto = "\n      experimental_model {\n        name: 'model42'\n      }"
        build_func = lambda *args: 42
        model_builder.EXPERIMENTAL_META_ARCH_BUILDER_MAP['model42'] = build_func
        model_proto = model_pb2.DetectionModel()
        text_format.Merge(model_text_proto, model_proto)
        self.assertEqual(model_builder.build(model_proto, is_training=True), 42)
if __name__ == '__main__':
    tf.test.main()