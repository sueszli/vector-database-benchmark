"""Tests for dataset_builder."""
import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from lstm_object_detection.inputs import seq_dataset_builder
from lstm_object_detection.protos import pipeline_pb2 as internal_pipeline_pb2
from object_detection.builders import preprocessor_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import preprocessor_pb2

class DatasetBuilderTest(tf.test.TestCase):

    def _get_model_configs_from_proto(self):
        if False:
            i = 10
            return i + 15
        'Creates a model text proto for testing.\n\n    Returns:\n      A dictionary of model configs.\n    '
        model_text_proto = "\n    [lstm_object_detection.protos.lstm_model] {\n      train_unroll_length: 4\n      eval_unroll_length: 4\n    }\n    model {\n      ssd {\n        feature_extractor {\n          type: 'lstm_mobilenet_v1_fpn'\n          conv_hyperparams {\n            regularizer {\n                l2_regularizer {\n                }\n              }\n              initializer {\n                truncated_normal_initializer {\n                }\n              }\n          }\n        }\n        negative_class_weight: 2.0\n        box_coder {\n          faster_rcnn_box_coder {\n          }\n        }\n        matcher {\n          argmax_matcher {\n          }\n        }\n        similarity_calculator {\n          iou_similarity {\n          }\n        }\n        anchor_generator {\n          ssd_anchor_generator {\n            aspect_ratios: 1.0\n          }\n        }\n        image_resizer {\n          fixed_shape_resizer {\n            height: 32\n            width: 32\n          }\n        }\n        box_predictor {\n          convolutional_box_predictor {\n            conv_hyperparams {\n              regularizer {\n                l2_regularizer {\n                }\n              }\n              initializer {\n                truncated_normal_initializer {\n                }\n              }\n            }\n          }\n        }\n        normalize_loc_loss_by_codesize: true\n        loss {\n          classification_loss {\n            weighted_softmax {\n            }\n          }\n          localization_loss {\n            weighted_smooth_l1 {\n            }\n          }\n        }\n      }\n    }"
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        text_format.Merge(model_text_proto, pipeline_config)
        configs = {}
        configs['model'] = pipeline_config.model
        configs['lstm_model'] = pipeline_config.Extensions[internal_pipeline_pb2.lstm_model]
        return configs

    def _get_data_augmentation_preprocessor_proto(self):
        if False:
            print('Hello World!')
        preprocessor_text_proto = '\n    random_horizontal_flip {\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        return preprocessor_proto

    def _create_training_dict(self, tensor_dict):
        if False:
            i = 10
            return i + 15
        image_dict = {}
        all_dict = {}
        all_dict['batch'] = tensor_dict.pop('batch')
        for (i, _) in enumerate(tensor_dict[fields.InputDataFields.image]):
            for (key, val) in tensor_dict.items():
                image_dict[key] = val[i]
            image_dict[fields.InputDataFields.image] = tf.to_float(tf.expand_dims(image_dict[fields.InputDataFields.image], 0))
            suffix = str(i)
            for (key, val) in image_dict.items():
                all_dict[key + suffix] = val
        return all_dict

    def _get_input_proto(self, input_reader):
        if False:
            return 10
        return "\n        external_input_reader {\n          [lstm_object_detection.protos.GoogleInputReader.google_input_reader] {\n            %s: {\n              input_path: '{0}'\n              data_type: TF_SEQUENCE_EXAMPLE\n              video_length: 4\n            }\n          }\n        }\n      " % input_reader

    def test_video_input_reader(self):
        if False:
            while True:
                i = 10
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(self._get_input_proto('tf_record_video_input_reader'), input_reader_proto)
        configs = self._get_model_configs_from_proto()
        tensor_dict = seq_dataset_builder.build(input_reader_proto, configs['model'], configs['lstm_model'], unroll_length=1)
        all_dict = self._create_training_dict(tensor_dict)
        self.assertEqual((1, 32, 32, 3), all_dict['image0'].shape)
        self.assertEqual(4, all_dict['groundtruth_boxes0'].shape[1])

    def test_build_with_data_augmentation(self):
        if False:
            i = 10
            return i + 15
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(self._get_input_proto('tf_record_video_input_reader'), input_reader_proto)
        configs = self._get_model_configs_from_proto()
        data_augmentation_options = [preprocessor_builder.build(self._get_data_augmentation_preprocessor_proto())]
        tensor_dict = seq_dataset_builder.build(input_reader_proto, configs['model'], configs['lstm_model'], unroll_length=1, data_augmentation_options=data_augmentation_options)
        all_dict = self._create_training_dict(tensor_dict)
        self.assertEqual((1, 32, 32, 3), all_dict['image0'].shape)
        self.assertEqual(4, all_dict['groundtruth_boxes0'].shape[1])

    def test_raises_error_without_input_paths(self):
        if False:
            while True:
                i = 10
        input_reader_text_proto = '\n      shuffle: false\n      num_readers: 1\n      load_instance_masks: true\n    '
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        configs = self._get_model_configs_from_proto()
        with self.assertRaises(ValueError):
            _ = seq_dataset_builder.build(input_reader_proto, configs['model'], configs['lstm_model'], unroll_length=1)
if __name__ == '__main__':
    tf.test.main()