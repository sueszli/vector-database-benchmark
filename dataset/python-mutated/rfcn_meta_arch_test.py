"""Tests for object_detection.meta_architectures.rfcn_meta_arch."""
import tensorflow as tf
from object_detection.meta_architectures import faster_rcnn_meta_arch_test_lib
from object_detection.meta_architectures import rfcn_meta_arch

class RFCNMetaArchTest(faster_rcnn_meta_arch_test_lib.FasterRCNNMetaArchTestBase):

    def _get_second_stage_box_predictor_text_proto(self, share_box_across_classes=False):
        if False:
            return 10
        del share_box_across_classes
        box_predictor_text_proto = '\n      rfcn_box_predictor {\n        conv_hyperparams {\n          op: CONV\n          activation: NONE\n          regularizer {\n            l2_regularizer {\n              weight: 0.0005\n            }\n          }\n          initializer {\n            variance_scaling_initializer {\n              factor: 1.0\n              uniform: true\n              mode: FAN_AVG\n            }\n          }\n        }\n      }\n    '
        return box_predictor_text_proto

    def _get_model(self, box_predictor, **common_kwargs):
        if False:
            print('Hello World!')
        return rfcn_meta_arch.RFCNMetaArch(second_stage_rfcn_box_predictor=box_predictor, **common_kwargs)

    def _get_box_classifier_features_shape(self, image_size, batch_size, max_num_proposals, initial_crop_size, maxpool_stride, num_features):
        if False:
            for i in range(10):
                print('nop')
        return (batch_size, image_size, image_size, num_features)
if __name__ == '__main__':
    tf.test.main()