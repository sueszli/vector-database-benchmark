"""Model defination for the RetinaNet Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
from absl import logging
import tensorflow.compat.v2 as tf
from tensorflow.python.keras import backend
from official.vision.detection.dataloader import mode_keys
from official.vision.detection.modeling import base_model
from official.vision.detection.modeling import losses
from official.vision.detection.modeling import postprocess
from official.vision.detection.modeling.architecture import factory
from official.vision.detection.evaluation import factory as eval_factory

class COCOMetrics(object):

    def __init__(self, params):
        if False:
            i = 10
            return i + 15
        self._evaluator = eval_factory.evaluator_generator(params.eval)

    def update_state(self, y_true, y_pred):
        if False:
            i = 10
            return i + 15
        labels = tf.nest.map_structure(lambda x: x.numpy(), y_true)
        outputs = tf.nest.map_structure(lambda x: x.numpy(), y_pred)
        groundtruths = {}
        predictions = {}
        for (key, val) in outputs.items():
            if isinstance(val, tuple):
                val = np.concatenate(val)
            predictions[key] = val
        for (key, val) in labels.items():
            if isinstance(val, tuple):
                val = np.concatenate(val)
            groundtruths[key] = val
        self._evaluator.update(predictions, groundtruths)

    def result(self):
        if False:
            i = 10
            return i + 15
        return self._evaluator.evaluate()

    def reset_states(self):
        if False:
            for i in range(10):
                print('nop')
        return self._evaluator.reset()

class RetinanetModel(base_model.Model):
    """RetinaNet model function."""

    def __init__(self, params):
        if False:
            i = 10
            return i + 15
        super(RetinanetModel, self).__init__(params)
        self._params = params
        self._backbone_fn = factory.backbone_generator(params)
        self._fpn_fn = factory.multilevel_features_generator(params)
        self._head_fn = factory.retinanet_head_generator(params.retinanet_head)
        self._cls_loss_fn = losses.RetinanetClassLoss(params.retinanet_loss)
        self._box_loss_fn = losses.RetinanetBoxLoss(params.retinanet_loss)
        self._box_loss_weight = params.retinanet_loss.box_loss_weight
        self._keras_model = None
        self._generate_detections_fn = postprocess.GenerateOneStageDetections(params.postprocess)
        self._l2_weight_decay = params.train.l2_weight_decay
        self._transpose_input = params.train.transpose_input
        assert not self._transpose_input, 'Transpose input is not supportted.'
        input_shape = params.retinanet_parser.output_size + [params.retinanet_parser.num_channels]
        self._input_layer = tf.keras.layers.Input(shape=input_shape, name='', dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)

    def build_outputs(self, inputs, mode):
        if False:
            i = 10
            return i + 15
        backbone_features = self._backbone_fn(inputs, is_training=mode == mode_keys.TRAIN)
        fpn_features = self._fpn_fn(backbone_features, is_training=mode == mode_keys.TRAIN)
        (cls_outputs, box_outputs) = self._head_fn(fpn_features, is_training=mode == mode_keys.TRAIN)
        if self._use_bfloat16:
            levels = cls_outputs.keys()
            for level in levels:
                cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
                box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
        model_outputs = {'cls_outputs': cls_outputs, 'box_outputs': box_outputs}
        return model_outputs

    def build_loss_fn(self):
        if False:
            print('Hello World!')
        if self._keras_model is None:
            raise ValueError('build_loss_fn() must be called after build_model().')
        filter_fn = self.make_filter_trainable_variables_fn()
        trainable_variables = filter_fn(self._keras_model.trainable_variables)

        def _total_loss_fn(labels, outputs):
            if False:
                while True:
                    i = 10
            cls_loss = self._cls_loss_fn(outputs['cls_outputs'], labels['cls_targets'], labels['num_positives'])
            box_loss = self._box_loss_fn(outputs['box_outputs'], labels['box_targets'], labels['num_positives'])
            model_loss = cls_loss + self._box_loss_weight * box_loss
            l2_regularization_loss = self.weight_decay_loss(self._l2_weight_decay, trainable_variables)
            total_loss = model_loss + l2_regularization_loss
            return {'total_loss': total_loss, 'cls_loss': cls_loss, 'box_loss': box_loss, 'model_loss': model_loss, 'l2_regularization_loss': l2_regularization_loss}
        return _total_loss_fn

    def build_model(self, params, mode=None):
        if False:
            i = 10
            return i + 15
        if self._keras_model is None:
            with backend.get_graph().as_default():
                outputs = self.model_outputs(self._input_layer, mode)
                model = tf.keras.models.Model(inputs=self._input_layer, outputs=outputs, name='retinanet')
                assert model is not None, 'Fail to build tf.keras.Model.'
                model.optimizer = self.build_optimizer()
                self._keras_model = model
        return self._keras_model

    def post_processing(self, labels, outputs):
        if False:
            for i in range(10):
                print('nop')
        required_output_fields = ['cls_outputs', 'box_outputs']
        for field in required_output_fields:
            if field not in outputs:
                raise ValueError('"%s" is missing in outputs, requried %s found %s', field, required_output_fields, outputs.keys())
        required_label_fields = ['image_info', 'groundtruths']
        for field in required_label_fields:
            if field not in labels:
                raise ValueError('"%s" is missing in outputs, requried %s found %s', field, required_label_fields, labels.keys())
        (boxes, scores, classes, valid_detections) = self._generate_detections_fn(inputs=(outputs['box_outputs'], outputs['cls_outputs'], labels['anchor_boxes'], labels['image_info'][:, 1:2, :]))
        outputs = {'source_id': labels['groundtruths']['source_id'], 'image_info': labels['image_info'], 'num_detections': valid_detections, 'detection_boxes': boxes, 'detection_classes': classes, 'detection_scores': scores}
        if 'groundtruths' in labels:
            labels['source_id'] = labels['groundtruths']['source_id']
            labels['boxes'] = labels['groundtruths']['boxes']
            labels['classes'] = labels['groundtruths']['classes']
            labels['areas'] = labels['groundtruths']['areas']
            labels['is_crowds'] = labels['groundtruths']['is_crowds']
        return (labels, outputs)

    def eval_metrics(self):
        if False:
            for i in range(10):
                print('nop')
        return COCOMetrics(self._params)