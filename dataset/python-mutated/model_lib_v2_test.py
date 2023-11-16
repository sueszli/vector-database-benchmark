"""Tests for object detection model library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from object_detection import model_hparams
from object_detection import model_lib_v2
from object_detection.utils import config_util
MODEL_NAME_FOR_TEST = 'ssd_mobilenet_v2_pets_keras'

def _get_data_path():
    if False:
        for i in range(10):
            print('nop')
    'Returns an absolute path to TFRecord file.'
    return os.path.join(tf.resource_loader.get_data_files_path(), 'test_data', 'pets_examples.record')

def get_pipeline_config_path(model_name):
    if False:
        while True:
            i = 10
    'Returns path to the local pipeline config file.'
    return os.path.join(tf.resource_loader.get_data_files_path(), 'samples', 'configs', model_name + '.config')

def _get_labelmap_path():
    if False:
        print('Hello World!')
    'Returns an absolute path to label map file.'
    return os.path.join(tf.resource_loader.get_data_files_path(), 'data', 'pet_label_map.pbtxt')

def _get_config_kwarg_overrides():
    if False:
        for i in range(10):
            print('nop')
    'Returns overrides to the configs that insert the correct local paths.'
    data_path = _get_data_path()
    label_map_path = _get_labelmap_path()
    return {'train_input_path': data_path, 'eval_input_path': data_path, 'label_map_path': label_map_path}

def _get_configs_for_model(model_name):
    if False:
        i = 10
        return i + 15
    'Returns configurations for model.'
    filename = get_pipeline_config_path(model_name)
    configs = config_util.get_configs_from_pipeline_file(filename)
    configs = config_util.merge_external_params_with_configs(configs, kwargs_dict=_get_config_kwarg_overrides())
    return configs

class ModelLibTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        tf.keras.backend.clear_session()

    def test_train_loop_then_eval_loop(self):
        if False:
            print('Hello World!')
        'Tests that Estimator and input function are constructed correctly.'
        hparams = model_hparams.create_hparams(hparams_overrides='load_pretrained=false')
        pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
        config_kwarg_overrides = _get_config_kwarg_overrides()
        model_dir = tf.test.get_temp_dir()
        train_steps = 2
        model_lib_v2.train_loop(hparams, pipeline_config_path, model_dir=model_dir, train_steps=train_steps, checkpoint_every_n=1, **config_kwarg_overrides)
        model_lib_v2.eval_continuously(hparams, pipeline_config_path, model_dir=model_dir, checkpoint_dir=model_dir, train_steps=train_steps, wait_interval=10, **config_kwarg_overrides)