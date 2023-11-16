from __future__ import annotations
import copy
import inspect
import json
import os
import random
import tempfile
import unittest
from importlib import import_module
from math import isnan
from typing import List, Tuple
from datasets import Dataset
from transformers import is_tf_available, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import CaptureLogger, _tf_gpu_memory_limit, is_pt_tf_cross_test, require_tf, require_tf2onnx, slow, torch_device
from transformers.utils import CONFIG_NAME, GENERATION_CONFIG_NAME, logging
from transformers.utils.generic import ModelOutput
logger = logging.get_logger(__name__)
if is_tf_available():
    import numpy as np
    import tensorflow as tf
    from transformers import TF_MODEL_FOR_CAUSAL_LM_MAPPING, TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING, TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING, TF_MODEL_FOR_MASKED_LM_MAPPING, TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING, TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING, TF_MODEL_FOR_PRETRAINING_MAPPING, TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING, TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING, TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING, TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, TFAutoModel, TFAutoModelForSequenceClassification, TFSharedEmbeddings
    from transformers.generation import TFBeamSampleDecoderOnlyOutput, TFBeamSampleEncoderDecoderOutput, TFBeamSearchDecoderOnlyOutput, TFBeamSearchEncoderDecoderOutput, TFGreedySearchDecoderOnlyOutput, TFGreedySearchEncoderDecoderOutput, TFSampleDecoderOnlyOutput, TFSampleEncoderDecoderOutput
    tf.config.experimental.enable_tensor_float_32_execution(False)
    if _tf_gpu_memory_limit is not None:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=_tf_gpu_memory_limit)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print('Logical GPUs', logical_gpus)
            except RuntimeError as e:
                print(e)
if is_torch_available():
    import torch

def _config_zero_init(config):
    if False:
        for i in range(10):
            print('nop')
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if '_range' in key or '_std' in key:
            setattr(configs_no_init, key, 0.0)
    return configs_no_init

@require_tf
class TFModelTesterMixin:
    model_tester = None
    all_model_classes = ()
    all_generative_model_classes = ()
    test_mismatched_shapes = True
    test_resize_embeddings = True
    test_head_masking = True
    is_encoder_decoder = False
    has_attentions = True

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False) -> dict:
        if False:
            while True:
                i = 10
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
            inputs_dict = {k: tf.tile(tf.expand_dims(v, 1), (1, self.model_tester.num_choices) + (1,) * (v.ndim - 1)) if isinstance(v, tf.Tensor) and v.ndim > 0 else v for (k, v) in inputs_dict.items()}
        if return_labels:
            if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict['labels'] = tf.ones(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [*get_values(TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING), *get_values(TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING)]:
                inputs_dict['start_positions'] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
                inputs_dict['end_positions'] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [*get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING), *get_values(TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING)]:
                inputs_dict['labels'] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING):
                inputs_dict['next_sentence_label'] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [*get_values(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING), *get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING), *get_values(TF_MODEL_FOR_MASKED_LM_MAPPING), *get_values(TF_MODEL_FOR_PRETRAINING_MAPPING), *get_values(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING), *get_values(TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING)] and 'labels' in dict(inspect.signature(model_class.call).parameters):
                inputs_dict['labels'] = tf.zeros((self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING):
                num_patches = self.model_tester.image_size // self.model_tester.patch_size
                inputs_dict['bool_masked_pos'] = tf.zeros((self.model_tester.batch_size, num_patches ** 2), dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING):
                (batch_size, num_channels, height, width) = inputs_dict['pixel_values'].shape
                inputs_dict['labels'] = tf.zeros((self.model_tester.batch_size, height, width), dtype=tf.int32)
            elif model_class.__name__.endswith('ForCTC'):
                inputs_dict['labels'] = tf.zeros((self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int32)
        return inputs_dict

    def test_initialization(self):
        if False:
            return 10
        pass

    def test_save_load(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME)))
                model = model_class.from_pretrained(tmpdirname)
                after_outputs = model(self._prepare_for_class(inputs_dict, model_class))
                self.assert_outputs_same(after_outputs, outputs)

    def test_save_load_config(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            model_config = model.get_config()
            json.dumps(model_config)
            new_model = model_class.from_config(model.get_config())
            _ = model_class.from_config(model.config)
            _ = new_model(self._prepare_for_class(inputs_dict, model_class))
            new_model.set_weights(model.get_weights())
            after_outputs = new_model(self._prepare_for_class(inputs_dict, model_class))
            self.assert_outputs_same(after_outputs, outputs)

    @slow
    def test_saved_model_creation(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = False
        config.output_attentions = False
        if hasattr(config, 'use_cache'):
            config.use_cache = False
        model_class = self.all_model_classes[0]
        class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
        model = model_class(config)
        model(class_inputs_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, saved_model=True)
            saved_model_dir = os.path.join(tmpdirname, 'saved_model', '1')
            self.assertTrue(os.path.exists(saved_model_dir))

    def test_prepare_serving_output(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions
        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)
            outputs = model(inputs)
            serving_outputs = model.serving_output(outputs)
            for (k, v) in serving_outputs.items():
                if isinstance(v, tuple):
                    self.assertTrue(all((isinstance(elem, tf.Tensor) for elem in v)))
                elif v is not None:
                    self.assertIsInstance(v, tf.Tensor)
                else:
                    self.assertIsNone(v)

    def test_forward_signature(self):
        if False:
            for i in range(10):
                print('nop')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            arg_names = [*signature.parameters.keys()]
            if model.config.is_encoder_decoder:
                expected_arg_names = ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask']
                expected_arg_names.extend(['decoder_position_ids'] if 'decoder_position_ids' in arg_names else [])
                expected_arg_names.extend(['head_mask', 'decoder_head_mask'] if 'head_mask' and 'decoder_head_mask' in arg_names else [])
                expected_arg_names.extend(['cross_attn_head_mask', 'encoder_outputs'] if 'cross_attn_head_mask' in arg_names else ['encoder_outputs'])
                self.assertListEqual(arg_names[:len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ['input_ids']
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_onnx_compliancy(self):
        if False:
            i = 10
            return i + 15
        if not self.test_onnx:
            return
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        INTERNAL_OPS = ['Assert', 'AssignVariableOp', 'EmptyTensorList', 'ReadVariableOp', 'ResourceGather', 'TruncatedNormal', 'VarHandleOp', 'VarIsInitializedOp']
        onnx_ops = []
        with open(os.path.join('.', 'utils', 'tf_ops', 'onnx.json')) as f:
            onnx_opsets = json.load(f)['opsets']
        for i in range(1, self.onnx_min_opset + 1):
            onnx_ops.extend(onnx_opsets[str(i)])
        for model_class in self.all_model_classes:
            model_op_names = set()
            with tf.Graph().as_default() as g:
                model = model_class(config)
                model.build()
                for op in g.get_operations():
                    model_op_names.add(op.node_def.op)
            model_op_names = sorted(model_op_names)
            incompatible_ops = []
            for op in model_op_names:
                if op not in onnx_ops and op not in INTERNAL_OPS:
                    incompatible_ops.append(op)
            self.assertEqual(len(incompatible_ops), 0, incompatible_ops)

    @unittest.skip('`tf2onnx` broke with TF 2.13')
    @require_tf2onnx
    @slow
    def test_onnx_runtime_optimize(self):
        if False:
            print('Hello World!')
        if not self.test_onnx:
            return
        import onnxruntime
        import tf2onnx
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes[:2]:
            model = model_class(config)
            model.build()
            (onnx_model_proto, _) = tf2onnx.convert.from_keras(model, opset=self.onnx_min_opset)
            onnxruntime.InferenceSession(onnx_model_proto.SerializeToString())

    def test_keras_save_load(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        tf_main_layer_classes = {module_member for model_class in self.all_model_classes for module in (import_module(model_class.__module__),) for module_member_name in dir(module) if module_member_name.endswith('MainLayer') and module_member_name[:-len('MainLayer')] == model_class.__name__[:-len('Model')] for module_member in (getattr(module, module_member_name),) if isinstance(module_member, type) and tf.keras.layers.Layer in module_member.__bases__ and getattr(module_member, '_keras_serializable', False)}
        for main_layer_class in tf_main_layer_classes:
            if 'T5' in main_layer_class.__name__:
                shared = TFSharedEmbeddings(99, 32, name='shared')
                config.use_cache = inputs_dict.pop('use_cache', None)
                main_layer = main_layer_class(config, embed_tokens=shared)
            else:
                main_layer = main_layer_class(config)
            symbolic_inputs = {name: tf.keras.Input(tensor.shape[1:], dtype=tensor.dtype) for (name, tensor) in inputs_dict.items()}
            model = tf.keras.Model(symbolic_inputs, outputs=main_layer(symbolic_inputs))
            outputs = model(inputs_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, 'keras_model.h5')
                model.save(filepath)
                if 'T5' in main_layer_class.__name__:
                    model = tf.keras.models.load_model(filepath, custom_objects={main_layer_class.__name__: main_layer_class, 'TFSharedEmbeddings': TFSharedEmbeddings})
                else:
                    model = tf.keras.models.load_model(filepath, custom_objects={main_layer_class.__name__: main_layer_class})
                assert isinstance(model, tf.keras.Model)
                after_outputs = model(inputs_dict)
                self.assert_outputs_same(after_outputs, outputs)

    def assert_outputs_same(self, after_outputs, outputs):
        if False:
            i = 10
            return i + 15
        if isinstance(after_outputs, tf.Tensor):
            out_1 = after_outputs.numpy()
        elif isinstance(after_outputs, dict):
            out_1 = after_outputs[list(after_outputs.keys())[0]].numpy()
        else:
            out_1 = after_outputs[0].numpy()
        out_2 = outputs[0].numpy()
        self.assertEqual(out_1.shape, out_2.shape)
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, 1e-05)

    def _make_attention_mask_non_null(self, inputs_dict):
        if False:
            for i in range(10):
                print('nop')
        'Make sure no sequence has all zeros as attention mask'
        for k in ['attention_mask', 'encoder_attention_mask', 'decoder_attention_mask']:
            if k in inputs_dict:
                attention_mask = inputs_dict[k]
                attention_mask = tf.concat([tf.ones_like(attention_mask[:, :1], dtype=attention_mask.dtype), attention_mask[:, 1:]], axis=-1)
                inputs_dict[k] = attention_mask

    def _postprocessing_to_ignore_test_cases(self, tf_outputs, pt_outputs, model_class):
        if False:
            i = 10
            return i + 15
        'For temporarily ignoring some failed test cases (issues to be fixed)'
        tf_keys = {k for (k, v) in tf_outputs.items() if v is not None}
        pt_keys = {k for (k, v) in pt_outputs.items() if v is not None}
        key_differences = tf_keys.symmetric_difference(pt_keys)
        if model_class.__name__ in ['TFFlaubertWithLMHeadModel', 'TFFunnelForPreTraining', 'TFElectraForPreTraining', 'TFXLMWithLMHeadModel', 'TFTransfoXLLMHeadModel']:
            for k in key_differences:
                if k in ['loss', 'losses']:
                    tf_keys.discard(k)
                    pt_keys.discard(k)
        elif model_class.__name__.startswith('TFGPT2'):
            tf_keys.discard('past_key_values')
            pt_keys.discard('past_key_values')
        new_tf_outputs = type(tf_outputs)(**{k: tf_outputs[k] for k in tf_keys})
        new_pt_outputs = type(pt_outputs)(**{k: pt_outputs[k] for k in pt_keys})
        return (new_tf_outputs, new_pt_outputs)

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=1e-05, name='outputs', attributes=None):
        if False:
            i = 10
            return i + 15
        "Check the outputs from PyTorch and TensorFlow models are close enough. Checks are done in a recursive way.\n\n        Args:\n            model_class: The class of the model that is currently testing. For example, `TFBertModel`,\n                TFBertForMaskedLM`, `TFBertForSequenceClassification`, etc. Mainly used for providing more informative\n                error messages.\n            name (`str`): The name of the output. For example, `output.hidden_states`, `output.attentions`, etc.\n            attributes (`Tuple[str]`): The names of the output's element if the output is a tuple/list with each element\n                being a named field in the output.\n        "
        self.assertEqual(type(name), str)
        if attributes is not None:
            self.assertEqual(type(attributes), tuple, f'{name}: The argument `attributes` should be a `tuple`')
        if isinstance(tf_outputs, ModelOutput):
            self.assertTrue(isinstance(pt_outputs, ModelOutput), f'{name}: `pt_outputs` should an instance of `ModelOutput` when `tf_outputs` is')
            (tf_outputs, pt_outputs) = self._postprocessing_to_ignore_test_cases(tf_outputs, pt_outputs, model_class)
            tf_keys = [k for (k, v) in tf_outputs.items() if v is not None]
            pt_keys = [k for (k, v) in pt_outputs.items() if v is not None]
            self.assertEqual(tf_keys, pt_keys, f'{name}: Output keys differ between TF and PyTorch')
            attributes = tuple([f'{name}.{k}' for k in tf_keys])
            self.check_pt_tf_outputs(tf_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, tol=tol, name=name, attributes=attributes)
        elif type(tf_outputs) in [tuple, list]:
            self.assertEqual(type(tf_outputs), type(pt_outputs), f'{name}: Output types differ between TF and PyTorch')
            self.assertEqual(len(tf_outputs), len(pt_outputs), f'{name}: Output lengths differ between TF and PyTorch')
            if attributes is not None:
                self.assertEqual(len(attributes), len(tf_outputs), f'{name}: The tuple `names` should have the same length as `tf_outputs`')
            else:
                attributes = tuple([f'{name}_{idx}' for idx in range(len(tf_outputs))])
            for (tf_output, pt_output, attr) in zip(tf_outputs, pt_outputs, attributes):
                self.check_pt_tf_outputs(tf_output, pt_output, model_class, tol=tol, name=attr)
        elif isinstance(tf_outputs, tf.Tensor):
            self.assertTrue(isinstance(pt_outputs, torch.Tensor), f'{name}: `pt_outputs` should a tensor when `tf_outputs` is')
            tf_outputs = tf_outputs.numpy()
            pt_outputs = pt_outputs.detach().to('cpu').numpy()
            self.assertEqual(tf_outputs.shape, pt_outputs.shape, f'{name}: Output shapes differ between TF and PyTorch')
            if np.isscalar(tf_outputs):
                tf_outputs = np.array([tf_outputs])
                pt_outputs = np.array([pt_outputs])
            tf_nans = np.isnan(tf_outputs)
            pt_nans = np.isnan(pt_outputs)
            pt_outputs[tf_nans] = 0
            tf_outputs[tf_nans] = 0
            pt_outputs[pt_nans] = 0
            tf_outputs[pt_nans] = 0
            max_diff = np.amax(np.abs(tf_outputs - pt_outputs))
            self.assertLessEqual(max_diff, tol, f'{name}: Difference between torch and tf is {max_diff} (>= {tol}).')
        else:
            raise ValueError(f'`tf_outputs` should be an instance of `tf.Tensor`, a `tuple`, or an instance of `tf.Tensor`. Got {type(tf_outputs)} instead.')

    def prepare_pt_inputs_from_tf_inputs(self, tf_inputs_dict):
        if False:
            i = 10
            return i + 15
        pt_inputs_dict = {}
        for (name, key) in tf_inputs_dict.items():
            if type(key) == bool:
                pt_inputs_dict[name] = key
            elif name == 'input_values':
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            elif name == 'pixel_values':
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            elif name == 'input_features':
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            elif tf_inputs_dict[name].dtype.is_floating:
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
            else:
                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)
        return pt_inputs_dict

    def check_pt_tf_models(self, tf_model, pt_model, tf_inputs_dict):
        if False:
            while True:
                i = 10
        pt_inputs_dict = self.prepare_pt_inputs_from_tf_inputs(tf_inputs_dict)
        pt_inputs_dict = {k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for (k, v) in pt_inputs_dict.items()}
        pt_model.to(torch_device)
        pt_model.eval()
        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs_dict)
        tf_outputs = tf_model(tf_inputs_dict)
        tf_loss = getattr(tf_outputs, 'loss', None)
        if tf_loss is not None:
            tf_outputs.loss = tf.math.reduce_mean(tf_loss)
        self.check_pt_tf_outputs(tf_outputs, pt_outputs, type(tf_model))

    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self, allow_missing_keys=False):
        if False:
            return 10
        import transformers
        for model_class in self.all_model_classes:
            (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
            config.output_hidden_states = True
            config.output_attentions = self.has_attentions
            self._make_attention_mask_non_null(inputs_dict)
            pt_model_class_name = model_class.__name__[2:]
            pt_model_class = getattr(transformers, pt_model_class_name)
            tf_model = model_class(config)
            pt_model = pt_model_class(config)
            tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            tf_inputs_dict_with_labels = self._prepare_for_class(inputs_dict, model_class, return_labels=True if 'labels' in inspect.signature(model_class.call).parameters.keys() else False)
            if not set(tf_inputs_dict_with_labels.keys()).symmetric_difference(tf_inputs_dict.keys()):
                tf_inputs_dict_with_labels = None
            tf_model = transformers.load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=tf_inputs_dict, allow_missing_keys=allow_missing_keys)
            pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=allow_missing_keys)
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)
            if tf_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict_with_labels)
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, 'pt_model.bin')
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path, allow_missing_keys=allow_missing_keys)
                tf_checkpoint_path = os.path.join(tmpdirname, 'tf_model.h5')
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, allow_missing_keys=allow_missing_keys)
            self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict)
            if tf_inputs_dict_with_labels:
                self.check_pt_tf_models(tf_model, pt_model, tf_inputs_dict_with_labels)

    @slow
    def test_compile_tf_model(self):
        if False:
            i = 10
            return i + 15
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes[:2]:
            model = model_class(config)
            functional_inputs = {key: tf.keras.Input(shape=val.shape[1:], dtype=val.dtype, name=key) for (key, val) in model.input_signature.items() if key in model.dummy_inputs}
            outputs_dict = model(functional_inputs)
            hidden_states = outputs_dict[0]
            functional_model = tf.keras.Model(inputs=functional_inputs, outputs=hidden_states)
            model_out = functional_model.predict(model.dummy_inputs)
            self.assertTrue(model_out is not None)
            with tempfile.TemporaryDirectory() as tmpdirname:
                functional_model.save(tmpdirname)

    def test_keyword_and_dict_args(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)
            outputs_dict = model(inputs)
            inputs_keywords = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            outputs_keywords = model(**inputs_keywords)
            output_dict = outputs_dict[0].numpy()
            output_keywords = outputs_keywords[0].numpy()
            self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-06)

    def test_attention_outputs(self):
        if False:
            print('Hello World!')
        if not self.has_attentions:
            self.skipTest(reason='Model does not output attentions')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        decoder_seq_length = getattr(self.model_tester, 'decoder_seq_length', self.model_tester.seq_length)
        encoder_seq_length = getattr(self.model_tester, 'encoder_seq_length', self.model_tester.seq_length)
        decoder_key_length = getattr(self.model_tester, 'key_length', decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, 'key_length', encoder_seq_length)

        def check_decoder_attentions_output(outputs):
            if False:
                i = 10
                return i + 15
            out_len = len(outputs)
            self.assertEqual(min(out_len % 2, out_len % 5), 0)
            decoder_attentions = outputs.decoder_attentions
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(decoder_attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length])

        def check_encoder_attentions_output(outputs):
            if False:
                print('Hello World!')
            attentions = [t.numpy() for t in (outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions)]
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length])
        for model_class in self.all_model_classes:
            inputs_dict['output_attentions'] = True
            config.output_hidden_states = False
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            out_len = len(outputs)
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)
            if self.is_encoder_decoder:
                model = model_class(config)
                outputs = model(self._prepare_for_class(inputs_dict, model_class))
                self.assertEqual(config.output_hidden_states, False)
                check_decoder_attentions_output(outputs)
            del inputs_dict['output_attentions']
            config.output_attentions = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)
            inputs_dict['output_attentions'] = True
            config.output_hidden_states = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(out_len + (2 if self.is_encoder_decoder else 1), len(outputs))
            self.assertEqual(model.config.output_hidden_states, True)
            check_encoder_attentions_output(outputs)

    def test_headmasking(self):
        if False:
            return 10
        if not self.test_head_masking:
            return
        random.Random().seed(42)
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        random.Random().seed()
        inputs_dict['output_attentions'] = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)

            def prepare_layer_head_mask(i, attention_heads, num_hidden_layers):
                if False:
                    i = 10
                    return i + 15
                if i == 0:
                    return tf.concat((tf.zeros(1, dtype=tf.float32), tf.ones(attention_heads - 1, dtype=tf.float32)), 0)
                elif i == num_hidden_layers - 1:
                    return tf.concat((tf.zeros(attention_heads - 1, dtype=tf.float32), tf.ones(1, dtype=tf.float32)), 0)
                else:
                    return tf.ones(attention_heads, dtype=tf.float32)
            head_mask = tf.stack([prepare_layer_head_mask(i, config.num_attention_heads, config.num_hidden_layers) for i in range(config.num_hidden_layers)], 0)
            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs['head_mask'] = head_mask
            if model.config.is_encoder_decoder:
                signature = inspect.signature(model.call)
                arg_names = [*signature.parameters.keys()]
                if 'decoder_head_mask' in arg_names:
                    inputs['decoder_head_mask'] = head_mask
                if 'cross_attn_head_mask' in arg_names:
                    inputs['cross_attn_head_mask'] = head_mask
            outputs = model(**inputs, return_dict=True)

            def check_attentions_validity(attentions):
                if False:
                    i = 10
                    return i + 15
                for t in attentions:
                    self.assertLess(tf.math.reduce_sum(tf.cast(tf.math.is_nan(t), tf.float32)).numpy(), (tf.size(t) / 4).numpy())
                attentions = [tf.where(tf.math.is_nan(t), 0.0, t) for t in attentions]
                self.assertAlmostEqual(tf.math.reduce_sum(attentions[0][..., 0, :, :]).numpy(), 0.0)
                self.assertNotEqual(tf.math.reduce_sum(attentions[0][..., -1, :, :]).numpy(), 0.0)
                if len(attentions) > 2:
                    self.assertNotEqual(tf.math.reduce_sum(attentions[1][..., 0, :, :]).numpy(), 0.0)
                self.assertAlmostEqual(tf.math.reduce_sum(attentions[-1][..., -2, :, :]).numpy(), 0.0)
                self.assertNotEqual(tf.math.reduce_sum(attentions[-1][..., -1, :, :]).numpy(), 0.0)
            if model.config.is_encoder_decoder:
                check_attentions_validity(outputs.encoder_attentions)
                check_attentions_validity(outputs.decoder_attentions)
                if 'cross_attn_head_mask' in arg_names:
                    check_attentions_validity(outputs.cross_attentions)
            else:
                check_attentions_validity(outputs.attentions)

    def test_hidden_states_output(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        def check_hidden_states_output(config, inputs_dict, model_class):
            if False:
                print('Hello World!')
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', self.model_tester.num_hidden_layers + 1)
            if model.config.is_encoder_decoder:
                encoder_hidden_states = outputs.encoder_hidden_states
                decoder_hidden_states = outputs.decoder_hidden_states
                self.assertEqual(config.output_attentions, False)
                self.assertEqual(len(encoder_hidden_states), expected_num_layers)
                self.assertListEqual(list(encoder_hidden_states[0].shape[-2:]), [self.model_tester.seq_length, self.model_tester.hidden_size])
                self.assertEqual(len(decoder_hidden_states), expected_num_layers)
                self.assertListEqual(list(decoder_hidden_states[0].shape[-2:]), [self.model_tester.seq_length, self.model_tester.hidden_size])
            else:
                hidden_states = outputs.hidden_states
                self.assertEqual(config.output_attentions, False)
                self.assertEqual(len(hidden_states), expected_num_layers)
                self.assertListEqual(list(hidden_states[0].shape[-2:]), [self.model_tester.seq_length, self.model_tester.hidden_size])
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(config, inputs_dict, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(config, inputs_dict, model_class)

    def test_model_common_attributes(self):
        if False:
            print('Hello World!')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        text_in_text_out_models = get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING) + get_values(TF_MODEL_FOR_MASKED_LM_MAPPING) + get_values(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING)
        speech_in_text_out_models = get_values(TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING)
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), tf.keras.layers.Layer)
            legacy_text_in_text_out = model.get_lm_head() is not None
            if model_class in text_in_text_out_models or legacy_text_in_text_out:
                out_embeddings = model.get_output_embeddings()
                self.assertIsInstance(out_embeddings, tf.keras.layers.Layer)
                bias = model.get_bias()
                if bias is not None:
                    self.assertIsInstance(bias, dict)
                    for (_, v) in bias.items():
                        self.assertIsInstance(v, tf.Variable)
            elif model_class in speech_in_text_out_models:
                out_embeddings = model.get_output_embeddings()
                self.assertIsInstance(out_embeddings, tf.keras.layers.Layer)
                bias = model.get_bias()
                self.assertIsNone(bias)
            else:
                out_embeddings = model.get_output_embeddings()
                assert out_embeddings is None
                bias = model.get_bias()
                self.assertIsNone(bias)

    def test_determinism(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            (first, second) = (model(self._prepare_for_class(inputs_dict, model_class), training=False)[0], model(self._prepare_for_class(inputs_dict, model_class), training=False)[0])
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-05)

    def test_model_outputs_equivalence(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            if False:
                return 10
            tuple_output = model(tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

            def recursive_check(tuple_object, dict_object):
                if False:
                    print('Hello World!')
                if isinstance(tuple_object, (List, Tuple)):
                    for (tuple_iterable_value, dict_iterable_value) in zip(tuple_object, dict_object):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif tuple_object is None:
                    return
                else:
                    self.assertTrue(all(tf.equal(tuple_object, dict_object)), msg=f'Tuple and dict output are not equal. Difference: {tf.math.reduce_max(tf.abs(tuple_object - dict_object))}')
                recursive_check(tuple_output, dict_output)
        for model_class in self.all_model_classes:
            model = model_class(config)
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {'output_hidden_states': True})
            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {'output_attentions': True})
            if 'labels' in inspect.signature(model.call).parameters.keys():
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs)
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {'output_hidden_states': True})
                if self.has_attentions:
                    tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    check_equivalence(model, tuple_inputs, dict_inputs, {'output_attentions': True})
                    tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    check_equivalence(model, tuple_inputs, dict_inputs, {'output_hidden_states': True, 'output_attentions': True})

    def test_inputs_embeds(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = copy.deepcopy(inputs_dict)
            if not self.is_encoder_decoder:
                input_ids = inputs['input_ids']
                del inputs['input_ids']
            else:
                encoder_input_ids = inputs['input_ids']
                decoder_input_ids = inputs.get('decoder_input_ids', encoder_input_ids)
                del inputs['input_ids']
                inputs.pop('decoder_input_ids', None)
            if not self.is_encoder_decoder:
                inputs['inputs_embeds'] = model.get_input_embeddings()(input_ids)
            else:
                inputs['inputs_embeds'] = model.get_input_embeddings()(encoder_input_ids)
                inputs['decoder_inputs_embeds'] = model.get_input_embeddings()(decoder_input_ids)
            inputs = self._prepare_for_class(inputs, model_class)
            model(inputs)

    def test_numpy_arrays_inputs(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        def prepare_numpy_arrays(inputs_dict):
            if False:
                while True:
                    i = 10
            inputs_np_dict = {}
            for (k, v) in inputs_dict.items():
                if tf.is_tensor(v):
                    inputs_np_dict[k] = v.numpy()
                else:
                    inputs_np_dict[k] = np.array(k)
            return inputs_np_dict
        for model_class in self.all_model_classes:
            model = model_class(config)
            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs_np = prepare_numpy_arrays(inputs)
            output_for_dict_input = model(inputs_np)
            output_for_kw_input = model(**inputs_np)
            self.assert_outputs_same(output_for_dict_input, output_for_kw_input)

    def test_valid_input_signature_and_dummies(self):
        if False:
            i = 10
            return i + 15
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            call_args = inspect.signature(model.call).parameters
            for key in model.input_signature:
                self.assertIn(key, call_args)
            for key in model.dummy_inputs:
                self.assertIn(key, call_args)

    def test_resize_token_embeddings(self):
        if False:
            return 10
        if not self.test_resize_embeddings:
            return
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        def _get_word_embedding_weight(model, embedding_layer):
            if False:
                print('Hello World!')
            if isinstance(embedding_layer, tf.keras.layers.Embedding):
                model.build()
                return embedding_layer.embeddings
            else:
                return model._get_word_embedding_weight(embedding_layer)
        for model_class in self.all_model_classes:
            for size in [config.vocab_size - 10, config.vocab_size + 10, None]:
                model = model_class(config=copy.deepcopy(config))
                old_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                old_bias = model.get_bias()
                old_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())
                model.resize_token_embeddings(size)
                new_input_embeddings = _get_word_embedding_weight(model, model.get_input_embeddings())
                new_bias = model.get_bias()
                new_output_embeddings = _get_word_embedding_weight(model, model.get_output_embeddings())
                assert_size = size if size is not None else config.vocab_size
                self.assertEqual(new_input_embeddings.shape[0], assert_size)
                models_equal = True
                for (p1, p2) in zip(old_input_embeddings.value(), new_input_embeddings.value()):
                    if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                        models_equal = False
                self.assertTrue(models_equal)
                if old_bias is not None and new_bias is not None:
                    for (old_weight, new_weight) in zip(old_bias.values(), new_bias.values()):
                        self.assertEqual(new_weight.shape[-1], assert_size)
                        models_equal = True
                        for (p1, p2) in zip(tf.squeeze(old_weight), tf.squeeze(new_weight)):
                            if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                                models_equal = False
                        self.assertTrue(models_equal)
                if old_output_embeddings is not None and new_output_embeddings is not None:
                    self.assertEqual(new_output_embeddings.shape[0], assert_size)
                    self.assertEqual(new_output_embeddings.shape[1], old_output_embeddings.shape[1])
                    models_equal = True
                    for (p1, p2) in zip(old_output_embeddings.value(), new_output_embeddings.value()):
                        if tf.math.reduce_sum(tf.math.abs(p1 - p2)) > 0:
                            models_equal = False
                    self.assertTrue(models_equal)

    @slow
    def test_save_load_after_resize_token_embeddings(self):
        if False:
            i = 10
            return i + 15
        if not self.test_resize_embeddings:
            return
        (config, original_inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            new_tokens_size = 10
            old_total_size = config.vocab_size
            new_total_size = old_total_size + new_tokens_size
            model = model_class(config=copy.deepcopy(config))
            model.build()
            model.resize_token_embeddings(new_total_size)
            inputs_dict = copy.deepcopy(original_inputs_dict)
            ids_feat_name = None
            if 'input_ids' in inputs_dict:
                ids_feat_name = 'input_ids'
            elif 'decoder_input_ids' in inputs_dict:
                ids_feat_name = 'decoder_input_ids'
            else:
                assert False, 'No input ids feature found in the inputs dict'
            new_vocab_input_ids = ids_tensor(inputs_dict[ids_feat_name].shape, new_tokens_size)
            new_vocab_input_ids += old_total_size
            inputs_dict[ids_feat_name] = new_vocab_input_ids
            if 'input_ids' in inputs_dict:
                inputs_dict['input_ids'] = new_vocab_input_ids
            if 'decoder_input_ids' in inputs_dict:
                inputs_dict['decoder_input_ids'] = new_vocab_input_ids
            prepared_inputs = self._prepare_for_class(inputs_dict, model_class)
            outputs = model(**prepared_inputs)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model = model_class.from_pretrained(tmpdirname)
                restored_model_outputs = model(**prepared_inputs)
                self.assert_outputs_same(restored_model_outputs, outputs)

    @unittest.skipIf(not is_tf_available() or len(tf.config.list_physical_devices('GPU')) == 0, reason='This test always passes on CPU.')
    def test_embeddings_out_of_bounds_raise_exception(self):
        if False:
            return 10
        if not self.test_resize_embeddings:
            return
        (config, original_inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config=config)
            inputs_dict = copy.deepcopy(original_inputs_dict)
            if 'input_ids' in inputs_dict:
                inputs_dict['input_ids'] = inputs_dict['input_ids'] * int(1000000000.0)
            if 'decoder_input_ids' in inputs_dict:
                inputs_dict['decoder_input_ids'] = inputs_dict['decoder_input_ids'] * int(1000000000.0)
            prepared_inputs = self._prepare_for_class(inputs_dict, model_class)
            with self.assertRaises(tf.errors.InvalidArgumentError):
                model(**prepared_inputs)

    def test_lm_head_model_random_no_beam_search_generate(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get('input_ids', None)
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            if config.bos_token_id is None:
                with self.assertRaises(ValueError):
                    model.generate(do_sample=True, max_length=5)
                self._check_generated_ids(model.generate(input_ids, do_sample=True))
            elif model_class.__name__ not in ['TFSpeech2TextForConditionalGeneration']:
                self._check_generated_ids(model.generate(do_sample=True, max_length=5))
            with self.assertRaises(ValueError):
                model.generate(input_ids, do_sample=False, num_return_sequences=2)
            self._check_generated_ids(model.generate(input_ids, do_sample=True, num_return_sequences=2))
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(input_ids, do_sample=True, bad_words_ids=bad_words_ids, num_return_sequences=2)
            generated_ids = output_tokens[:, input_ids.shape[-1]:]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    def test_lm_head_model_no_beam_search_generate_dict_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get('input_ids', None)
        if input_ids is None:
            input_ids = inputs_dict.get('input_features', None)
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            output_greedy = model.generate(input_ids, do_sample=False, output_scores=True, output_hidden_states=True, output_attentions=True, return_dict_in_generate=True)
            output_sample = model.generate(input_ids, do_sample=True, output_scores=True, output_hidden_states=True, output_attentions=True, return_dict_in_generate=True)
            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_greedy, TFGreedySearchEncoderDecoderOutput)
                self.assertIsInstance(output_sample, TFSampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_greedy, TFGreedySearchDecoderOnlyOutput)
                self.assertIsInstance(output_sample, TFSampleDecoderOnlyOutput)

    def test_lm_head_model_random_beam_search_generate(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get('input_ids', None)
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            if config.bos_token_id is None:
                self._check_generated_ids(model.generate(input_ids, do_sample=True, num_beams=2))
            else:
                self._check_generated_ids(model.generate(do_sample=True, max_length=5, num_beams=2))
            with self.assertRaises(ValueError):
                model.generate(input_ids, do_sample=False, num_return_sequences=3, num_beams=2)
            self._check_generated_ids(model.generate(input_ids, do_sample=True, num_beams=2, num_return_sequences=2))
            self._check_generated_ids(model.generate(input_ids, do_sample=False, num_beams=2, num_return_sequences=2))
            bad_words_ids = [self._generate_random_bad_tokens(1, model), self._generate_random_bad_tokens(2, model)]
            output_tokens = model.generate(input_ids, do_sample=False, bad_words_ids=bad_words_ids, num_beams=2, num_return_sequences=2)
            generated_ids = output_tokens[:, input_ids.shape[-1]:]
            self.assertFalse(self._check_match_tokens(generated_ids.numpy().tolist(), bad_words_ids))

    def test_lm_head_model_beam_search_generate_dict_outputs(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict.get('input_ids', None)
        if input_ids is None:
            input_ids = inputs_dict.get('input_features', None)
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            output_beam_search = model.generate(input_ids, num_beams=2, do_sample=False, output_scores=True, output_hidden_states=True, output_attentions=True, return_dict_in_generate=True)
            output_beam_sample = model.generate(input_ids, num_beams=2, do_sample=True, output_scores=True, output_hidden_states=True, output_attentions=True, return_dict_in_generate=True)
            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_beam_search, TFBeamSearchEncoderDecoderOutput)
                self.assertIsInstance(output_beam_sample, TFBeamSampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_beam_search, TFBeamSearchDecoderOnlyOutput)
                self.assertIsInstance(output_beam_sample, TFBeamSampleDecoderOnlyOutput)

    def test_loss_computation(self):
        if False:
            i = 10
            return i + 15
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            added_label_names = sorted(prepared_for_class.keys() - inputs_dict.keys(), reverse=True)
            if not added_label_names:
                continue
            added_label = prepared_for_class[added_label_names[0]]
            expected_loss_size = added_label.shape.as_list()[:1]
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            possible_input_names = {'input_ids', 'pixel_values', 'input_features', 'input_values'}
            input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
            model_input = prepared_for_class.pop(input_name)
            outputs = model(model_input, **prepared_for_class)
            if not isinstance(outputs, ModelOutput) or not hasattr(outputs, 'loss'):
                continue
            loss = outputs.loss
            self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            possible_input_names = {'input_ids', 'pixel_values', 'input_features', 'input_values'}
            input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
            model_input = prepared_for_class.pop(input_name)
            if 'labels' in prepared_for_class:
                labels = prepared_for_class['labels'].numpy()
                if len(labels.shape) > 1 and labels.shape[1] != 1:
                    labels[0] = -100
                    prepared_for_class['labels'] = tf.convert_to_tensor(labels)
                    loss = model(model_input, **prepared_for_class)[0]
                    self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])
                    self.assertTrue(not np.any(np.isnan(loss.numpy())))
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            loss = model(prepared_for_class)[0]
            self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            label_keys = prepared_for_class.keys() - inputs_dict.keys()
            signature = inspect.signature(model.call).parameters
            signature_names = list(signature.keys())
            tuple_index_mapping = {0: input_name}
            for label_key in label_keys:
                label_key_index = signature_names.index(label_key)
                tuple_index_mapping[label_key_index] = label_key
            sorted_tuple_index_mapping = sorted(tuple_index_mapping.items())
            list_input = []
            for name in signature_names:
                if name != 'kwargs':
                    list_input.append(signature[name].default)
            for (index, value) in sorted_tuple_index_mapping:
                list_input[index] = prepared_for_class[value]
            tuple_input = tuple(list_input)
            loss = model(tuple_input[:-1])[0]
            self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

    def check_keras_fit_results(self, val_loss1, val_loss2, atol=0.01, rtol=0.001):
        if False:
            while True:
                i = 10
        self.assertTrue(np.allclose(val_loss1, val_loss2, atol=atol, rtol=rtol))

    @slow
    def test_keras_fit(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
            prepared_for_class = {key: val for (key, val) in prepared_for_class.items() if key not in ('head_mask', 'decoder_head_mask', 'cross_attn_head_mask', 'return_loss')}
            if 'labels' in prepared_for_class and 'decoder_input_ids' in prepared_for_class:
                del prepared_for_class['decoder_input_ids']
            accuracy_classes = ['ForPreTraining', 'ForCausalLM', 'ForMaskedLM', 'ForQuestionAnswering', 'ForMultipleChoice', 'ForSequenceClassification', 'ForTokenClassification', 'ForNextSentencePrediction', 'LMHeadModel']
            for accuracy_class in accuracy_classes:
                if model.__class__.__name__.endswith(accuracy_class):
                    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
                    break
            else:
                metrics = []
            if hasattr(self.model_tester, 'batch_size'):
                sample_weight = tf.convert_to_tensor([0.5] * self.model_tester.batch_size, dtype=tf.float32)
            else:
                sample_weight = None
            outputs = model(prepared_for_class)
            if getattr(outputs, 'loss', None) is None:
                continue
            model_weights = model.get_weights()
            model.compile(optimizer=tf.keras.optimizers.SGD(0.0), run_eagerly=True, metrics=metrics)
            history1 = model.fit(prepared_for_class, validation_data=prepared_for_class, sample_weight=sample_weight, steps_per_epoch=1, validation_steps=1, shuffle=False)
            val_loss1 = history1.history['val_loss'][0]
            self.assertTrue(not isnan(val_loss1))
            accuracy1 = {key: val[0] for (key, val) in history1.history.items() if key.endswith('accuracy')}
            possible_label_cols = {'labels', 'label', 'label_ids', 'start_positions', 'start_position', 'end_positions', 'end_position', 'next_sentence_label'}
            label_names = possible_label_cols.intersection(set(prepared_for_class))
            if len(label_names) == 0:
                return
            labels = {key: val for (key, val) in prepared_for_class.items() if key in label_names}
            inputs_minus_labels = {key: val for (key, val) in prepared_for_class.items() if key not in label_names}
            self.assertGreater(len(inputs_minus_labels), 0)
            model.set_weights(model_weights)
            history2 = model.fit(inputs_minus_labels, labels, validation_data=(inputs_minus_labels, labels), sample_weight=sample_weight, steps_per_epoch=1, validation_steps=1, shuffle=False)
            val_loss2 = history2.history['val_loss'][0]
            self.assertTrue(not isnan(val_loss2))
            accuracy2 = {key: val[0] for (key, val) in history2.history.items() if key.endswith('accuracy')}
            self.check_keras_fit_results(val_loss1, val_loss2)
            self.assertEqual(history1.history.keys(), history2.history.keys())
            for key in history1.history.keys():
                if not key.startswith('val_'):
                    self.assertTrue('val_' + key in history1.history.keys(), 'Outputs differ in train/test step!')
            if metrics:
                self.assertTrue(len(accuracy1) == len(accuracy2) > 0, 'Missing metrics!')

    def test_int_support(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True if 'labels' in inspect.signature(model_class.call).parameters.keys() else False)
            if not any((tensor.dtype.is_integer for tensor in prepared_for_class.values() if isinstance(tensor, tf.Tensor))):
                return
            prepared_for_class = {key: tf.cast(tensor, tf.int64) if isinstance(tensor, tf.Tensor) and tensor.dtype.is_integer else tensor for (key, tensor) in prepared_for_class.items()}
            model = model_class(config)
            model(**prepared_for_class)
            int32_prepared_for_class = {key: tf.cast(tensor, tf.int32) if isinstance(tensor, tf.Tensor) and tensor.dtype.is_integer else tensor for (key, tensor) in prepared_for_class.items()}
            model(**int32_prepared_for_class)
            for (key, tensor) in model.dummy_inputs.items():
                self.assertTrue(isinstance(tensor, tf.Tensor) or tf.keras.backend.is_keras_tensor(tensor), 'Dummy inputs should be tf.Tensor!')
                if tensor.dtype.is_integer:
                    self.assertTrue(tensor.dtype == tf.int32, 'Integer dummy inputs should be tf.int32!')
            for (key, tensor_spec) in model.input_signature.items():
                if tensor_spec.dtype.is_integer:
                    self.assertTrue(tensor_spec.dtype == tf.int32, 'Input signatures should use tf.int32 for ints!')

    def test_generate_with_headmasking(self):
        if False:
            print('Hello World!')
        attention_names = ['encoder_attentions', 'decoder_attentions', 'cross_attentions']
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            if not config.is_encoder_decoder:
                continue
            head_masking = {'head_mask': tf.zeros((config.encoder_layers, config.encoder_attention_heads)), 'decoder_head_mask': tf.zeros((config.decoder_layers, config.decoder_attention_heads)), 'cross_attn_head_mask': tf.zeros((config.decoder_layers, config.decoder_attention_heads))}
            signature = inspect.signature(model.call)
            if set(head_masking.keys()) < {*signature.parameters.keys()}:
                continue
            for (attn_name, (name, mask)) in zip(attention_names, head_masking.items()):
                out = model.generate(inputs_dict['input_ids'], num_beams=1, max_length=inputs_dict['input_ids'] + 5, output_attentions=True, return_dict_in_generate=True, **{name: mask})
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([tf.reduce_sum(w).numpy() for w in attn_weights]), 0.0)

    def test_load_with_mismatched_shapes(self):
        if False:
            return 10
        if not self.test_mismatched_shapes:
            return
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if model_class not in get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                continue
            with self.subTest(msg=f'Testing {model_class}'):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    _ = model(**inputs)
                    model.save_pretrained(tmp_dir)
                    with self.assertRaises(ValueError):
                        new_model = TFAutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42)
                    with self.assertRaises(ValueError):
                        new_model_without_prefix = TFAutoModel.from_pretrained(tmp_dir, vocab_size=10)
                    logger = logging.get_logger('transformers.modeling_tf_utils')
                    with CaptureLogger(logger) as cl:
                        new_model = TFAutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42, ignore_mismatched_sizes=True)
                    self.assertIn('the shapes did not match', cl.out)
                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)
                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = TFAutoModel.from_pretrained(tmp_dir, vocab_size=10, ignore_mismatched_sizes=True)
                    self.assertIn('the shapes did not match', cl.out)
                    input_ids = ids_tensor((2, 8), 10)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    def test_model_main_input_name(self):
        if False:
            i = 10
            return i + 15
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, 'call'))
            observed_main_input_name = list(model_signature.parameters.keys())[1]
            self.assertEqual(model_class.main_input_name, observed_main_input_name)

    def test_dataset_conversion(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class, return_labels=False)
            if 'labels' in tf_inputs_dict:
                return
            tf_inputs_dict = {key: val for (key, val) in tf_inputs_dict.items() if 'head_mask' not in key and isinstance(val, tf.Tensor)}
            tf_inputs_dict['extra_unwanted_column'] = list(tf_inputs_dict.values())[0]
            input_dataset = Dataset.from_dict(tf_inputs_dict)
            tf_dataset = model.prepare_tf_dataset(input_dataset, batch_size=len(input_dataset), drop_remainder=False, shuffle=False)
            test_batch = next(iter(tf_dataset))
            if isinstance(test_batch, tf.Tensor):
                self.assertEqual(len(test_batch), len(input_dataset))
            elif isinstance(test_batch, dict):
                self.assertEqual(len(test_batch), len(input_dataset.features) - 1)
                self.assertNotIn('extra_unwanted_column', test_batch)
                for tensor in test_batch.values():
                    self.assertTrue(isinstance(tensor, tf.Tensor))
                    self.assertEqual(len(tensor), len(input_dataset))
            model(test_batch, training=False)
            if 'labels' in inspect.signature(model_class.call).parameters.keys():
                tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                if 'labels' not in tf_inputs_dict:
                    return
                tf_inputs_dict = {key: val for (key, val) in tf_inputs_dict.items() if 'head_mask' not in key}
                tf_inputs_dict['extra_unwanted_column'] = list(tf_inputs_dict.values())[0]
                input_dataset = Dataset.from_dict(tf_inputs_dict)
                tf_dataset = model.prepare_tf_dataset(input_dataset, batch_size=len(input_dataset), drop_remainder=False, shuffle=False)
                (test_batch, test_batch_labels) = next(iter(tf_dataset))
                self.assertGreater(len(test_batch_labels), 0)
                feature_columns = 1 if isinstance(test_batch, tf.Tensor) else len(test_batch)
                label_columns = 1 if isinstance(test_batch_labels, tf.Tensor) else len(test_batch_labels)
                self.assertEqual(feature_columns + label_columns, len(input_dataset.features) - 1)
                if isinstance(test_batch, dict):
                    self.assertNotIn('extra_unwanted_column', test_batch)
                if isinstance(test_batch_labels, dict):
                    self.assertNotIn('extra_unwanted_column', test_batch_labels)
                model.compile(optimizer='sgd', run_eagerly=True)
                model.train_on_batch(test_batch, test_batch_labels)

    def _test_xla_generate(self, **generate_kwargs):
        if False:
            return 10

        def _generate_and_check_results(model, inputs_dict):
            if False:
                return 10
            if 'input_ids' in inputs_dict:
                inputs = inputs_dict['input_ids']
                if model.generation_config.pad_token_id is not None:
                    if config.pad_token_id == 0:
                        new_pad_token = model.generation_config.pad_token_id + 1
                    else:
                        new_pad_token = model.generation_config.pad_token_id - 1
                else:
                    new_pad_token = None
                inputs = tf.where(inputs != model.generation_config.pad_token_id, inputs, new_pad_token)
            elif 'input_features' in inputs_dict:
                inputs = inputs_dict['input_features']
            else:
                raise ValueError('No valid generate input found in inputs_dict')
            generated = model.generate(inputs, **generate_kwargs).numpy()
            generate_xla = tf.function(model.generate, jit_compile=True)
            generated_xla = generate_xla(inputs, **generate_kwargs).numpy()
            diff = [[], []]
            for (_generated, _generated_xla) in zip(generated.tolist(), generated_xla.tolist()):
                if _generated != _generated_xla:
                    diff[0].append(_generated)
                    diff[1].append(_generated_xla)
            ratio = len(diff[0]) / len(generated)
            if ratio > 0.1 or (len(diff[0]) > 0 and len(generated) < 10):
                self.assertListEqual(diff[0], diff[1])
        for model_class in self.all_generative_model_classes:
            (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
            config.eos_token_id = None
            config.do_sample = False
            for var_name in ['max_position_embeddings', 'max_target_positions']:
                attr = getattr(config, var_name, None)
                if attr is not None and attr < generate_kwargs['max_new_tokens']:
                    try:
                        setattr(config, var_name, generate_kwargs['max_new_tokens'])
                    except NotImplementedError:
                        pass
            model = model_class(config)
            if model.supports_xla_generation:
                _generate_and_check_results(model, inputs_dict)
            else:
                with self.assertRaises(ValueError):
                    _generate_and_check_results(model, inputs_dict)

    def test_xla_generate_fast(self):
        if False:
            return 10
        '\n        Basic quick test for generate-compatible classes that confirms that XLA-generated tokens are the same as their\n        non XLA counterparts.\n\n        Either the model supports XLA generation and passes the inner test, or it raises an appropriate exception\n        '
        self._test_xla_generate(num_beams=1, num_return_sequences=1, max_new_tokens=3)

    @slow
    def test_xla_generate_contrastive(self):
        if False:
            while True:
                i = 10
        '\n        Slow and challenging version of `test_xla_generate_fast` for contrastive search -- contrastive search directly\n        manipulates the model cache and other outputs, and this test ensures that they are in a valid format that is\n        also supported by XLA.\n\n        Either the model supports XLA generation and passes the inner test, or it raises an appropriate exception\n        '
        self._test_xla_generate(num_beams=1, num_return_sequences=1, max_new_tokens=16, penalty_alpha=0.5, top_k=4)

    @slow
    def test_xla_generate_slow(self):
        if False:
            i = 10
            return i + 15
        '\n        Slow and challenging version of `test_xla_generate_fast` -- this test asks for several long sequences using\n        beam search, with and without XLA. The two outputs should match, and a failure in this test indicates that the\n        model may need further analysis if it is to be used for XLA generation.\n\n        Either the model supports XLA generation and passes the inner test, or it raises an appropriate exception\n        '
        self._test_xla_generate(num_beams=8, num_return_sequences=2, max_new_tokens=128)

    def _generate_random_bad_tokens(self, num_bad_tokens, model):
        if False:
            print('Hello World!')
        special_tokens = []
        if model.config.bos_token_id is not None:
            special_tokens.append(model.config.bos_token_id)
        if model.config.pad_token_id is not None:
            special_tokens.append(model.config.pad_token_id)
        if model.config.eos_token_id is not None:
            special_tokens.append(model.config.eos_token_id)
        bad_tokens = []
        while len(bad_tokens) < num_bad_tokens:
            token = tf.squeeze(ids_tensor((1, 1), self.model_tester.vocab_size), 0).numpy()[0]
            if token not in special_tokens:
                bad_tokens.append(token)
        return bad_tokens

    def _check_generated_ids(self, output_ids):
        if False:
            for i in range(10):
                print('nop')
        for token_id in output_ids[0].numpy().tolist():
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.model_tester.vocab_size)

    def _check_match_tokens(self, generated_ids, bad_words_ids):
        if False:
            while True:
                i = 10
        for bad_word_ids in bad_words_ids:
            for generated_ids_slice in generated_ids:
                for i in range(len(bad_word_ids), len(generated_ids_slice)):
                    if generated_ids_slice[i - len(bad_word_ids):i] == bad_word_ids:
                        return True
        return False

def ids_tensor(shape, vocab_size, rng=None, name=None, dtype=None):
    if False:
        i = 10
        return i + 15
    'Creates a random int32 tensor of the shape within the vocab size.'
    if rng is None:
        rng = random.Random()
    total_dims = 1
    for dim in shape:
        total_dims *= dim
    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))
    output = tf.constant(values, shape=shape, dtype=dtype if dtype is not None else tf.int32)
    return output

def random_attention_mask(shape, rng=None, name=None, dtype=None):
    if False:
        i = 10
        return i + 15
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None, dtype=dtype)
    attn_mask = tf.concat([attn_mask[:, :-1], tf.ones_like(attn_mask[:, -1:], dtype=dtype)], axis=-1)
    return attn_mask

def floats_tensor(shape, scale=1.0, rng=None, name=None, dtype=None):
    if False:
        print('Hello World!')
    'Creates a random float32 tensor'
    if rng is None:
        rng = random.Random()
    total_dims = 1
    for dim in shape:
        total_dims *= dim
    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)
    return tf.reshape(tf.constant(values, dtype=dtype if dtype is not None else tf.float32), shape=shape)