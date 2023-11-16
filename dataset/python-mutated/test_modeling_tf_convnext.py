""" Testing suite for the TensorFlow ConvNext model. """
from __future__ import annotations
import inspect
import unittest
from typing import List, Tuple
from transformers import ConvNextConfig
from transformers.testing_utils import require_tf, require_vision, slow
from transformers.utils import cached_property, is_tf_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_tf_available():
    import tensorflow as tf
    from transformers import TFConvNextForImageClassification, TFConvNextModel
if is_vision_available():
    from PIL import Image
    from transformers import ConvNextImageProcessor

class TFConvNextModelTester:

    def __init__(self, parent, batch_size=13, image_size=32, num_channels=3, num_stages=4, hidden_sizes=[10, 20, 30, 40], depths=[2, 2, 3, 2], is_training=True, use_labels=True, intermediate_size=37, hidden_act='gelu', type_sequence_label_size=10, initializer_range=0.02, num_labels=3, scope=None):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return ConvNextConfig(num_channels=self.num_channels, hidden_sizes=self.hidden_sizes, depths=self.depths, num_stages=self.num_stages, hidden_act=self.hidden_act, is_decoder=False, initializer_range=self.initializer_range)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            return 10
        model = TFConvNextModel(config=config)
        result = model(pixel_values, training=False)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.hidden_sizes[-1], self.image_size // 32, self.image_size // 32))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        if False:
            while True:
                i = 10
        config.num_labels = self.type_sequence_label_size
        model = TFConvNextForImageClassification(config)
        result = model(pixel_values, labels=labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_tf
class TFConvNextModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ConvNext does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (TFConvNextModel, TFConvNextForImageClassification) if is_tf_available() else ()
    pipeline_model_mapping = {'feature-extraction': TFConvNextModel, 'image-classification': TFConvNextForImageClassification} if is_tf_available() else {}
    test_pruning = False
    test_onnx = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = TFConvNextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ConvNextConfig, has_text_modality=False, hidden_size=37)

    @unittest.skip(reason='ConvNext does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skipIf(not is_tf_available() or len(tf.config.list_physical_devices('GPU')) == 0, reason='TF does not support backprop for grouped convolutions on CPU.')
    @slow
    def test_keras_fit(self):
        if False:
            return 10
        super().test_keras_fit()

    @unittest.skip(reason='ConvNext does not support input and output embeddings')
    def test_model_common_attributes(self):
        if False:
            return 10
        pass

    def test_forward_signature(self):
        if False:
            while True:
                i = 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skipIf(not is_tf_available() or len(tf.config.list_physical_devices('GPU')) == 0, reason='TF does not support backprop for grouped convolutions on CPU.')
    def test_dataset_conversion(self):
        if False:
            while True:
                i = 10
        super().test_dataset_conversion()

    def test_hidden_states_output(self):
        if False:
            print('Hello World!')

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                return 10
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            expected_num_stages = self.model_tester.num_stages
            self.assertEqual(len(hidden_states), expected_num_stages + 1)
            self.assertListEqual(list(hidden_states[0].shape[-2:]), [self.model_tester.image_size // 4, self.model_tester.image_size // 4])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_model_outputs_equivalence(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            if False:
                for i in range(10):
                    print('nop')
            tuple_output = model(tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

            def recursive_check(tuple_object, dict_object):
                if False:
                    i = 10
                    return i + 15
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
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {'output_hidden_states': True})
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {'output_hidden_states': True})

    def test_for_image_classification(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            print('Hello World!')
        model = TFConvNextModel.from_pretrained('facebook/convnext-tiny-224')
        self.assertIsNotNone(model)

def prepare_img():
    if False:
        while True:
            i = 10
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_tf
@require_vision
class TFConvNextModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            for i in range(10):
                print('nop')
        return ConvNextImageProcessor.from_pretrained('facebook/convnext-tiny-224') if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        if False:
            print('Hello World!')
        model = TFConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224')
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='tf')
        outputs = model(**inputs)
        expected_shape = tf.TensorShape((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = tf.constant([-0.026, -0.4739, 0.1911])
        tf.debugging.assert_near(outputs.logits[0, :3], expected_slice, atol=0.0001)