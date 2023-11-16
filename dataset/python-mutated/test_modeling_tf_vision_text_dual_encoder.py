""" Testing suite for the PyTorch VisionTextDualEncoder model. """
from __future__ import annotations
import collections
import tempfile
import unittest
import numpy as np
from transformers.testing_utils import require_tf, require_vision, slow
from transformers.utils import is_tf_available, is_vision_available
from ...test_modeling_tf_common import floats_tensor, ids_tensor, random_attention_mask
from ..bert.test_modeling_tf_bert import TFBertModelTester
from ..clip.test_modeling_tf_clip import TFCLIPVisionModelTester
from ..deit.test_modeling_tf_deit import TFDeiTModelTester
from ..roberta.test_modeling_tf_roberta import TFRobertaModelTester
from ..vit.test_modeling_tf_vit import TFViTModelTester
if is_tf_available():
    from transformers import TFBertModel, TFCLIPVisionModel, TFDeiTModel, TFRobertaModel, TFVisionTextDualEncoderModel, TFViTModel, VisionTextDualEncoderConfig
if is_vision_available():
    from PIL import Image
    from transformers import VisionTextDualEncoderProcessor

def to_2tuple(x):
    if False:
        return 10
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

@require_tf
class TFVisionTextDualEncoderMixin:

    def get_vision_text_model(self, config, text_config):
        if False:
            for i in range(10):
                print('nop')
        pass

    def prepare_config_and_inputs(self):
        if False:
            return 10
        pass

    def get_pretrained_model_and_inputs(self):
        if False:
            i = 10
            return i + 15
        pass

    def check_model_from_pretrained_configs(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)
        model = TFVisionTextDualEncoderModel(config)
        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        self.assertEqual(output['text_embeds'].shape, (input_ids.shape[0], config.projection_dim))
        self.assertEqual(output['image_embeds'].shape, (pixel_values.shape[0], config.projection_dim))

    def check_vision_text_dual_encoder_model(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        if False:
            while True:
                i = 10
        (vision_model, text_model) = self.get_vision_text_model(vision_config, text_config)
        model = TFVisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        self.assertEqual(output['text_embeds'].shape, (input_ids.shape[0], model.config.projection_dim))
        self.assertEqual(output['image_embeds'].shape, (pixel_values.shape[0], model.config.projection_dim))

    def check_vision_text_dual_encoder_from_pretrained(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        if False:
            print('Hello World!')
        (vision_model, text_model) = self.get_vision_text_model(vision_config, text_config)
        kwargs = {'vision_model': vision_model, 'text_model': text_model}
        model = TFVisionTextDualEncoderModel.from_vision_text_pretrained(**kwargs)
        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        self.assertEqual(output['text_embeds'].shape, (input_ids.shape[0], model.config.projection_dim))
        self.assertEqual(output['image_embeds'].shape, (pixel_values.shape[0], model.config.projection_dim))

    def check_save_load(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (vision_model, text_model) = self.get_vision_text_model(vision_config, text_config)
        model = TFVisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        out_1 = output[0].numpy()
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = TFVisionTextDualEncoderModel.from_pretrained(tmpdirname)
            after_output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            out_2 = after_output[0].numpy()
            max_diff = np.amax(np.abs(out_2 - out_1))
            self.assertLessEqual(max_diff, 1e-05)

    def check_vision_text_output_attention(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        if False:
            while True:
                i = 10
        (vision_model, text_model) = self.get_vision_text_model(vision_config, text_config)
        model = TFVisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_attentions=True)
        vision_attentions = output.vision_model_output.attentions
        self.assertEqual(len(vision_attentions), vision_config.num_hidden_layers)
        image_size = to_2tuple(vision_model.config.image_size)
        patch_size = to_2tuple(vision_model.config.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        self.assertEqual(vision_attentions[0].shape[-3:], (vision_config.num_attention_heads, seq_len, seq_len))
        text_attentions = output.text_model_output.attentions
        self.assertEqual(len(text_attentions), text_config.num_hidden_layers)
        self.assertEqual(text_attentions[0].shape[-3:], (text_config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1]))

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        if False:
            while True:
                i = 10
        diff = np.abs(a - b).max()
        self.assertLessEqual(diff, tol, f'Difference between torch and flax is {diff} (>= {tol}).')

    def test_vision_text_dual_encoder_model(self):
        if False:
            print('Hello World!')
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_dual_encoder_model(**inputs_dict)

    def test_model_from_pretrained_configs(self):
        if False:
            print('Hello World!')
        inputs_dict = self.prepare_config_and_inputs()
        self.check_model_from_pretrained_configs(**inputs_dict)

    def test_vision_text_dual_encoder_from_pretrained(self):
        if False:
            print('Hello World!')
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_dual_encoder_from_pretrained(**inputs_dict)

    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        inputs_dict = self.prepare_config_and_inputs()
        self.check_save_load(**inputs_dict)

    def test_vision_text_output_attention(self):
        if False:
            while True:
                i = 10
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_output_attention(**inputs_dict)

    @slow
    def test_real_model_save_load_from_pretrained(self):
        if False:
            while True:
                i = 10
        (model_2, inputs) = self.get_pretrained_model_and_inputs()
        outputs = model_2(**inputs)
        out_2 = outputs[0].numpy()
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model_2.save_pretrained(tmp_dirname)
            model_1 = TFVisionTextDualEncoderModel.from_pretrained(tmp_dirname)
            after_outputs = model_1(**inputs)
            out_1 = after_outputs[0].numpy()
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-05)

@require_tf
class TFViTBertModelTest(TFVisionTextDualEncoderMixin, unittest.TestCase):

    def get_pretrained_model_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        model = TFVisionTextDualEncoderModel.from_vision_text_pretrained('hf-internal-testing/tiny-random-vit', 'hf-internal-testing/tiny-random-bert')
        batch_size = 13
        pixel_values = floats_tensor([batch_size, model.vision_model.config.num_channels, model.vision_model.config.image_size, model.vision_model.config.image_size])
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask}
        return (model, inputs)

    def get_vision_text_model(self, vision_config, text_config):
        if False:
            while True:
                i = 10
        vision_model = TFViTModel(vision_config, name='vision_model')
        text_model = TFBertModel(text_config, name='text_model')
        return (vision_model, text_model)

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        vit_model_tester = TFViTModelTester(self)
        bert_model_tester = TFBertModelTester(self)
        vision_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values, _) = vision_config_and_inputs
        (text_config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = text_config_and_inputs
        return {'text_config': text_config, 'vision_config': vision_config, 'pixel_values': pixel_values, 'attention_mask': input_mask, 'input_ids': input_ids, 'text_token_type_ids': token_type_ids, 'text_sequence_labels': sequence_labels, 'text_token_labels': token_labels, 'text_choice_labels': choice_labels}

@require_tf
class TFDeiTRobertaModelTest(TFVisionTextDualEncoderMixin, unittest.TestCase):

    def get_pretrained_model_and_inputs(self):
        if False:
            i = 10
            return i + 15
        model = TFVisionTextDualEncoderModel.from_vision_text_pretrained('Rocketknight1/tiny-random-deit-tf', 'hf-internal-testing/tiny-random-roberta')
        batch_size = 13
        pixel_values = floats_tensor([batch_size, model.vision_model.config.num_channels, model.vision_model.config.image_size, model.vision_model.config.image_size])
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask}
        return (model, inputs)

    def check_vision_text_output_attention(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (vision_model, text_model) = self.get_vision_text_model(vision_config, text_config)
        model = TFVisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_attentions=True)
        vision_attentions = output.vision_model_output.attentions
        self.assertEqual(len(vision_attentions), vision_config.num_hidden_layers)
        image_size = to_2tuple(vision_model.config.image_size)
        patch_size = to_2tuple(vision_model.config.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        seq_len = num_patches + 2
        self.assertEqual(vision_attentions[0].shape[-3:], (vision_config.num_attention_heads, seq_len, seq_len))
        text_attentions = output.text_model_output.attentions
        self.assertEqual(len(text_attentions), text_config.num_hidden_layers)
        self.assertEqual(text_attentions[0].shape[-3:], (text_config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1]))

    def get_vision_text_model(self, vision_config, text_config):
        if False:
            while True:
                i = 10
        vision_model = TFDeiTModel(vision_config, name='vision_model')
        text_model = TFRobertaModel(text_config, name='text_model')
        return (vision_model, text_model)

    def prepare_config_and_inputs(self):
        if False:
            print('Hello World!')
        vit_model_tester = TFDeiTModelTester(self)
        bert_model_tester = TFRobertaModelTester(self)
        vision_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values, _) = vision_config_and_inputs
        (text_config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = text_config_and_inputs
        return {'text_config': text_config, 'vision_config': vision_config, 'pixel_values': pixel_values, 'attention_mask': input_mask, 'input_ids': input_ids, 'text_token_type_ids': token_type_ids, 'text_sequence_labels': sequence_labels, 'text_token_labels': token_labels, 'text_choice_labels': choice_labels}

@require_tf
class TFCLIPVisionBertModelTest(TFVisionTextDualEncoderMixin, unittest.TestCase):

    def get_pretrained_model_and_inputs(self):
        if False:
            while True:
                i = 10
        model = TFVisionTextDualEncoderModel.from_vision_text_pretrained('Rocketknight1/tiny-random-clip-tf', 'hf-internal-testing/tiny-random-bert')
        batch_size = 13
        pixel_values = floats_tensor([batch_size, model.vision_model.config.num_channels, model.vision_model.config.image_size, model.vision_model.config.image_size])
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask}
        return (model, inputs)

    def get_vision_text_model(self, vision_config, text_config):
        if False:
            for i in range(10):
                print('nop')
        vision_model = TFCLIPVisionModel(vision_config, name='vision_model')
        text_model = TFBertModel(text_config, name='text_model')
        return (vision_model, text_model)

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        clip_model_tester = TFCLIPVisionModelTester(self)
        bert_model_tester = TFBertModelTester(self)
        vision_config_and_inputs = clip_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values) = vision_config_and_inputs
        (text_config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = text_config_and_inputs
        return {'text_config': text_config, 'vision_config': vision_config, 'pixel_values': pixel_values, 'attention_mask': input_mask, 'input_ids': input_ids, 'text_token_type_ids': token_type_ids, 'text_sequence_labels': sequence_labels, 'text_token_labels': token_labels, 'text_choice_labels': choice_labels}

@require_vision
@require_tf
class TFVisionTextDualEncoderIntegrationTest(unittest.TestCase):

    @slow
    def test_inference(self):
        if False:
            while True:
                i = 10
        model = TFVisionTextDualEncoderModel.from_pretrained('clip-italian/clip-italian', logit_scale_init_value=1.0, from_pt=True)
        processor = VisionTextDualEncoderProcessor.from_pretrained('clip-italian/clip-italian')
        image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
        inputs = processor(text=['una foto di un gatto', 'una foto di un cane'], images=image, padding=True, return_tensors='np')
        outputs = model(**inputs)
        self.assertEqual(outputs.logits_per_image.shape, (inputs.pixel_values.shape[0], inputs.input_ids.shape[0]))
        self.assertEqual(outputs.logits_per_text.shape, (inputs.input_ids.shape[0], inputs.pixel_values.shape[0]))
        expected_logits = np.array([[1.2284727, 0.3104122]])
        self.assertTrue(np.allclose(outputs.logits_per_image.numpy(), expected_logits, atol=0.001))