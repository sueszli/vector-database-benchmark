""" Testing suite for the PyTorch Owlv2 model. """
import inspect
import os
import tempfile
import unittest
import numpy as np
import requests
from transformers import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
from transformers.testing_utils import require_torch, require_torch_accelerator, require_torch_fp16, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import Owlv2ForObjectDetection, Owlv2Model, Owlv2TextModel, Owlv2VisionModel
    from transformers.models.owlv2.modeling_owlv2 import OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import OwlViTProcessor

class Owlv2VisionModelTester:

    def __init__(self, parent, batch_size=12, image_size=32, patch_size=2, num_channels=3, is_training=True, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37, dropout=0.1, attention_dropout=0.1, initializer_range=0.02, scope=None):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return (config, pixel_values)

    def get_config(self):
        if False:
            while True:
                i = 10
        return Owlv2VisionConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, dropout=self.dropout, attention_dropout=self.attention_dropout, initializer_range=self.initializer_range)

    def create_and_check_model(self, config, pixel_values):
        if False:
            while True:
                i = 10
        model = Owlv2VisionModel(config=config).to(torch_device)
        model.eval()
        pixel_values = pixel_values.to(torch.float32)
        with torch.no_grad():
            result = model(pixel_values)
        num_patches = (self.image_size // self.patch_size) ** 2
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class Owlv2VisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as OWLV2 does not use input_ids,
    inputs_embeds, attention_mask and seq_length.
    """
    all_model_classes = (Owlv2VisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = Owlv2VisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Owlv2VisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    @unittest.skip(reason='OWLV2 does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        if False:
            return 10
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='OwlV2 does not support training yet')
    def test_training(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='OwlV2 does not support training yet')
    def test_training_gradient_checkpointing(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='Owlv2VisionModel has no base class and is not available in MODEL_MAPPING')
    def test_save_load_fast_init_from_base(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Owlv2VisionModel has no base class and is not available in MODEL_MAPPING')
    def test_save_load_fast_init_to_base(self):
        if False:
            return 10
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_name in OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Owlv2VisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

class Owlv2TextModelTester:

    def __init__(self, parent, batch_size=12, num_queries=4, seq_length=16, is_training=True, use_input_mask=True, use_labels=True, vocab_size=99, hidden_size=64, num_hidden_layers=12, num_attention_heads=4, intermediate_size=37, dropout=0.1, attention_dropout=0.1, max_position_embeddings=16, initializer_range=0.02, scope=None):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        input_ids = ids_tensor([self.batch_size * self.num_queries, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size * self.num_queries, self.seq_length])
        if input_mask is not None:
            (num_text, seq_length) = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(num_text,))
            for (idx, start_index) in enumerate(rnd_start_indices):
                input_mask[idx, :start_index] = 1
                input_mask[idx, start_index:] = 0
        config = self.get_config()
        return (config, input_ids, input_mask)

    def get_config(self):
        if False:
            return 10
        return Owlv2TextConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, dropout=self.dropout, attention_dropout=self.attention_dropout, max_position_embeddings=self.max_position_embeddings, initializer_range=self.initializer_range)

    def create_and_check_model(self, config, input_ids, input_mask):
        if False:
            while True:
                i = 10
        model = Owlv2TextModel(config=config).to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size * self.num_queries, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size * self.num_queries, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

@require_torch
class Owlv2TextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Owlv2TextModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.model_tester = Owlv2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Owlv2TextConfig, hidden_size=37)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='OwlV2 does not support training yet')
    def test_training(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='OwlV2 does not support training yet')
    def test_training_gradient_checkpointing(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='OWLV2 does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='Owlv2TextModel has no base class and is not available in MODEL_MAPPING')
    def test_save_load_fast_init_from_base(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Owlv2TextModel has no base class and is not available in MODEL_MAPPING')
    def test_save_load_fast_init_to_base(self):
        if False:
            return 10
        pass

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_name in OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Owlv2TextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

class Owlv2ModelTester:

    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if False:
            return 10
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}
        self.parent = parent
        self.text_model_tester = Owlv2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Owlv2VisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training
        self.text_config = self.text_model_tester.get_config().to_dict()
        self.vision_config = self.vision_model_tester.get_config().to_dict()

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        (text_config, input_ids, attention_mask) = self.text_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values) = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return (config, input_ids, attention_mask, pixel_values)

    def get_config(self):
        if False:
            while True:
                i = 10
        return Owlv2Config.from_text_vision_configs(self.text_config, self.vision_config, projection_dim=64)

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        if False:
            return 10
        model = Owlv2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        image_logits_size = (self.vision_model_tester.batch_size, self.text_model_tester.batch_size * self.text_model_tester.num_queries)
        text_logits_size = (self.text_model_tester.batch_size * self.text_model_tester.num_queries, self.vision_model_tester.batch_size)
        self.parent.assertEqual(result.logits_per_image.shape, image_logits_size)
        self.parent.assertEqual(result.logits_per_text.shape, text_logits_size)

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask, 'return_loss': False}
        return (config, inputs_dict)

@require_torch
class Owlv2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Owlv2Model,) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': Owlv2Model, 'zero-shot-object-detection': Owlv2ForObjectDetection} if is_torch_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_tester = Owlv2ModelTester(self)

    def test_model(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='Hidden_states is tested in individual model tests')
    def test_hidden_states_output(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='Inputs_embeds is tested in individual model tests')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='Retain_grad is tested in individual model tests')
    def test_retain_grad_hidden_states_attentions(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='Owlv2Model does not have input/output embeddings')
    def test_model_common_attributes(self):
        if False:
            while True:
                i = 10
        pass

    def test_initialization(self):
        if False:
            print('Hello World!')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for (name, param) in model.named_parameters():
                if param.requires_grad:
                    if name == 'logit_scale':
                        self.assertAlmostEqual(param.data.item(), np.log(1 / 0.07), delta=0.001, msg=f'Parameter {name} of model {model_class} seems not properly initialized')
                    else:
                        self.assertIn(((param.data.mean() * 1000000000.0).round() / 1000000000.0).item(), [0.0, 1.0], msg=f'Parameter {name} of model {model_class} seems not properly initialized')

    def _create_and_check_torchscript(self, config, inputs_dict):
        if False:
            for i in range(10):
                print('nop')
        if not self.test_torchscript:
            return
        configs_no_init = _config_zero_init(config)
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init).to(torch_device)
            model.eval()
            try:
                input_ids = inputs_dict['input_ids']
                pixel_values = inputs_dict['pixel_values']
                traced_model = torch.jit.trace(model, (input_ids, pixel_values))
            except RuntimeError:
                self.fail("Couldn't trace module.")
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, 'traced_model.pt')
                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")
                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")
            loaded_model = loaded_model.to(torch_device)
            loaded_model.eval()
            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()
            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]
            loaded_model_state_dict = {key: value for (key, value) in loaded_model_state_dict.items() if key not in non_persistent_buffers}
            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))
            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for (i, model_buffer) in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break
                self.assertTrue(found_buffer)
                model_buffers.pop(i)
            models_equal = True
            for (layer_name, p1) in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False
            self.assertTrue(models_equal)

    def test_load_vision_text_config(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Owlv2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = Owlv2TextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        for model_name in OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Owlv2Model.from_pretrained(model_name)
            self.assertIsNotNone(model)

class Owlv2ForObjectDetectionTester:

    def __init__(self, parent, is_training=True):
        if False:
            return 10
        self.parent = parent
        self.text_model_tester = Owlv2TextModelTester(parent)
        self.vision_model_tester = Owlv2VisionModelTester(parent)
        self.is_training = is_training
        self.text_config = self.text_model_tester.get_config().to_dict()
        self.vision_config = self.vision_model_tester.get_config().to_dict()

    def prepare_config_and_inputs(self):
        if False:
            while True:
                i = 10
        (text_config, input_ids, attention_mask) = self.text_model_tester.prepare_config_and_inputs()
        (vision_config, pixel_values) = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return (config, pixel_values, input_ids, attention_mask)

    def get_config(self):
        if False:
            return 10
        return Owlv2Config.from_text_vision_configs(self.text_config, self.vision_config, projection_dim=64)

    def create_and_check_model(self, config, pixel_values, input_ids, attention_mask):
        if False:
            print('Hello World!')
        model = Owlv2ForObjectDetection(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pred_boxes_size = (self.vision_model_tester.batch_size, (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2, 4)
        pred_logits_size = (self.vision_model_tester.batch_size, (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2, 4)
        pred_class_embeds_size = (self.vision_model_tester.batch_size, (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2, self.text_model_tester.hidden_size)
        self.parent.assertEqual(result.pred_boxes.shape, pred_boxes_size)
        self.parent.assertEqual(result.logits.shape, pred_logits_size)
        self.parent.assertEqual(result.class_embeds.shape, pred_class_embeds_size)

    def prepare_config_and_inputs_for_common(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, input_ids, attention_mask) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask}
        return (config, inputs_dict)

@require_torch
class Owlv2ForObjectDetectionTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Owlv2ForObjectDetection,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_tester = Owlv2ForObjectDetectionTester(self)

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='Hidden_states is tested in individual model tests')
    def test_hidden_states_output(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='Inputs_embeds is tested in individual model tests')
    def test_inputs_embeds(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='Retain_grad is tested in individual model tests')
    def test_retain_grad_hidden_states_attentions(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Owlv2Model does not have input/output embeddings')
    def test_model_common_attributes(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='Test_initialization is tested in individual model tests')
    def test_initialization(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='Test_forward_signature is tested in individual model tests')
    def test_forward_signature(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='Test_save_load_fast_init_from_base is tested in individual model tests')
    def test_save_load_fast_init_from_base(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='OwlV2 does not support training yet')
    def test_training(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='OwlV2 does not support training yet')
    def test_training_gradient_checkpointing(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        if False:
            while True:
                i = 10
        pass

    def _create_and_check_torchscript(self, config, inputs_dict):
        if False:
            return 10
        if not self.test_torchscript:
            return
        configs_no_init = _config_zero_init(config)
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init).to(torch_device)
            model.eval()
            try:
                input_ids = inputs_dict['input_ids']
                pixel_values = inputs_dict['pixel_values']
                traced_model = torch.jit.trace(model, (input_ids, pixel_values))
            except RuntimeError:
                self.fail("Couldn't trace module.")
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, 'traced_model.pt')
                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")
                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")
            loaded_model = loaded_model.to(torch_device)
            loaded_model.eval()
            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()
            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]
            loaded_model_state_dict = {key: value for (key, value) in loaded_model_state_dict.items() if key not in non_persistent_buffers}
            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))
            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for (i, model_buffer) in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break
                self.assertTrue(found_buffer)
                model_buffers.pop(i)
            models_equal = True
            for (layer_name, p1) in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False
            self.assertTrue(models_equal)

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Owlv2ForObjectDetection.from_pretrained(model_name)
            self.assertIsNotNone(model)

def prepare_img():
    if False:
        for i in range(10):
            print('nop')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@require_vision
@require_torch
class Owlv2ModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference(self):
        if False:
            i = 10
            return i + 15
        model_name = 'google/owlv2-base-patch16'
        model = Owlv2Model.from_pretrained(model_name).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)
        image = prepare_img()
        inputs = processor(text=[['a photo of a cat', 'a photo of a dog']], images=image, max_length=16, padding='max_length', return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertEqual(outputs.logits_per_image.shape, torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])))
        self.assertEqual(outputs.logits_per_text.shape, torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])))
        expected_logits = torch.tensor([[-6.2229, -8.2601]], device=torch_device)
        self.assertTrue(torch.allclose(outputs.logits_per_image, expected_logits, atol=0.001))

    @slow
    def test_inference_object_detection(self):
        if False:
            print('Hello World!')
        model_name = 'google/owlv2-base-patch16'
        model = Owlv2ForObjectDetection.from_pretrained(model_name).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)
        image = prepare_img()
        inputs = processor(text=[['a photo of a cat', 'a photo of a dog']], images=image, max_length=16, padding='max_length', return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))
        expected_slice_logits = torch.tensor([[-21.4139, -21.613], [-19.0084, -19.5491], [-20.9592, -21.383]])
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=0.0001))
        expected_slice_boxes = torch.tensor([[0.2413, 0.0519, 0.4533], [0.1395, 0.0457, 0.2507], [0.233, 0.0505, 0.4277]]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=0.0001))

    @slow
    def test_inference_one_shot_object_detection(self):
        if False:
            i = 10
            return i + 15
        model_name = 'google/owlv2-base-patch16'
        model = Owlv2ForObjectDetection.from_pretrained(model_name).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)
        image = prepare_img()
        query_image = prepare_img()
        inputs = processor(images=image, query_images=query_image, max_length=16, padding='max_length', return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)
        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))
        expected_slice_boxes = torch.tensor([[0.2413, 0.0519, 0.4533], [0.1395, 0.0457, 0.2507], [0.233, 0.0505, 0.4277]]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.target_pred_boxes[0, :3, :3], expected_slice_boxes, atol=0.0001))

    @slow
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_one_shot_object_detection_fp16(self):
        if False:
            for i in range(10):
                print('nop')
        model_name = 'google/owlv2-base-patch16'
        model = Owlv2ForObjectDetection.from_pretrained(model_name, torch_dtype=torch.float16).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)
        image = prepare_img()
        query_image = prepare_img()
        inputs = processor(images=image, query_images=query_image, max_length=16, padding='max_length', return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)
        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))