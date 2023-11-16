""" Testing suite for the PyTorch EfficientFormer model. """
import inspect
import unittest
import warnings
from typing import List
from transformers import EfficientFormerConfig
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, MODEL_MAPPING, EfficientFormerForImageClassification, EfficientFormerForImageClassificationWithTeacher, EfficientFormerModel
    from transformers.models.efficientformer.modeling_efficientformer import EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST
if is_vision_available():
    from PIL import Image
    from transformers import EfficientFormerImageProcessor

class EfficientFormerModelTester:

    def __init__(self, parent, batch_size: int=13, image_size: int=64, patch_size: int=2, embed_dim: int=3, num_channels: int=3, is_training: bool=True, use_labels: bool=True, hidden_size: int=128, hidden_sizes=[16, 32, 64, 128], num_hidden_layers: int=7, num_attention_heads: int=4, intermediate_size: int=37, hidden_act: str='gelu', hidden_dropout_prob: float=0.1, attention_probs_dropout_prob: float=0.1, type_sequence_label_size: int=10, initializer_range: float=0.02, encoder_stride: int=2, num_attention_outputs: int=1, dim: int=128, depths: List[int]=[2, 2, 2, 2], resolution: int=2, mlp_expansion_ratio: int=2):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.encoder_stride = encoder_stride
        self.num_attention_outputs = num_attention_outputs
        self.embed_dim = embed_dim
        self.seq_length = embed_dim + 1
        self.resolution = resolution
        self.depths = depths
        self.hidden_sizes = hidden_sizes
        self.dim = dim
        self.mlp_expansion_ratio = mlp_expansion_ratio

    def prepare_config_and_inputs(self):
        if False:
            return 10
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        config = self.get_config()
        return (config, pixel_values, labels)

    def get_config(self):
        if False:
            return 10
        return EfficientFormerConfig(image_size=self.image_size, patch_size=self.patch_size, num_channels=self.num_channels, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, is_decoder=False, initializer_range=self.initializer_range, encoder_stride=self.encoder_stride, resolution=self.resolution, depths=self.depths, hidden_sizes=self.hidden_sizes, dim=self.dim, mlp_expansion_ratio=self.mlp_expansion_ratio)

    def create_and_check_model(self, config, pixel_values, labels):
        if False:
            return 10
        model = EfficientFormerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        if False:
            i = 10
            return i + 15
        config.num_labels = self.type_sequence_label_size
        model = EfficientFormerForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))
        config.num_channels = 1
        model = EfficientFormerForImageClassification(config)
        model.to(torch_device)
        model.eval()
        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, labels) = config_and_inputs
        inputs_dict = {'pixel_values': pixel_values}
        return (config, inputs_dict)

@require_torch
class EfficientFormerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as EfficientFormer does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """
    all_model_classes = (EfficientFormerModel, EfficientFormerForImageClassificationWithTeacher, EfficientFormerForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': EfficientFormerModel, 'image-classification': (EfficientFormerForImageClassification, EfficientFormerForImageClassificationWithTeacher)} if is_torch_available() else {}
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        if False:
            return 10
        self.model_tester = EfficientFormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EfficientFormerConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        if False:
            return 10
        self.config_tester.run_common_tests()

    @unittest.skip(reason='EfficientFormer does not use inputs_embeds')
    def test_inputs_embeds(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip(reason='EfficientFormer does not support input and output embeddings')
    def test_model_common_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_forward_signature(self):
        if False:
            print('Hello World!')
        (config, _) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        if False:
            i = 10
            return i + 15

        def check_hidden_states_output(inputs_dict, config, model_class):
            if False:
                return 10
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            expected_num_layers = getattr(self.model_tester, 'expected_num_hidden_layers', self.model_tester.num_hidden_layers + 1)
            self.assertEqual(len(hidden_states), expected_num_layers)
            if hasattr(self.model_tester, 'encoder_seq_length'):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, 'chunk_length') and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length
            self.assertListEqual(list(hidden_states[-1].shape[-2:]), [seq_length, self.model_tester.hidden_size])
            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states
                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, 'seq_length', None)
                decoder_seq_length = getattr(self.model_tester, 'decoder_seq_length', seq_len)
                self.assertListEqual(list(hidden_states[-1].shape[-2:]), [decoder_seq_length, self.model_tester.hidden_size])
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict['output_hidden_states'] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            del inputs_dict['output_hidden_states']
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        if False:
            return 10
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels:
            if model_class.__name__ == 'EfficientFormerForImageClassificationWithTeacher':
                del inputs_dict['labels']
        return inputs_dict

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason='EfficientFormer does not implement masked image modeling yet')
    def test_for_masked_image_modeling(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_image_modeling(*config_and_inputs)

    def test_for_image_classification(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_training(self):
        if False:
            while True:
                i = 10
        if not self.model_tester.is_training:
            return
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        for model_class in self.all_model_classes:
            if model_class in get_values(MODEL_MAPPING) or model_class.__name__ == 'EfficientFormerForImageClassificationWithTeacher':
                continue
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_problem_types(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        problem_types = [{'title': 'multi_label_classification', 'num_labels': 2, 'dtype': torch.float}, {'title': 'single_label_classification', 'num_labels': 1, 'dtype': torch.long}, {'title': 'regression', 'num_labels': 1, 'dtype': torch.float}]
        for model_class in self.all_model_classes:
            if model_class not in [*get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING)] or model_class.__name__ == 'EfficientFormerForImageClassificationWithTeacher':
                continue
            for problem_type in problem_types:
                with self.subTest(msg=f"Testing {model_class} with {problem_type['title']}"):
                    config.problem_type = problem_type['title']
                    config.num_labels = problem_type['num_labels']
                    model = model_class(config)
                    model.to(torch_device)
                    model.train()
                    inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    if problem_type['num_labels'] > 1:
                        inputs['labels'] = inputs['labels'].unsqueeze(1).repeat(1, problem_type['num_labels'])
                    inputs['labels'] = inputs['labels'].to(problem_type['dtype'])
                    with warnings.catch_warnings(record=True) as warning_list:
                        loss = model(**inputs).loss
                    for w in warning_list:
                        if 'Using a target size that is different to the input size' in str(w.message):
                            raise ValueError(f'Something is going wrong in the regression problem: intercepted {w.message}')
                    loss.backward()

    @slow
    def test_model_from_pretrained(self):
        if False:
            return 10
        for model_name in EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = EfficientFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_attention_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = getattr(self.model_tester, 'seq_length', None)
        encoder_seq_length = getattr(self.model_tester, 'encoder_seq_length', seq_len)
        encoder_key_length = getattr(self.model_tester, 'key_length', encoder_seq_length)
        chunk_length = getattr(self.model_tester, 'chunk_length', None)
        if chunk_length is not None and hasattr(self.model_tester, 'num_hashes'):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes
        for model_class in self.all_model_classes:
            inputs_dict['output_attentions'] = True
            inputs_dict['output_hidden_states'] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_attention_outputs)
            del inputs_dict['output_attentions']
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_attention_outputs)
            if chunk_length is not None:
                self.assertListEqual(list(attentions[0].shape[-4:]), [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length])
            else:
                self.assertListEqual(list(attentions[0].shape[-3:]), [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length])

def prepare_img():
    if False:
        print('Hello World!')
    image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
    return image

@require_torch
@require_vision
class EfficientFormerModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        if False:
            for i in range(10):
                print('nop')
        return EfficientFormerImageProcessor.from_pretrained('snap-research/efficientformer-l1-300') if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        if False:
            for i in range(10):
                print('nop')
        model = EfficientFormerForImageClassification.from_pretrained('snap-research/efficientformer-l1-300').to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = (1, 1000)
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.0555, 0.4825, -0.0852]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0][:3], expected_slice, atol=0.0001))

    @slow
    def test_inference_image_classification_head_with_teacher(self):
        if False:
            return 10
        model = EfficientFormerForImageClassificationWithTeacher.from_pretrained('snap-research/efficientformer-l1-300').to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors='pt').to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_shape = (1, 1000)
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.1312, 0.4353, -1.0499]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0][:3], expected_slice, atol=0.0001))