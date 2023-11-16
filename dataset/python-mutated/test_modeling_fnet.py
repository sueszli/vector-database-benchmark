""" Testing suite for the PyTorch FNet model. """
import unittest
from typing import Dict, List, Tuple
from transformers import FNetConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_tokenizers, require_torch, slow, torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers import MODEL_FOR_PRETRAINING_MAPPING, FNetForMaskedLM, FNetForMultipleChoice, FNetForNextSentencePrediction, FNetForPreTraining, FNetForQuestionAnswering, FNetForSequenceClassification, FNetForTokenClassification, FNetModel, FNetTokenizerFast
    from transformers.models.fnet.modeling_fnet import FNET_PRETRAINED_MODEL_ARCHIVE_LIST, FNetBasicFourierTransform, is_scipy_available

class FNetConfigTester(ConfigTester):

    def create_and_test_config_common_properties(self):
        if False:
            i = 10
            return i + 15
        config = self.config_class(**self.inputs_dict)
        if self.has_text_modality:
            self.parent.assertTrue(hasattr(config, 'vocab_size'))
        self.parent.assertTrue(hasattr(config, 'hidden_size'))
        self.parent.assertTrue(hasattr(config, 'num_hidden_layers'))

class FNetModelTester:

    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, use_token_type_ids=True, use_labels=True, vocab_size=99, hidden_size=32, num_hidden_layers=2, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, scope=None):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        if False:
            i = 10
            return i + 15
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)
        config = self.get_config()
        return (config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels)

    def get_config(self):
        if False:
            print('Hello World!')
        return FNetConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, max_position_embeddings=self.max_position_embeddings, type_vocab_size=self.type_vocab_size, initializer_range=self.initializer_range, tpu_short_seq_length=self.seq_length)

    @require_torch
    def create_and_check_fourier_transform(self, config):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = floats_tensor([self.batch_size, self.seq_length, config.hidden_size])
        transform = FNetBasicFourierTransform(config)
        fftn_output = transform(hidden_states)
        config.use_tpu_fourier_optimizations = True
        if is_scipy_available():
            transform = FNetBasicFourierTransform(config)
            dft_output = transform(hidden_states)
        config.max_position_embeddings = 4097
        transform = FNetBasicFourierTransform(config)
        fft_output = transform(hidden_states)
        if is_scipy_available():
            self.parent.assertTrue(torch.allclose(fftn_output[0][0], dft_output[0][0], atol=0.0001))
            self.parent.assertTrue(torch.allclose(fft_output[0][0], dft_output[0][0], atol=0.0001))
        self.parent.assertTrue(torch.allclose(fftn_output[0][0], fft_output[0][0], atol=0.0001))

    def create_and_check_model(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = FNetModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_pretraining(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = FNetForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels, next_sentence_label=sequence_labels)
        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertEqual(result.seq_relationship_logits.shape, (self.batch_size, 2))

    def create_and_check_for_masked_lm(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        if False:
            return 10
        model = FNetForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_next_sentence_prediction(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = FNetForNextSentencePrediction(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, next_sentence_label=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 2))

    def create_and_check_for_question_answering(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        if False:
            print('Hello World!')
        model = FNetForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, start_positions=sequence_labels, end_positions=sequence_labels)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        if False:
            return 10
        config.num_labels = self.num_labels
        model = FNetForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        if False:
            while True:
                i = 10
        config.num_labels = self.num_labels
        model = FNetForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        if False:
            i = 10
            return i + 15
        config.num_choices = self.num_choices
        model = FNetForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(multiple_choice_inputs_ids, token_type_ids=multiple_choice_token_type_ids, labels=choice_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        if False:
            return 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids}
        return (config, inputs_dict)

@require_torch
class FNetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (FNetModel, FNetForPreTraining, FNetForMaskedLM, FNetForNextSentencePrediction, FNetForMultipleChoice, FNetForQuestionAnswering, FNetForSequenceClassification, FNetForTokenClassification) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': FNetModel, 'fill-mask': FNetForMaskedLM, 'question-answering': FNetForQuestionAnswering, 'text-classification': FNetForSequenceClassification, 'token-classification': FNetForTokenClassification, 'zero-shot': FNetForSequenceClassification} if is_torch_available() else {}
    test_pruning = False
    test_head_masking = False
    test_pruning = False

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            print('Hello World!')
        if pipeline_test_casse_name == 'QAPipelineTests' and (not tokenizer_name.endswith('Fast')):
            return True
        return False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        if False:
            for i in range(10):
                print('nop')
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels:
            if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict['labels'] = torch.zeros((self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device)
                inputs_dict['next_sentence_label'] = torch.zeros(self.model_tester.batch_size, dtype=torch.long, device=torch_device)
        return inputs_dict

    def test_attention_outputs(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing(self):
        if False:
            while True:
                i = 10
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip(reason='This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124')
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_model_outputs_equivalence(self):
        if False:
            while True:
                i = 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            if False:
                while True:
                    i = 10
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            if False:
                while True:
                    i = 10
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if False:
                        return 10
                    if isinstance(tuple_object, (List, Tuple)):
                        for (tuple_iterable_value, dict_iterable_value) in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for (tuple_iterable_value, dict_iterable_value) in zip(tuple_object.values(), dict_object.values()):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(torch.allclose(set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-05), msg=f'Tuple and dict output are not equal. Difference: {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`: {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}.')
                recursive_check(tuple_output, dict_output)
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {'output_hidden_states': True})

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            return 10
        (config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)
        inputs = self._prepare_for_class(inputs_dict, model_class)
        outputs = model(**inputs)
        output = outputs[0]
        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()
        output.flatten()[0].backward(retain_graph=True)
        self.assertIsNotNone(hidden_states.grad)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = FNetModelTester(self)
        self.config_tester = FNetConfigTester(self, config_class=FNetConfig, hidden_size=37)

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    def test_model(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_masked_lm(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        if False:
            print('Hello World!')
        for model_name in FNET_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = FNetModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

@require_torch
class FNetModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_for_masked_lm(self):
        if False:
            i = 10
            return i + 15
        '\n        For comparison:\n        1. Modify the pre-training model `__call__` to skip computing metrics and return masked_lm_output like so:\n            ```\n            ...\n            sequence_output, pooled_output = EncoderModel(\n            self.config, random_seed=self.random_seed, name="encoder")(\n                input_ids, input_mask, type_ids, deterministic=deterministic)\n\n            masked_lm_output = nn.Dense(\n                self.config.d_emb,\n                kernel_init=default_kernel_init,\n                name="predictions_dense")(\n                    sequence_output)\n            masked_lm_output = nn.gelu(masked_lm_output)\n            masked_lm_output = nn.LayerNorm(\n                epsilon=LAYER_NORM_EPSILON, name="predictions_layer_norm")(\n                    masked_lm_output)\n            masked_lm_logits = layers.OutputProjection(\n                kernel=self._get_embedding_table(), name="predictions_output")(\n                    masked_lm_output)\n\n            next_sentence_logits = layers.OutputProjection(\n                n_out=2, kernel_init=default_kernel_init, name="classification")(\n                    pooled_output)\n\n            return masked_lm_logits\n            ...\n            ```\n        2. Run the following:\n            >>> import jax.numpy as jnp\n            >>> import sentencepiece as spm\n            >>> from flax.training import checkpoints\n            >>> from f_net.models import PreTrainingModel\n            >>> from f_net.configs.pretraining import get_config, ModelArchitecture\n\n            >>> pretrained_params = checkpoints.restore_checkpoint(\'./f_net/f_net_checkpoint\', None) # Location of original checkpoint\n            >>> pretrained_config  = get_config()\n            >>> pretrained_config.model_arch = ModelArchitecture.F_NET\n\n            >>> vocab_filepath = "./f_net/c4_bpe_sentencepiece.model" # Location of the sentence piece model\n            >>> tokenizer = spm.SentencePieceProcessor()\n            >>> tokenizer.Load(vocab_filepath)\n            >>> with pretrained_config.unlocked():\n            >>>     pretrained_config.vocab_size = tokenizer.GetPieceSize()\n            >>> tokens = jnp.array([[0, 1, 2, 3, 4, 5]])\n            >>> type_ids = jnp.zeros_like(tokens, dtype="i4")\n            >>> attention_mask = jnp.ones_like(tokens) # Dummy. This gets deleted inside the model.\n\n            >>> flax_pretraining_model = PreTrainingModel(pretrained_config)\n            >>> pretrained_model_params = freeze(pretrained_params[\'target\'])\n            >>> flax_model_outputs = flax_pretraining_model.apply({"params": pretrained_model_params}, tokens, attention_mask, type_ids, None, None, None, None, deterministic=True)\n            >>> masked_lm_logits[:, :3, :3]\n        '
        model = FNetForMaskedLM.from_pretrained('google/fnet-base')
        model.to(torch_device)
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device=torch_device)
        with torch.no_grad():
            output = model(input_ids)[0]
        vocab_size = 32000
        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor([[[-1.7819, -7.7384, -7.5002], [-3.4746, -8.5943, -7.7762], [-3.2052, -9.0771, -8.3468]]], device=torch_device)
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=0.0001))

    @slow
    @require_tokenizers
    def test_inference_long_sentence(self):
        if False:
            return 10
        tokenizer = FNetTokenizerFast.from_pretrained('google/fnet-base')
        inputs = tokenizer('the man worked as a [MASK].', 'this is his [MASK].', return_tensors='pt', padding='max_length', max_length=512)
        torch.testing.assert_allclose(inputs['input_ids'], torch.tensor([[4, 13, 283, 2479, 106, 8, 6, 845, 5, 168, 65, 367, 6, 845, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]))
        inputs = {k: v.to(torch_device) for (k, v) in inputs.items()}
        model = FNetForMaskedLM.from_pretrained('google/fnet-base')
        model.to(torch_device)
        logits = model(**inputs).logits
        predictions_mask_1 = tokenizer.decode(logits[0, 6].topk(5).indices)
        predictions_mask_2 = tokenizer.decode(logits[0, 12].topk(5).indices)
        self.assertEqual(predictions_mask_1.split(' '), ['man', 'child', 'teacher', 'woman', 'model'])
        self.assertEqual(predictions_mask_2.split(' '), ['work', 'wife', 'job', 'story', 'name'])

    @slow
    def test_inference_for_next_sentence_prediction(self):
        if False:
            print('Hello World!')
        model = FNetForNextSentencePrediction.from_pretrained('google/fnet-base')
        model.to(torch_device)
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device=torch_device)
        with torch.no_grad():
            output = model(input_ids)[0]
        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor([[-0.2234, -0.0226]], device=torch_device)
        self.assertTrue(torch.allclose(output, expected_slice, atol=0.0001))

    @slow
    def test_inference_model(self):
        if False:
            i = 10
            return i + 15
        model = FNetModel.from_pretrained('google/fnet-base')
        model.to(torch_device)
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device=torch_device)
        with torch.no_grad():
            output = model(input_ids)[0]
        expected_shape = torch.Size((1, 6, model.config.hidden_size))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor([[[4.1541, -0.1051, -0.1667], [-0.9144, 0.2939, -0.0086], [-0.8472, -0.7281, 0.0256]]], device=torch_device)
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=0.0001))