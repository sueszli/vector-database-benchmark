""" Testing suite for the PyTorch TrOCR model. """
import unittest
from transformers import TrOCRConfig
from transformers.testing_utils import is_torch_available, require_torch, torch_device
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from transformers.models.trocr.modeling_trocr import TrOCRDecoder, TrOCRForCausalLM

@require_torch
class TrOCRStandaloneDecoderModelTester:

    def __init__(self, parent, vocab_size=99, batch_size=13, d_model=16, decoder_seq_length=7, is_training=True, is_decoder=True, use_attention_mask=True, use_cache=False, use_labels=True, decoder_start_token_id=2, decoder_ffn_dim=32, decoder_layers=2, decoder_attention_heads=4, max_position_embeddings=30, pad_token_id=0, bos_token_id=1, eos_token_id=2, scope=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.decoder_seq_length = decoder_seq_length
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.num_hidden_layers = decoder_layers
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.num_attention_heads = decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 2
        self.decoder_attention_idx = 1

    def prepare_config_and_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)
        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)
        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)
        config = TrOCRConfig(vocab_size=self.vocab_size, d_model=self.d_model, decoder_layers=self.decoder_layers, decoder_ffn_dim=self.decoder_ffn_dim, decoder_attention_heads=self.decoder_attention_heads, eos_token_id=self.eos_token_id, bos_token_id=self.bos_token_id, use_cache=self.use_cache, pad_token_id=self.pad_token_id, decoder_start_token_id=self.decoder_start_token_id, max_position_embeddings=self.max_position_embeddings)
        return (config, input_ids, attention_mask, lm_labels)

    def create_and_check_decoder_model_past(self, config, input_ids, attention_mask, lm_labels):
        if False:
            print('Hello World!')
        config.use_cache = True
        model = TrOCRDecoder(config=config).to(torch_device).eval()
        input_ids = input_ids[:2]
        input_ids[input_ids == 0] += 1
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)
        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)
        past_key_values = outputs['past_key_values']
        next_tokens = ids_tensor((2, 1), config.vocab_size - 1) + 1
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        output_from_no_past = model(next_input_ids)['last_hidden_state']
        output_from_past = model(next_tokens, past_key_values=past_key_values)['last_hidden_state']
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=0.001)

    def prepare_config_and_inputs_for_common(self):
        if False:
            return 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, lm_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return (config, inputs_dict)

@require_torch
class TrOCRStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TrOCRDecoder, TrOCRForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (TrOCRForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = {'text-generation': TrOCRForCausalLM} if is_torch_available() else {}
    fx_compatible = True
    test_pruning = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = TrOCRStandaloneDecoderModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=TrOCRConfig)

    def test_inputs_embeds(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_save_load_fast_init_from_base(self):
        if False:
            return 10
        pass

    def test_save_load_fast_init_to_base(self):
        if False:
            print('Hello World!')
        pass

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    def test_decoder_model_past(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            return 10
        return

    @unittest.skip("The model doesn't support left padding")
    def test_left_padding_compatibility(self):
        if False:
            i = 10
            return i + 15
        pass