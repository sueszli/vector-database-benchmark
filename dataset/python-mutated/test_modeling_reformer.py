import unittest
from transformers import ReformerConfig, is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, require_torch_fp16, require_torch_multi_gpu, slow, torch_device
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
if is_torch_available():
    import torch
    from torch import nn
    from transformers import REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST, ReformerForMaskedLM, ReformerForQuestionAnswering, ReformerForSequenceClassification, ReformerLayer, ReformerModel, ReformerModelWithLMHead, ReformerTokenizer

class ReformerModelTester:

    def __init__(self, parent, batch_size=13, seq_length=32, is_training=True, is_decoder=True, use_input_mask=True, use_labels=True, vocab_size=32, attention_head_size=16, hidden_size=32, num_attention_heads=2, local_attn_chunk_length=4, local_num_chunks_before=1, local_num_chunks_after=0, num_buckets=None, num_hashes=1, lsh_attn_chunk_length=None, lsh_num_chunks_before=None, lsh_num_chunks_after=None, chunk_size_lm_head=0, chunk_size_feed_forward=0, feed_forward_size=32, hidden_act='gelu', hidden_dropout_prob=0.1, local_attention_probs_dropout_prob=0.1, lsh_attention_probs_dropout_prob=None, max_position_embeddings=512, initializer_range=0.02, axial_norm_std=1.0, layer_norm_eps=1e-12, axial_pos_embds=True, axial_pos_shape=[4, 8], axial_pos_embds_dim=[16, 16], attn_layers=['local', 'local', 'local', 'local'], pad_token_id=0, eos_token_id=2, scope=None, hash_seed=0, num_labels=2):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.is_decoder = is_decoder
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.attention_head_size = attention_head_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = len(attn_layers) if attn_layers is not None else 0
        self.local_attn_chunk_length = local_attn_chunk_length
        self.local_num_chunks_after = local_num_chunks_after
        self.local_num_chunks_before = local_num_chunks_before
        self.num_hashes = num_hashes
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.axial_pos_embds = axial_pos_embds
        self.axial_pos_shape = tuple(axial_pos_shape)
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        self.axial_norm_std = axial_norm_std
        self.chunk_size_lm_head = chunk_size_lm_head
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.scope = scope
        self.attn_layers = attn_layers
        self.pad_token_id = pad_token_id
        self.hash_seed = hash_seed
        attn_chunk_length = local_attn_chunk_length if local_attn_chunk_length is not None else lsh_attn_chunk_length
        num_chunks_after = local_num_chunks_after if local_num_chunks_after is not None else lsh_num_chunks_after
        num_chunks_before = local_num_chunks_before if local_num_chunks_before is not None else lsh_num_chunks_before
        self.encoder_seq_length = seq_length // attn_chunk_length + (self.seq_length % attn_chunk_length != 0)
        self.key_length = (num_chunks_before + num_chunks_after + 1) * attn_chunk_length
        self.chunk_length = attn_chunk_length
        self.num_labels = num_labels

    def prepare_config_and_inputs(self):
        if False:
            return 10
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        choice_labels = None
        if self.use_labels:
            choice_labels = ids_tensor([self.batch_size], 2)
        config = self.get_config()
        return (config, input_ids, input_mask, choice_labels)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return ReformerConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads, feed_forward_size=self.feed_forward_size, hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob, local_attention_probs_dropout_prob=self.local_attention_probs_dropout_prob, lsh_attention_probs_dropout_prob=self.lsh_attention_probs_dropout_prob, max_position_embeddings=self.max_position_embeddings, is_decoder=self.is_decoder, axial_pos_embds=self.axial_pos_embds, axial_pos_shape=self.axial_pos_shape, axial_pos_embds_dim=self.axial_pos_embds_dim, local_attn_chunk_length=self.local_attn_chunk_length, local_num_chunks_after=self.local_num_chunks_after, local_num_chunks_before=self.local_num_chunks_before, num_hashes=self.num_hashes, num_buckets=self.num_buckets, lsh_attn_chunk_length=self.lsh_attn_chunk_length, lsh_num_chunks_after=self.lsh_num_chunks_after, lsh_num_chunks_before=self.lsh_num_chunks_before, attn_layers=self.attn_layers, pad_token_id=self.pad_token_id, hash_seed=self.hash_seed)

    def get_pipeline_config(self):
        if False:
            return 10
        config = self.get_config()
        config.vocab_size = 100
        config.max_position_embeddings = 100
        config.axial_pos_shape = (4, 25)
        config.is_decoder = False
        return config

    def create_and_check_reformer_model(self, config, input_ids, input_mask, choice_labels):
        if False:
            i = 10
            return i + 15
        model = ReformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, 2 * self.hidden_size))

    def create_and_check_reformer_model_with_lm_backward(self, config, input_ids, input_mask, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_training:
            return
        config.is_decoder = False
        config.lsh_num_chunks_after = 1
        model = ReformerForMaskedLM(config=config)
        model.to(torch_device)
        model.train()
        loss = model(input_ids, attention_mask=input_mask, labels=input_ids)['loss']
        loss.backward()

    def create_and_check_reformer_with_lm(self, config, input_ids, input_mask, choice_labels):
        if False:
            print('Hello World!')
        config.lsh_num_chunks_after = 0
        config.is_decoder = True
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_reformer_with_mlm(self, config, input_ids, input_mask, choice_labels):
        if False:
            while True:
                i = 10
        config.is_decoder = False
        model = ReformerForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_reformer_model_with_attn_mask(self, config, input_ids, input_mask, choice_labels, is_decoder=False):
        if False:
            print('Hello World!')
        config.axial_pos_embds = False
        config.is_decoder = is_decoder
        if self.lsh_attn_chunk_length is not None:
            config.lsh_attn_chunk_length = self.seq_length
        model = ReformerModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            embedding = model.embeddings.position_embeddings.embedding
            embedding.weight = nn.Parameter(torch.zeros(embedding.weight.shape).to(torch_device))
            embedding.weight.requires_grad = False
        half_seq_len = self.seq_length // 2
        roll = self.chunk_length
        half_input_ids = input_ids[:, :half_seq_len]
        attn_mask = torch.cat([torch.ones_like(half_input_ids), torch.zeros_like(half_input_ids)], dim=-1)
        input_ids_padded = torch.cat([half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)], dim=-1)
        input_ids_roll = torch.cat([half_input_ids, ids_tensor((self.batch_size, half_seq_len), self.vocab_size)], dim=-1)
        input_ids_roll = torch.roll(input_ids_roll, roll, dims=-1)
        attn_mask_roll = torch.roll(attn_mask, roll, dims=-1)
        output_padded = model(input_ids_padded, attention_mask=attn_mask)[0][:, :half_seq_len]
        output_padded_rolled = model(input_ids_roll, attention_mask=attn_mask_roll)[0][:, roll:half_seq_len + roll]
        self.parent.assertTrue(torch.allclose(output_padded, output_padded_rolled, atol=0.001))

    def create_and_check_reformer_layer_dropout_seed(self, config, input_ids, input_mask, choice_labels, is_decoder=False):
        if False:
            while True:
                i = 10
        config.is_decoder = is_decoder
        layer = ReformerLayer(config).to(torch_device)
        layer.train()
        shape = (self.batch_size, self.seq_length, config.hidden_size)
        hidden_states = floats_tensor(shape)
        prev_attn_output = floats_tensor(shape)
        layer_outputs = layer(prev_attn_output, hidden_states, attention_mask=input_mask)
        next_attn_output = layer_outputs.attn_output
        next_hidden_states = layer_outputs.hidden_states
        torch.manual_seed(layer.attention_seed)
        attn_outputs = layer.attention(hidden_states, attention_mask=input_mask)
        self.parent.assertTrue(torch.allclose(prev_attn_output + attn_outputs.hidden_states, next_attn_output, atol=0.001))
        torch.manual_seed(layer.feed_forward_seed)
        feed_forward_hidden_states = layer.feed_forward(next_attn_output)
        self.parent.assertTrue(torch.allclose(next_hidden_states, hidden_states + feed_forward_hidden_states, atol=0.001))

    def create_and_check_reformer_feed_backward_chunking(self, config, input_ids, input_mask, choice_labels):
        if False:
            while True:
                i = 10
        if not self.is_training:
            return
        config.hidden_dropout_prob = 0
        config.local_attention_probs_dropout_prob = 0
        config.lsh_attention_probs_dropout_prob = 0
        config.lsh_num_chunks_after = 1
        config.is_decoder = False
        torch.manual_seed(0)
        model = ReformerForMaskedLM(config=config)
        model.to(torch_device)
        model.train()
        model.zero_grad()
        (loss_no_chunk, output_no_chunk) = model(input_ids, labels=input_ids, attention_mask=input_mask)[:2]
        loss_no_chunk.backward()
        grad_slice_word_no_chunk = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        grad_slice_position_factor_1_no_chunk = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        grad_slice_position_factor_2_no_chunk = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        config.chunk_size_lm_head = 1
        config.chunk_size_feed_forward = 1
        torch.manual_seed(0)
        model = ReformerForMaskedLM(config=config)
        model.to(torch_device)
        model.train()
        model.zero_grad()
        (loss_chunk, output_chunk) = model(input_ids, labels=input_ids, attention_mask=input_mask)[:2]
        loss_chunk.backward()
        grad_slice_word_chunk = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        grad_slice_position_factor_1_chunk = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        grad_slice_position_factor_2_chunk = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        self.parent.assertTrue(torch.allclose(loss_chunk, loss_no_chunk, atol=0.001))
        self.parent.assertTrue(torch.allclose(grad_slice_word_no_chunk, grad_slice_word_chunk, atol=0.001))
        self.parent.assertTrue(torch.allclose(grad_slice_position_factor_1_chunk, grad_slice_position_factor_1_no_chunk, atol=0.001))
        self.parent.assertTrue(torch.allclose(grad_slice_position_factor_2_chunk, grad_slice_position_factor_2_no_chunk, atol=0.001))

    def create_and_check_reformer_random_seed(self, config, input_ids, input_mask, choice_labels):
        if False:
            i = 10
            return i + 15
        layer = ReformerLayer(config).to(torch_device)
        layer.train()
        shape = (self.batch_size, self.seq_length, config.hidden_size)
        hidden_states = floats_tensor(shape)
        attn_output = floats_tensor(shape)
        seeds = []
        for _ in range(100):
            layer_outputs = layer(attn_output, hidden_states, attention_mask=input_mask)
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            torch.manual_seed(layer.attention_seed)
            seeds.append(layer.attention_seed)
        self.parent.assertGreater(len(set(seeds)), 70)
        seeds = []
        for _ in range(100):
            layer_outputs = layer(attn_output, hidden_states, attention_mask=input_mask)
            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            torch.manual_seed(layer.feed_forward_seed)
            seeds.append(layer.feed_forward_seed)
        self.parent.assertGreater(len(set(seeds)), 70)

    def create_and_check_reformer_model_fp16_forward(self, config, input_ids, input_mask, choice_labels):
        if False:
            while True:
                i = 10
        model = ReformerModel(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        output = model(input_ids, attention_mask=input_mask)['last_hidden_state']
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_reformer_model_generate(self, config, input_ids, input_mask, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        config.is_decoder = True
        config.lsh_num_chunks_after = 0
        config.bos_token_id = 0
        config.eos_token_id = None
        config.max_length = 20
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.eval()
        output = model.generate()
        self.parent.assertIsNotNone(output)

    def create_and_check_reformer_model_fp16_generate(self, config, input_ids, input_mask, choice_labels):
        if False:
            return 10
        config.is_decoder = True
        config.lsh_num_chunks_after = 0
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        output = model.generate(input_ids[:, -10:], attention_mask=input_mask, do_sample=False)
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_reformer_no_chunking(self, config, input_ids, input_mask, choice_labels):
        if False:
            i = 10
            return i + 15
        config.lsh_attn_chunk_length = 2 * input_ids.shape[-1]
        config.local_attn_chunk_length = 2 * input_ids.shape[-1]
        config.lsh_num_chunks_after = 1
        config.is_decoder = False
        model = ReformerForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        output_logits = model(input_ids, attention_mask=input_mask)['logits']
        self.parent.assertTrue(output_logits.shape[1] == input_ids.shape[-1])

    def create_and_check_reformer_for_question_answering(self, config, input_ids, input_mask, choice_labels):
        if False:
            for i in range(10):
                print('nop')
        model = ReformerForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, start_positions=choice_labels, end_positions=choice_labels)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_past_buckets_states(self, config, input_ids, input_mask, choice_labels):
        if False:
            print('Hello World!')
        config.is_decoder = True
        config.lsh_num_chunks_before = 1
        config.lsh_num_chunks_after = 0
        model = ReformerModelWithLMHead(config=config)
        model.to(torch_device)
        model.eval()
        input_ids_first = input_ids[:, :-1]
        input_ids_second = input_ids[:, -1:]
        past_buckets_states = model(input_ids_first, use_cache=True)['past_buckets_states']
        outputs_with_cache = model(input_ids_second, past_buckets_states=past_buckets_states, use_cache=True)['logits']
        outputs_without_cache = model(input_ids)['logits'][:, -1]
        random_slice_idx = torch.randint(outputs_without_cache.shape[-1], (1, 1), device=torch_device).item()
        self.parent.assertTrue(torch.allclose(outputs_with_cache[:, 0, random_slice_idx], outputs_without_cache[:, random_slice_idx], atol=0.01))

    def prepare_config_and_inputs_for_common(self):
        if False:
            return 10
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, choice_labels) = config_and_inputs
        inputs_dict = {'input_ids': input_ids, 'attention_mask': input_mask}
        return (config, inputs_dict)

    def create_and_check_reformer_for_sequence_classification(self, config, input_ids, input_mask, choice_labels, is_decoder):
        if False:
            return 10
        config.is_decoder = is_decoder
        sequence_labels = ids_tensor([self.batch_size], config.num_labels)
        model = ReformerForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

class ReformerTesterMixin:
    """
    Reformer Local and Reformer LSH run essentially the same tests
    """

    def test_config(self):
        if False:
            i = 10
            return i + 15
        self.config_tester.run_common_tests()

    def test_reformer_model(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model(*config_and_inputs)

    def test_reformer_lm_model_backward(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_with_lm_backward(*config_and_inputs)

    def test_reformer_model_attn_masking(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, is_decoder=True)
        self.model_tester.create_and_check_reformer_model_with_attn_mask(*config_and_inputs, is_decoder=False)

    def test_reformer_with_lm(self):
        if False:
            return 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_with_lm(*config_and_inputs)

    def test_reformer_with_mlm(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_with_mlm(*config_and_inputs)

    def test_reformer_layer_training_dropout(self):
        if False:
            while True:
                i = 10
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_layer_dropout_seed(*config_and_inputs, is_decoder=True)
        self.model_tester.create_and_check_reformer_layer_dropout_seed(*config_and_inputs, is_decoder=False)

    def test_reformer_chunking_backward_equality(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_feed_backward_chunking(*config_and_inputs)

    def test_reformer_no_chunking(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_no_chunking(*config_and_inputs)

    def test_reformer_qa_answering(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_for_question_answering(*config_and_inputs)

    def test_reformer_cached_inference(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_past_buckets_states(*config_and_inputs)

    def test_reformer_cached_generate(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_generate(*config_and_inputs)

    @slow
    def test_dropout_random_seed_is_changing(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_random_seed(*config_and_inputs)

    @require_torch_fp16
    def test_reformer_model_fp16_forward(self):
        if False:
            for i in range(10):
                print('nop')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_fp16_forward(*config_and_inputs)

    @require_torch_fp16
    def test_reformer_model_fp16_generate(self):
        if False:
            i = 10
            return i + 15
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_model_fp16_generate(*config_and_inputs)

    @require_torch_multi_gpu
    @unittest.skip(reason='Reformer does not work with data parallel (DP) because of a bug in PyTorch: https://github.com/pytorch/pytorch/issues/36035')
    def test_multi_gpu_data_parallel_forward(self):
        if False:
            while True:
                i = 10
        pass

    def test_for_sequence_classification(self):
        if False:
            print('Hello World!')
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_reformer_for_sequence_classification(*config_and_inputs, is_decoder=False)

    def test_retain_grad_hidden_states_attentions(self):
        if False:
            i = 10
            return i + 15
        return

    def test_resize_embeddings_untied(self):
        if False:
            i = 10
            return i + 15
        return

@require_torch
class ReformerLocalAttnModelTest(ReformerTesterMixin, GenerationTesterMixin, ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ReformerModel, ReformerModelWithLMHead, ReformerForSequenceClassification, ReformerForQuestionAnswering) if is_torch_available() else ()
    all_generative_model_classes = (ReformerModelWithLMHead,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    test_sequence_classification_problem_types = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_tester = ReformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ReformerConfig, hidden_size=37)

    @slow
    def test_model_from_pretrained(self):
        if False:
            for i in range(10):
                print('nop')
        for model_name in REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ReformerModelWithLMHead.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def _check_attentions_for_generate(self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1):
        if False:
            print('Hello World!')
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual([isinstance(iter_attentions, list) for iter_attentions in attentions], [True] * len(attentions))
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)
        for (idx, iter_attentions) in enumerate(attentions):
            tgt_len = min_length + idx if not use_cache else 1
            num_chunks = tgt_len // config.local_attn_chunk_length + (tgt_len % config.local_attn_chunk_length != 0)
            tgt_chunk_len = config.local_attn_chunk_length
            src_chunk_len = config.local_attn_chunk_length * (1 + config.local_num_chunks_after + config.local_num_chunks_before)
            if use_cache:
                expected_shape = (batch_size * num_beam_groups, config.num_attention_heads, tgt_len, min_length // config.local_attn_chunk_length + 1 + idx)
            else:
                expected_shape = (batch_size * num_beam_groups, config.num_attention_heads, num_chunks, tgt_chunk_len, src_chunk_len)
            self.assertListEqual([layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions))

    def _check_hidden_states_for_generate(self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual([isinstance(iter_hidden_states, list) for iter_hidden_states in hidden_states], [True] * len(hidden_states))
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)
        for (idx, iter_hidden_states) in enumerate(hidden_states):
            seq_len = min_length + idx
            seq_len = config.local_attn_chunk_length * (seq_len // config.local_attn_chunk_length + (seq_len % config.local_attn_chunk_length != 0))
            if use_cache:
                seq_len = 1
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            self.assertListEqual([layer_hidden_states.shape for layer_hidden_states in iter_hidden_states], [expected_shape] * len(iter_hidden_states))

    @unittest.skip("The model doesn't support left padding")
    def test_left_padding_compatibility(self):
        if False:
            return 10
        pass

@require_torch
class ReformerLSHAttnModelTest(ReformerTesterMixin, ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (ReformerModel, ReformerModelWithLMHead, ReformerForSequenceClassification, ReformerForQuestionAnswering) if is_torch_available() else ()
    all_generative_model_classes = (ReformerModelWithLMHead,) if is_torch_available() else ()
    pipeline_model_mapping = {'feature-extraction': ReformerModel, 'fill-mask': ReformerForMaskedLM, 'question-answering': ReformerForQuestionAnswering, 'text-classification': ReformerForSequenceClassification, 'text-generation': ReformerModelWithLMHead, 'zero-shot': ReformerForSequenceClassification} if is_torch_available() else {}
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            i = 10
            return i + 15
        if pipeline_test_casse_name == 'QAPipelineTests' and tokenizer_name is not None and (not tokenizer_name.endswith('Fast')):
            return True
        return False

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_tester = ReformerModelTester(self, batch_size=13, seq_length=13, use_input_mask=True, use_labels=True, is_training=False, is_decoder=True, vocab_size=32, attention_head_size=16, hidden_size=64, num_attention_heads=2, num_buckets=2, num_hashes=4, lsh_attn_chunk_length=4, lsh_num_chunks_before=1, lsh_num_chunks_after=0, chunk_size_lm_head=5, chunk_size_feed_forward=6, feed_forward_size=32, hidden_act='relu', hidden_dropout_prob=0.1, lsh_attention_probs_dropout_prob=0.1, max_position_embeddings=512, initializer_range=0.02, axial_norm_std=1.0, layer_norm_eps=1e-12, axial_pos_embds=True, axial_pos_shape=[4, 8], axial_pos_embds_dim=[16, 48], attn_layers=['lsh'], pad_token_id=0, eos_token_id=2, scope=None, hash_seed=0, num_labels=2)
        self.config_tester = ConfigTester(self, config_class=ReformerConfig, hidden_size=37)

    def _check_attentions_for_generate(self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1):
        if False:
            print('Hello World!')
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual([isinstance(iter_attentions, list) for iter_attentions in attentions], [True] * len(attentions))
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)
        for (idx, iter_attentions) in enumerate(attentions):
            tgt_len = min_length + idx if not use_cache else 1
            num_chunks = tgt_len // config.lsh_attn_chunk_length + (tgt_len % config.lsh_attn_chunk_length != 0)
            tgt_chunk_len = config.lsh_attn_chunk_length
            src_chunk_len = config.lsh_attn_chunk_length * (1 + config.lsh_num_chunks_after + config.lsh_num_chunks_before)
            if use_cache:
                expected_shape = (batch_size * num_beam_groups, config.num_attention_heads, config.num_hashes, tgt_len, config.num_hashes * (1 + config.lsh_num_chunks_after + config.lsh_num_chunks_before))
            else:
                expected_shape = (batch_size * num_beam_groups, config.num_attention_heads, num_chunks * config.num_hashes, tgt_chunk_len, src_chunk_len)
            self.assertListEqual([layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions))

    def _check_hidden_states_for_generate(self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual([isinstance(iter_hidden_states, list) for iter_hidden_states in hidden_states], [True] * len(hidden_states))
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)
        for (idx, iter_hidden_states) in enumerate(hidden_states):
            seq_len = min_length + idx if not use_cache else 1
            seq_len = config.lsh_attn_chunk_length * (seq_len // config.lsh_attn_chunk_length + (seq_len % config.lsh_attn_chunk_length != 0))
            if use_cache:
                seq_len = 1
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            self.assertListEqual([layer_hidden_states.shape for layer_hidden_states in iter_hidden_states], [expected_shape] * len(iter_hidden_states))

    @unittest.skip('Fails because the sequence length is not a multiple of 4')
    def test_problem_types(self):
        if False:
            return 10
        pass

    @unittest.skip('Fails because the sequence length is not a multiple of 4')
    def test_past_key_values_format(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip("The model doesn't support left padding")
    def test_left_padding_compatibility(self):
        if False:
            i = 10
            return i + 15
        pass

@require_torch
@require_sentencepiece
@require_tokenizers
class ReformerIntegrationTests(unittest.TestCase):
    """
    These integration tests test the current layer activations and gradients againts the output of the Hugging Face Reformer model at time of integration: 29/06/2020. During integration, the model was tested against the output of the official Trax ReformerLM model for various cases ("lsh" only, "lsh" only, masked / non-masked, different chunk length, ....). In order to recover the original trax integration tests, one should use patrickvonplaten's fork of trax and the code that lives on the branch `reformer_trax_tests`.
    """

    def _get_basic_config_and_input(self):
        if False:
            i = 10
            return i + 15
        config = {'vocab_size': 320, 'attention_head_size': 8, 'hidden_size': 16, 'num_attention_heads': 2, 'num_buckets': 2, 'num_hashes': 4, 'lsh_attn_chunk_length': 4, 'local_attn_chunk_length': 4, 'lsh_num_chunks_before': 1, 'lsh_num_chunks_after': 0, 'local_num_chunks_before': 1, 'local_num_chunks_after': 0, 'chunk_size_lm_head': 0, 'chunk_size_feed_forward': 0, 'feed_forward_size': 32, 'hidden_act': 'gelu', 'hidden_dropout_prob': 0.0, 'lsh_attention_probs_dropout_prob': 0.0, 'local_attention_probs_dropout_prob': 0.0, 'max_position_embeddings': 32, 'initializer_range': 0.02, 'axial_norm_std': 1.0, 'layer_norm_eps': 1e-12, 'sinusoidal_pos_embds': False, 'axial_pos_embds': True, 'axial_pos_shape': [4, 8], 'axial_pos_embds_dim': [8, 8], 'hash_seed': 0, 'is_decoder': True}
        return config

    def _get_hidden_states(self):
        if False:
            i = 10
            return i + 15
        return torch.tensor([[[1.90826353, -1.4599973, -0.620405462, 1.52503433, -0.364464232, -0.827359235, 0.839670803, 0.244492178, 0.498332758, 2.69175139, -0.00708081422, 1.04915401, -1.83476661, 0.767220476, 0.298580543, 0.0284803992], [-0.0266374286, 0.433497576, 0.310386309, 0.546039944, -0.000247292666, -0.752305019, 0.239162103, 0.725216186, -0.758357372, 0.420635998, -0.0404739919, 0.159924145, 2.05135748, -1.15997978, 0.537166397, 0.262873606], [0.185247482, 0.707046037, -0.677089715, -2.24209655, -0.037530798, -0.859380874, -2.81027884, 1.01276376, -1.69438001, 0.41757466, -1.49196962, -1.76483717, -0.194566312, -1.71183858, 0.772903565, -1.11557056], [0.946069193, 0.153417623, -0.958686996, 0.118126669, 1.75967724, 1.6219459, -0.574108159, 0.679920443, 0.544028163, 0.205466114, -0.363045868, 0.241865062, 0.320348382, -0.905611176, -0.192690727, -1.19917547]]], dtype=torch.float32, device=torch_device)

    def _get_attn_mask(self):
        if False:
            i = 10
            return i + 15
        return torch.tensor([[0, 1, 0, 0]], dtype=torch.long, device=torch_device)

    def _get_input_ids_and_mask(self):
        if False:
            for i in range(10):
                print('nop')
        mask = torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0]], dtype=torch.long, device=torch_device)
        input_ids = torch.tensor([[89, 279, 286, 84, 194, 316, 182, 28, 283, 37, 169, 7, 253, 267, 107, 250, 44, 7, 102, 62, 3, 243, 171, 265, 302, 48, 164, 264, 148, 229, 280, 150], [9, 192, 66, 112, 163, 83, 135, 70, 224, 96, 31, 80, 196, 80, 63, 22, 85, 100, 47, 283, 0, 163, 126, 143, 195, 82, 53, 82, 18, 27, 182, 52]], dtype=torch.long, device=torch_device)
        return (input_ids, mask)

    def test_lsh_layer_forward(self):
        if False:
            i = 10
            return i + 15
        config = self._get_basic_config_and_input()
        config['lsh_num_chunks_before'] = 0
        config['attn_layers'] = ['lsh']
        config['is_decoder'] = False
        hidden_states = self._get_hidden_states()
        torch.manual_seed(0)
        layer = ReformerLayer(ReformerConfig(**config)).to(torch_device)
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states.clone(), hidden_states=hidden_states)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor([1.6879, -1.3083, -0.4708, 1.3555, -0.6292], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=0.001))

    def test_lsh_layer_forward_complex(self):
        if False:
            while True:
                i = 10
        config = self._get_basic_config_and_input()
        config['lsh_num_chunks_before'] = 0
        config['attn_layers'] = ['lsh']
        config['num_buckets'] = [2, 4]
        attn_mask = self._get_attn_mask()
        hidden_states = self._get_hidden_states()
        torch.manual_seed(0)
        layer = ReformerLayer(ReformerConfig(**config)).to(torch_device)
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states.clone(), hidden_states=hidden_states, attention_mask=attn_mask)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor([1.6439, -1.2306, -0.5108, 1.3006, -0.6537], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=0.001))

    def test_local_layer_forward(self):
        if False:
            while True:
                i = 10
        config = self._get_basic_config_and_input()
        config['local_num_chunks_before'] = 0
        config['attn_layers'] = ['local']
        config['is_decoder'] = False
        hidden_states = self._get_hidden_states()
        torch.manual_seed(0)
        layer = ReformerLayer(ReformerConfig(**config)).to(torch_device)
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states, hidden_states=hidden_states)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor([1.4212, -2.0576, -0.9688, 1.4599, -0.1344], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=0.001))

    def test_local_layer_forward_complex(self):
        if False:
            print('Hello World!')
        config = self._get_basic_config_and_input()
        config['local_num_chunks_before'] = 0
        config['attn_layers'] = ['local']
        attn_mask = self._get_attn_mask()
        hidden_states = self._get_hidden_states()
        torch.manual_seed(0)
        layer = ReformerLayer(ReformerConfig(**config)).to(torch_device)
        layer.eval()
        reformer_output = layer(prev_attn_output=hidden_states, hidden_states=hidden_states, attention_mask=attn_mask)
        output_slice = reformer_output.hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor([1.475, -2.0235, -0.9743, 1.4463, -0.1269], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=0.001))

    def test_lsh_model_forward(self):
        if False:
            return 10
        config = self._get_basic_config_and_input()
        config['attn_layers'] = ['lsh', 'lsh', 'lsh', 'lsh']
        config['num_buckets'] = [2, 4]
        torch.manual_seed(0)
        model = ReformerModel(ReformerConfig(**config)).to(torch_device)
        model.eval()
        (input_ids, attn_mask) = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor([-0.9896, -0.9396, -1.0831, -0.0597, 0.2456], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=0.001))

    def test_local_model_forward(self):
        if False:
            print('Hello World!')
        config = self._get_basic_config_and_input()
        config['attn_layers'] = ['local', 'local', 'local', 'local']
        torch.manual_seed(0)
        model = ReformerModel(ReformerConfig(**config)).to(torch_device)
        model.eval()
        (input_ids, attn_mask) = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[0, 0, :5]
        expected_output_slice = torch.tensor([-1.6791, 0.7171, 0.1594, 0.4063, 1.2584], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=0.001))

    def test_lm_model_forward(self):
        if False:
            for i in range(10):
                print('nop')
        config = self._get_basic_config_and_input()
        config['attn_layers'] = ['local', 'lsh', 'local', 'lsh', 'local', 'lsh']
        config['num_buckets'] = [2, 4]
        config['is_decoder'] = False
        torch.manual_seed(0)
        model = ReformerForMaskedLM(ReformerConfig(**config)).to(torch_device)
        model.eval()
        (input_ids, attn_mask) = self._get_input_ids_and_mask()
        hidden_states = model(input_ids=input_ids, attention_mask=attn_mask)[0]
        output_slice = hidden_states[1, -1, :5]
        expected_output_slice = torch.tensor([0.1018, -0.2026, 0.2116, 0.027, -0.1233], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=0.001))

    def test_local_lm_model_grad(self):
        if False:
            i = 10
            return i + 15
        config = self._get_basic_config_and_input()
        config['attn_layers'] = ['local', 'local', 'local', 'local']
        config['hidden_dropout_prob'] = 0.0
        config['local_attention_probs_dropout_prob'] = 0.0
        torch.manual_seed(0)
        model = ReformerModelWithLMHead(ReformerConfig(**config)).to(torch_device)
        model.train()
        model.zero_grad()
        (input_ids, _) = self._get_input_ids_and_mask()
        loss = model(input_ids=input_ids, labels=input_ids)[0]
        self.assertTrue(torch.allclose(loss, torch.tensor(5.8019, dtype=torch.float, device=torch_device), atol=0.001))
        loss.backward()
        grad_slice_word = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        expected_grad_slice_word = torch.tensor([-0.0005, -0.0001, -0.0002, -0.0006, -0.0006], dtype=torch.float, device=torch_device)
        grad_slice_position_factor_1 = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        expected_grad_slice_pos_fac_1 = torch.tensor([-0.5235, 0.5704, 0.0922, -0.314, 0.9928], dtype=torch.float, device=torch_device)
        grad_slice_position_factor_2 = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        expected_grad_slice_pos_fac_2 = torch.tensor([1.796, 1.7668, 0.5593, 0.0907, 1.8342], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(grad_slice_word, expected_grad_slice_word, atol=0.001))
        self.assertTrue(torch.allclose(grad_slice_position_factor_1, expected_grad_slice_pos_fac_1, atol=0.001))
        self.assertTrue(torch.allclose(grad_slice_position_factor_2, expected_grad_slice_pos_fac_2, atol=0.001))

    def test_lsh_lm_model_grad(self):
        if False:
            return 10
        config = self._get_basic_config_and_input()
        config['attn_layers'] = ['lsh', 'lsh', 'lsh', 'lsh']
        config['hidden_dropout_prob'] = 0.0
        config['lsh_attention_probs_dropout_prob'] = 0.0
        config['num_buckets'] = [2, 4]
        config['num_hashes'] = 6
        torch.manual_seed(0)
        model = ReformerModelWithLMHead(ReformerConfig(**config)).to(torch_device)
        model.train()
        model.zero_grad()
        (input_ids, _) = self._get_input_ids_and_mask()
        loss = model(input_ids=input_ids, labels=input_ids)[0]
        self.assertTrue(torch.allclose(loss, torch.tensor(5.7854, dtype=torch.float, device=torch_device), atol=0.001))
        loss.backward()
        grad_slice_word = model.reformer.embeddings.word_embeddings.weight.grad[0, :5]
        expected_grad_slice_word = torch.tensor([0.0004, 0.0003, 0.0006, -0.0004, 0.0002], dtype=torch.float, device=torch_device)
        grad_slice_position_factor_1 = model.reformer.embeddings.position_embeddings.weights[0][1, 0, -5:]
        expected_grad_slice_pos_fac_1 = torch.tensor([-0.3792, 0.5593, -1.6993, 0.2033, 0.4131], dtype=torch.float, device=torch_device)
        grad_slice_position_factor_2 = model.reformer.embeddings.position_embeddings.weights[1][0, 1, :5]
        expected_grad_slice_pos_fac_2 = torch.tensor([-1.4212, -0.3201, -1.1944, 0.1258, 0.2856], dtype=torch.float, device=torch_device)
        self.assertTrue(torch.allclose(grad_slice_word, expected_grad_slice_word, atol=0.001))
        self.assertTrue(torch.allclose(grad_slice_position_factor_1, expected_grad_slice_pos_fac_1, atol=0.001))
        self.assertTrue(torch.allclose(grad_slice_position_factor_2, expected_grad_slice_pos_fac_2, atol=0.001))

    @slow
    def test_pretrained_generate_crime_and_punish(self):
        if False:
            return 10
        model = ReformerModelWithLMHead.from_pretrained('google/reformer-crime-and-punishment').to(torch_device)
        tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
        model.eval()
        input_ids = tokenizer.encode('A few months later', return_tensors='pt').to(torch_device)
        output_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True, do_sample=False, num_hashes=8)
        output = tokenizer.decode(output_ids[0])
        self.assertEqual(output, 'A few months later state expression in his ideas, at the first entrance. He was positively for an inst')

    @slow
    def test_pretrained_generate_use_cache_equality(self):
        if False:
            i = 10
            return i + 15
        model = ReformerModelWithLMHead.from_pretrained('google/reformer-crime-and-punishment').to(torch_device)
        tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
        model.eval()
        input_ids = tokenizer.encode('A few months later', return_tensors='pt').to(torch_device)
        output_ids_with_cache = model.generate(input_ids, max_length=130, num_hashes=8, use_cache=False)
        output_ids_without_cache = model.generate(input_ids, max_length=130, num_hashes=8, use_cache=True)
        output_with_cache = tokenizer.decode(output_ids_with_cache[0])
        output_without_cache = tokenizer.decode(output_ids_without_cache[0])
        self.assertEqual(output_with_cache, output_without_cache)