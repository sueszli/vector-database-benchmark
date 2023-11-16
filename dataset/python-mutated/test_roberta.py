import functools
import unittest
from typing import Any, Dict, Sequence
import fairseq
import fairseq.options
import fairseq.tasks
import torch
from tests.utils import dummy_dictionary
VOCAB_SIZE = 100

@fairseq.tasks.register_task('fake_task')
class FakeTask(fairseq.tasks.LegacyFairseqTask):

    def __init__(self, args):
        if False:
            return 10
        super().__init__(args)
        self.dictionary = dummy_dictionary(VOCAB_SIZE - 4)
        assert len(self.dictionary) == VOCAB_SIZE

    @property
    def source_dictionary(self):
        if False:
            while True:
                i = 10
        return self.dictionary

    @property
    def target_dictionary(self):
        if False:
            return 10
        return self.dictionary

@functools.lru_cache()
def get_toy_model(device: str, architecture: str='roberta_enc_dec', **extra_args: Any):
    if False:
        print('Hello World!')
    assert device in ('gpu', 'cpu')
    kwargs = {'arch': architecture, 'encoder_layers': 3, 'encoder_embed_dim': 12, 'encoder_ffn_embed_dim': 14, 'encoder_attention_heads': 4, 'decoder_layers': 3, 'decoder_embed_dim': 12, 'decoder_ffn_embed_dim': 14, 'decoder_attention_heads': 4, 'dropout': 0, 'attention_dropout': 0, 'activation_dropout': 0, 'encoder_layerdrop': 0, 'tokens_per_sample': 256, 'data': '/tmp/test_roberta'}
    kwargs.update(extra_args)
    fake_task = FakeTask(kwargs)
    args = fairseq.options.get_args(task='online_backtranslation', mono_langs='en,ro', valid_lang_pairs='en-ro', **kwargs)
    torch.manual_seed(0)
    model = fake_task.build_model(args)
    if device == 'gpu':
        model.cuda()
    return (fake_task, model)

def mk_sample(lang: str, device: str, tok: Sequence[int]=None, batch_size: int=2) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    assert device in ('gpu', 'cpu')
    if not tok:
        if lang == 'en':
            tok = [10, 11, 12, 13, 14, 15, 2]
        else:
            tok = [20, 21, 22, 23, 24, 25, 26, 27, 2]
    batch = torch.stack([torch.tensor(tok, dtype=torch.long)] * batch_size)
    if device == 'gpu':
        batch = batch.cuda()
    sample = {'net_input': {'src_tokens': batch, 'prev_output_tokens': batch, 'src_lengths': torch.tensor([len(tok)] * batch_size, dtype=torch.long, device=batch.device)}, 'target': batch[:, 1:]}
    return sample

def cpu_gpu(fn):
    if False:
        for i in range(10):
            print('nop')

    def helper(self):
        if False:
            for i in range(10):
                print('nop')
        fn(self, 'cpu')
        if torch.cuda.is_available():
            fn(self, 'gpu')
    return helper

def architectures(fn):
    if False:
        while True:
            i = 10

    def helper(self):
        if False:
            print('Hello World!')
        for arch in ['roberta_enc_dec', 'transformer']:
            fn(self, arch)
    return helper

class RobertaTest(unittest.TestCase):

    def assertTensorEqual(self, t1, t2, delta: float=1e-06):
        if False:
            return 10
        self.assertEqual(t1.size(), t2.size(), 'size mismatch')
        if delta == 0.0:
            self.assertEqual(t1.ne(t2).long().sum(), 0)
        else:
            self.assertEqual(((t2 - t1).abs() > delta).long().sum(), 0)

    def assertSharing(self, model, link_groups: Sequence[Sequence[str]]):
        if False:
            while True:
                i = 10
        ids = {}
        for group in link_groups:
            group_ids = {name: id(params(model, name)) for name in group}
            shared_id = group_ids[group[0]]
            self.assertEqual(group_ids, {name: shared_id for name in group})
            self.assertNotIn(shared_id, ids)
            ids[shared_id] = group

    def test_roberta_shared_params(self):
        if False:
            print('Hello World!')
        (_, roberta) = get_toy_model('cpu', architecture='roberta')
        self.assertSharing(roberta, [['encoder.sentence_encoder.embed_tokens.weight', 'encoder.lm_head.weight']])
        (_, roberta) = get_toy_model('cpu', architecture='roberta', untie_weights_roberta=True)
        self.assertSharing(roberta, [['encoder.sentence_encoder.embed_tokens.weight'], ['encoder.lm_head.weight']])

    def test_roberta_enc_dec_shared_params(self):
        if False:
            while True:
                i = 10
        (_, enc_dec) = get_toy_model('cpu', architecture='roberta_enc_dec')
        self.assertSharing(enc_dec, [['encoder.embed_tokens.weight'], ['decoder.embed_tokens.weight'], ['decoder.output_projection.weight']])
        (_, enc_dec) = get_toy_model('cpu', architecture='roberta_enc_dec', share_decoder_input_output_embed=True)
        self.assertSharing(enc_dec, [['encoder.embed_tokens.weight'], ['decoder.embed_tokens.weight', 'decoder.output_projection.weight']])
        (_, enc_dec) = get_toy_model('cpu', architecture='roberta_enc_dec', share_all_embeddings=True)
        self.assertSharing(enc_dec, [['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'decoder.output_projection.weight']])

    def test_roberta_max_positions_is_correctly_set(self):
        if False:
            return 10
        device = 'cpu'
        (task, model) = get_toy_model(device)
        max_pos = model.max_decoder_positions()
        self.assertEqual(max_pos, 256)
        self.assertEqual(max_pos, model.decoder.max_positions())
        self.assertEqual(max_pos, model.encoder.max_positions())
        self.assertEqual(max_pos, model.encoder.embed_positions.max_positions)
        sentence = [31 for _ in range(max_pos)]
        sample = mk_sample('en', device, sentence, batch_size=1)
        self.assertEqual(list(sample['net_input']['src_lengths']), [max_pos])
        self.assertEqual(len(sample['net_input']['src_tokens'][0]), max_pos)
        (x, _) = model.forward(**sample['net_input'])
        self.assertEqual(x.shape, (1, max_pos, VOCAB_SIZE))

    @cpu_gpu
    def test_roberta_forward_backward(self, device: str):
        if False:
            return 10
        (_, model) = get_toy_model(device)
        sample = mk_sample('en', device)
        en_tokens = sample['net_input']['src_tokens']
        (bs, l) = en_tokens.shape
        (logits, _) = model(**sample['net_input'])
        self.assertEqual(logits.shape, (bs, l, VOCAB_SIZE))
        loss = logits.sum()
        loss.backward()

    @cpu_gpu
    def test_roberta_forward_backward_bs1(self, device: str):
        if False:
            i = 10
            return i + 15
        (_, model) = get_toy_model(device)
        sample = mk_sample('en', device, batch_size=1)
        (o, _) = model.forward(**sample['net_input'])
        loss = o.sum()
        sample2 = mk_sample('ro', device, batch_size=1)
        (o, _) = model.forward(**sample2['net_input'])
        loss += o.sum()
        loss.backward()

    @cpu_gpu
    def test_roberta_batching(self, device: str):
        if False:
            i = 10
            return i + 15
        '\n        Checks that the batch of size 2 give twice the same results than the batch of size 1.\n        '
        (_, model) = get_toy_model(device)
        sample = mk_sample('en', device, batch_size=1)
        slen = sample['net_input']['src_lengths'][0]
        sample2 = mk_sample('en', device, batch_size=2)
        with torch.no_grad():
            z = model.encoder.forward(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])
            z = z['encoder_out'][-1]
            (logits, _) = model.forward(**sample['net_input'])
            z2 = model.encoder.forward(sample2['net_input']['src_tokens'], sample['net_input']['src_lengths'])
            z2 = z2['encoder_out'][-1]
            (logits2, _) = model.forward(**sample2['net_input'])
        self.assertEqual(z.shape, (slen, 1, 12))
        self.assertEqual(z2.shape, (slen, 2, 12))
        self.assertTensorEqual(logits2[0], logits2[1])
        self.assertTensorEqual(logits[0], logits2[0])

    @cpu_gpu
    def test_roberta_incremental_decoder(self, device: str):
        if False:
            i = 10
            return i + 15
        '\n        Checks that incremental decoding yields the same result than non incremental one.\n        '
        (task, model) = get_toy_model(device)
        en_sample = mk_sample('en', device)
        en_tokens = en_sample['net_input']['src_tokens']
        ro_sample = mk_sample('ro', device)
        ro_tokens = ro_sample['net_input']['src_tokens']
        en_enc = model.encoder.forward(en_tokens, src_lengths=en_sample['net_input']['src_lengths'])
        (bs, tgt_len) = ro_tokens.shape
        (ro_dec, _) = model.decoder.forward(ro_tokens, encoder_out=en_enc)
        self.assertEqual(ro_dec.shape, (bs, tgt_len, VOCAB_SIZE))
        self.assertTensorEqual(ro_dec[0], ro_dec[1])
        inc_state = {}
        ro_dec_inc = []
        for i in range(tgt_len):
            (ro, _) = model.decoder.forward(ro_tokens[:, :i + 1], encoder_out=en_enc, incremental_state=inc_state)
            self.assertEqual(ro.shape, (bs, 1, VOCAB_SIZE))
            ro_dec_inc.append(ro)
        for i in range(tgt_len):
            self.assertTensorEqual(ro_dec_inc[i][0], ro_dec_inc[i][1])
            self.assertTensorEqual(ro_dec_inc[i][:, 0], ro_dec[:, i])

    @cpu_gpu
    def test_regularize_for_adaprune_in_roberta(self, device: str):
        if False:
            while True:
                i = 10
        (_, model) = get_toy_model(device=device, architecture='roberta_base', mha_reg_scale_factor=0.000375, ffn_reg_scale_factor=0.000375)
        sample = mk_sample('en', device, batch_size=1)
        (task_loss, _) = model.forward(**sample['net_input'])
        head_loss = model._get_adaptive_head_loss()
        ffn_loss = model._get_adaptive_ffn_loss()
        loss = task_loss.sum() + head_loss + ffn_loss
        loss.backward()

    @cpu_gpu
    def test_ffn_prune_for_adaprune_in_roberta(self, device: str):
        if False:
            for i in range(10):
                print('nop')
        (_, model) = get_toy_model(device=device, architecture='roberta_base')
        sample = mk_sample('en', device, batch_size=1)
        for layer in model.encoder.sentence_encoder.layers:
            fc1_original_size = layer.fc1.out_features
            remove_index = layer._get_fc_rank(remove_num=2)
            layer._prune_fc_layer(remove_index=remove_index)
            self.assertEqual(layer.fc1.out_features, fc1_original_size - 2)
        (task_loss, _) = model.forward(**sample['net_input'])

def params(model, name):
    if False:
        i = 10
        return i + 15
    if '.' not in name:
        return getattr(model, name)
    (prefix, name) = name.split('.', 1)
    return params(getattr(model, prefix), name)