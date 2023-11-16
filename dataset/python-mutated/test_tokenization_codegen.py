import json
import os
import re
import unittest
from transformers import CodeGenTokenizer, CodeGenTokenizerFast
from transformers.models.codegen.tokenization_codegen import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers, slow
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class CodeGenTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = CodeGenTokenizer
    rust_tokenizer_class = CodeGenTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_kwargs = {'add_prefix_space': True}
    test_seq2seq = False

    def setUp(self):
        if False:
            return 10
        super().setUp()
        vocab = ['l', 'o', 'w', 'e', 'r', 's', 't', 'i', 'd', 'n', 'Ġ', 'Ġl', 'Ġn', 'Ġlo', 'Ġlow', 'er', 'Ġlowest', 'Ġnewer', 'Ġwider', '<unk>', '<|endoftext|>']
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ['#version: 0.2', 'Ġ l', 'Ġl o', 'Ġlo w', 'e r', '']
        self.special_tokens_map = {'unk_token': '<unk>'}
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['merges_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(vocab_tokens) + '\n')
        with open(self.merges_file, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(merges))

    def get_tokenizer(self, **kwargs):
        if False:
            while True:
                i = 10
        kwargs.update(self.special_tokens_map)
        return CodeGenTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        if False:
            print('Hello World!')
        kwargs.update(self.special_tokens_map)
        return CodeGenTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            for i in range(10):
                print('nop')
        input_text = 'lower newer'
        output_text = 'lower newer'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            i = 10
            return i + 15
        tokenizer = CodeGenTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = 'lower newer'
        bpe_tokens = ['Ġlow', 'er', 'Ġ', 'n', 'e', 'w', 'er']
        tokens = tokenizer.tokenize(text, add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if False:
            i = 10
            return i + 15
        if not self.test_rust_tokenizer:
            return
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer(add_prefix_space=True)
        sequence = 'lower newer'
        tokens = tokenizer.tokenize(sequence, add_prefix_space=True)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)
        ids = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=True)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)
        rust_tokenizer = self.get_rust_tokenizer(add_prefix_space=True)
        ids = tokenizer.encode(sequence, add_prefix_space=True)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)
        input_tokens = tokens + [rust_tokenizer.unk_token]
        input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(rust_tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_pretokenized_inputs(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def test_padding(self, max_length=15):
        if False:
            print('Hello World!')
        for (tokenizer, pretrained_name, kwargs) in self.tokenizers_list:
            with self.subTest(f'{tokenizer.__class__.__name__} ({pretrained_name})'):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                s = 'This is a simple input'
                s2 = ['This is a simple input 1', 'This is a simple input 2']
                p = ('This is a simple input', 'This is a pair')
                p2 = [('This is a simple input 1', 'This is a simple input 2'), ('This is a simple pair 1', 'This is a simple pair 2')]
                self.assertRaises(ValueError, tokenizer_r.encode, s, max_length=max_length, padding='max_length')
                self.assertRaises(ValueError, tokenizer_r.encode_plus, s, max_length=max_length, padding='max_length')
                self.assertRaises(ValueError, tokenizer_r.batch_encode_plus, s2, max_length=max_length, padding='max_length')
                self.assertRaises(ValueError, tokenizer_r.encode, p, max_length=max_length, padding='max_length')
                self.assertRaises(ValueError, tokenizer_r.encode_plus, p, max_length=max_length, padding='max_length')
                self.assertRaises(ValueError, tokenizer_r.batch_encode_plus, p2, max_length=max_length, padding='max_length')

    def test_padding_if_pad_token_set_slow(self):
        if False:
            i = 10
            return i + 15
        tokenizer = CodeGenTokenizer.from_pretrained(self.tmpdirname, pad_token='<pad>')
        s = 'This is a simple input'
        s2 = ['This is a simple input looooooooong', 'This is a simple input']
        p = ('This is a simple input', 'This is a pair')
        p2 = [('This is a simple input loooooong', 'This is a simple input'), ('This is a simple pair loooooong', 'This is a simple pair')]
        pad_token_id = tokenizer.pad_token_id
        out_s = tokenizer(s, padding='max_length', max_length=30, return_tensors='np')
        out_s2 = tokenizer(s2, padding=True, truncate=True, return_tensors='np')
        out_p = tokenizer(*p, padding='max_length', max_length=60, return_tensors='np')
        out_p2 = tokenizer(p2, padding=True, truncate=True, return_tensors='np')
        self.assertEqual(out_s['input_ids'].shape[-1], 30)
        self.assertTrue(pad_token_id in out_s['input_ids'])
        self.assertTrue(0 in out_s['attention_mask'])
        self.assertEqual(out_s2['input_ids'].shape[-1], 33)
        self.assertFalse(pad_token_id in out_s2['input_ids'][0])
        self.assertFalse(0 in out_s2['attention_mask'][0])
        self.assertTrue(pad_token_id in out_s2['input_ids'][1])
        self.assertTrue(0 in out_s2['attention_mask'][1])
        self.assertEqual(out_p['input_ids'].shape[-1], 60)
        self.assertTrue(pad_token_id in out_p['input_ids'])
        self.assertTrue(0 in out_p['attention_mask'])
        self.assertEqual(out_p2['input_ids'].shape[-1], 52)
        self.assertFalse(pad_token_id in out_p2['input_ids'][0])
        self.assertFalse(0 in out_p2['attention_mask'][0])
        self.assertTrue(pad_token_id in out_p2['input_ids'][1])
        self.assertTrue(0 in out_p2['attention_mask'][1])

    def test_add_bos_token_slow(self):
        if False:
            return 10
        bos_token = '$$$'
        tokenizer = CodeGenTokenizer.from_pretrained(self.tmpdirname, bos_token=bos_token, add_bos_token=True)
        s = 'This is a simple input'
        s2 = ['This is a simple input 1', 'This is a simple input 2']
        bos_token_id = tokenizer.bos_token_id
        out_s = tokenizer(s)
        out_s2 = tokenizer(s2)
        self.assertEqual(out_s.input_ids[0], bos_token_id)
        self.assertTrue(all((o[0] == bos_token_id for o in out_s2.input_ids)))
        decode_s = tokenizer.decode(out_s.input_ids)
        decode_s2 = tokenizer.batch_decode(out_s2.input_ids)
        self.assertTrue(decode_s.startswith(bos_token))
        self.assertTrue(all((d.startswith(bos_token) for d in decode_s2)))

    @slow
    def test_truncation(self):
        if False:
            return 10
        tokenizer = CodeGenTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
        text = '\nif len_a > len_b:\n    result = a\nelse:\n    result = b\n\n\n\n#'
        expected_trucated_text = '\nif len_a > len_b:      result = a\nelse:      result = b'
        input_ids = tokenizer.encode(text)
        truncation_pattern = ['^#', re.escape('<|endoftext|>'), "^'''", '^"""', '\n\n\n']
        decoded_text = tokenizer.decode(input_ids, truncate_before_pattern=truncation_pattern)
        self.assertEqual(decoded_text, expected_trucated_text)

    def test_padding_different_model_input_name(self):
        if False:
            while True:
                i = 10
        pass