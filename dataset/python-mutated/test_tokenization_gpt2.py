import json
import os
import unittest
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import VOCAB_FILES_NAMES
from transformers.testing_utils import require_jinja, require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class GPT2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = GPT2Tokenizer
    rust_tokenizer_class = GPT2TokenizerFast
    test_rust_tokenizer = True
    from_pretrained_kwargs = {'add_prefix_space': True}
    test_seq2seq = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        kwargs.update(self.special_tokens_map)
        return GPT2Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        if False:
            print('Hello World!')
        kwargs.update(self.special_tokens_map)
        return GPT2TokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            i = 10
            return i + 15
        input_text = 'lower newer'
        output_text = 'lower newer'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            while True:
                i = 10
        tokenizer = GPT2Tokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = 'lower newer'
        bpe_tokens = ['Ġlow', 'er', 'Ġ', 'n', 'e', 'w', 'er']
        tokens = tokenizer.tokenize(text, add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
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
            print('Hello World!')
        tokenizer = GPT2Tokenizer.from_pretrained(self.tmpdirname, pad_token='<pad>')
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
        tokenizer = GPT2Tokenizer.from_pretrained(self.tmpdirname, bos_token=bos_token, add_bos_token=True)
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

    def test_padding_different_model_input_name(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_special_tokens_mask_input_pairs_and_bos_token(self):
        if False:
            return 10
        tokenizers = [self.get_tokenizer(do_lower_case=False, add_bos_token=True)]
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                sequence_0 = 'Encode this.'
                sequence_1 = 'This one too please.'
                encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
                encoded_sequence += tokenizer.encode(sequence_1, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(sequence_0, sequence_1, add_special_tokens=True, return_special_tokens_mask=True)
                encoded_sequence_w_special = encoded_sequence_dict['input_ids']
                special_tokens_mask = encoded_sequence_dict['special_tokens_mask']
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
                filtered_sequence = [x if not special_tokens_mask[i] else None for (i, x) in enumerate(encoded_sequence_w_special)]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                self.assertEqual(encoded_sequence, filtered_sequence)

    @require_jinja
    def test_tokenization_for_chat(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = GPT2Tokenizer.from_pretrained(self.tmpdirname)
        test_chats = [[{'role': 'system', 'content': 'You are a helpful chatbot.'}, {'role': 'user', 'content': 'Hello!'}], [{'role': 'system', 'content': 'You are a helpful chatbot.'}, {'role': 'user', 'content': 'Hello!'}, {'role': 'assistant', 'content': 'Nice to meet you.'}], [{'role': 'assistant', 'content': 'Nice to meet you.'}, {'role': 'user', 'content': 'Hello!'}]]
        tokenized_chats = [tokenizer.apply_chat_template(test_chat) for test_chat in test_chats]
        expected_tokens = [[20, 1, 20, 10, 20, 4, 3, 10, 20, 10, 20, 3, 0, 20, 20, 20, 0, 10, 20, 20, 20, 6, 20, 1, 6, 20, 20, 20, 3, 0, 0, 1, 20, 20], [20, 1, 20, 10, 20, 4, 3, 10, 20, 10, 20, 3, 0, 20, 20, 20, 0, 10, 20, 20, 20, 6, 20, 1, 6, 20, 20, 20, 3, 0, 0, 1, 20, 20, 20, 7, 20, 3, 10, 6, 1, 10, 20, 3, 3, 6, 10, 20, 1, 20, 20, 20], [20, 7, 20, 3, 10, 6, 1, 10, 20, 3, 3, 6, 10, 20, 1, 20, 20, 20, 20, 3, 0, 0, 1, 20, 20]]
        for (tokenized_chat, expected_tokens) in zip(tokenized_chats, expected_tokens):
            self.assertListEqual(tokenized_chat, expected_tokens)

@require_tokenizers
class OPTTokenizationTest(unittest.TestCase):

    def test_serialize_deserialize_fast_opt(self):
        if False:
            while True:
                i = 10
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m', from_slow=True)
        text = 'A photo of a cat'
        tokens_ids = tokenizer.encode(text)
        self.assertEqual(tokens_ids, [2, 250, 1345, 9, 10, 4758])
        tokenizer.save_pretrained('test_opt')
        tokenizer = AutoTokenizer.from_pretrained('./test_opt')
        tokens_ids = tokenizer.encode(text)
        self.assertEqual(tokens_ids, [2, 250, 1345, 9, 10, 4758])

    def test_fast_slow_equivalence(self):
        if False:
            print('Hello World!')
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m', use_slow=True)
        text = 'A photo of a cat'
        tokens_ids = tokenizer.encode(text)
        self.assertEqual(tokens_ids, [2, 250, 1345, 9, 10, 4758])

    @unittest.skip('This test is failing because of a bug in the fast tokenizer')
    def test_users_can_modify_bos(self):
        if False:
            return 10
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m', from_slow=True)
        tokenizer.bos_token = 'bos'
        tokenizer.bos_token_id = tokenizer.get_vocab()['bos']
        text = 'A photo of a cat'
        tokens_ids = tokenizer.encode(text)
        self.assertEqual(tokens_ids, [31957, 250, 1345, 9, 10, 4758])
        tokenizer.save_pretrained('./tok')
        tokenizer = AutoTokenizer.from_pretrained('./tok')
        self.assertTrue(tokenizer.is_fast)
        tokens_ids = tokenizer.encode(text)
        self.assertEqual(tokens_ids, [31957, 250, 1345, 9, 10, 4758])