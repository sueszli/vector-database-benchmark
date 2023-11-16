import itertools
import json
import os
import unittest
from transformers import AddedToken, RobertaTokenizer, RobertaTokenizerFast
from transformers.models.roberta.tokenization_roberta import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers, slow
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class RobertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = RobertaTokenizer
    rust_tokenizer_class = RobertaTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_kwargs = {'cls_token': '<s>'}

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        vocab = ['l', 'o', 'w', 'e', 'r', 's', 't', 'i', 'd', 'n', 'Ġ', 'Ġl', 'Ġn', 'Ġlo', 'Ġlow', 'er', 'Ġlowest', 'Ġnewer', 'Ġwider', '<unk>']
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
            return 10
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        if False:
            return 10
        kwargs.update(self.special_tokens_map)
        return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            return 10
        input_text = 'lower newer'
        output_text = 'lower newer'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            print('Hello World!')
        tokenizer = self.tokenizer_class(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = 'lower newer'
        bpe_tokens = ['l', 'o', 'w', 'er', 'Ġ', 'n', 'e', 'w', 'er']
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def roberta_dict_integration_testing(self):
        if False:
            print('Hello World!')
        tokenizer = self.get_tokenizer()
        self.assertListEqual(tokenizer.encode('Hello world!', add_special_tokens=False), [0, 31414, 232, 328, 2])
        self.assertListEqual(tokenizer.encode('Hello world! cécé herlolip 418', add_special_tokens=False), [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2])

    @slow
    def test_sequence_builders(self):
        if False:
            i = 10
            return i + 15
        tokenizer = self.tokenizer_class.from_pretrained('roberta-base')
        text = tokenizer.encode('sequence builders', add_special_tokens=False)
        text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
        encoded_text_from_decode = tokenizer.encode('sequence builders', add_special_tokens=True, add_prefix_space=False)
        encoded_pair_from_decode = tokenizer.encode('sequence builders', 'multi-sequence build', add_special_tokens=True, add_prefix_space=False)
        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
        assert encoded_sentence == encoded_text_from_decode
        assert encoded_pair == encoded_pair_from_decode

    def test_space_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = self.get_tokenizer()
        sequence = 'Encode this sequence.'
        space_encoding = tokenizer.byte_encoder[' '.encode('utf-8')[0]]
        encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=False)
        first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
        self.assertNotEqual(first_char, space_encoding)
        encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=True)
        first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
        self.assertEqual(first_char, space_encoding)
        tokenizer.add_special_tokens({'bos_token': '<s>'})
        encoded = tokenizer.encode(sequence, add_special_tokens=True)
        first_char = tokenizer.convert_ids_to_tokens(encoded[1])[0]
        self.assertNotEqual(first_char, space_encoding)
        mask = '<mask>'
        tokenizer.add_special_tokens({'mask_token': AddedToken(mask, lstrip=True, rstrip=False)})
        mask_ind = tokenizer.convert_tokens_to_ids(mask)
        sequence = 'Encode <mask> sequence'
        sequence_nospace = 'Encode <mask>sequence'
        encoded = tokenizer.encode(sequence)
        mask_loc = encoded.index(mask_ind)
        first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
        self.assertEqual(first_char, space_encoding)
        encoded = tokenizer.encode(sequence_nospace)
        mask_loc = encoded.index(mask_ind)
        first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
        self.assertNotEqual(first_char, space_encoding)

    def test_pretokenized_inputs(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_embeded_special_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        for (tokenizer, pretrained_name, kwargs) in self.tokenizers_list:
            with self.subTest(f'{tokenizer.__class__.__name__} ({pretrained_name})'):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                sentence = 'A, <mask> AllenNLP sentence.'
                tokens_r = tokenizer_r.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)
                tokens_p = tokenizer_p.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)
                self.assertEqual(sum(tokens_r['token_type_ids']), sum(tokens_p['token_type_ids']))
                self.assertEqual(sum(tokens_r['attention_mask']) / len(tokens_r['attention_mask']), sum(tokens_p['attention_mask']) / len(tokens_p['attention_mask']))
                tokens_r_str = tokenizer_r.convert_ids_to_tokens(tokens_r['input_ids'])
                tokens_p_str = tokenizer_p.convert_ids_to_tokens(tokens_p['input_ids'])
                self.assertSequenceEqual(tokens_p['input_ids'], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])
                self.assertSequenceEqual(tokens_r['input_ids'], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])
                self.assertSequenceEqual(tokens_p_str, ['<s>', 'A', ',', '<mask>', 'ĠAllen', 'N', 'LP', 'Ġsentence', '.', '</s>'])
                self.assertSequenceEqual(tokens_r_str, ['<s>', 'A', ',', '<mask>', 'ĠAllen', 'N', 'LP', 'Ġsentence', '.', '</s>'])

    def test_change_add_prefix_space_and_trim_offsets_args(self):
        if False:
            return 10
        for (trim_offsets, add_prefix_space) in itertools.product([True, False], repeat=2):
            tokenizer_r = self.rust_tokenizer_class.from_pretrained(self.tmpdirname, use_fast=True, add_prefix_space=add_prefix_space, trim_offsets=trim_offsets)
            pre_tokenizer_state = json.loads(tokenizer_r.backend_tokenizer.pre_tokenizer.__getstate__())
            post_processor_state = json.loads(tokenizer_r.backend_tokenizer.post_processor.__getstate__())
            self.assertEqual(pre_tokenizer_state['add_prefix_space'], add_prefix_space)
            self.assertEqual(post_processor_state['add_prefix_space'], add_prefix_space)
            self.assertEqual(post_processor_state['trim_offsets'], trim_offsets)

    def test_offsets_mapping_with_different_add_prefix_space_and_trim_space_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        for (tokenizer, pretrained_name, kwargs) in self.tokenizers_list:
            with self.subTest(f'{tokenizer.__class__.__name__} ({pretrained_name})'):
                text_of_1_token = 'hello'
                text = f'{text_of_1_token} {text_of_1_token}'
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True, add_prefix_space=True, trim_offsets=True)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (len(text_of_1_token) + 1, len(text_of_1_token) + 1 + len(text_of_1_token)))
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True, add_prefix_space=False, trim_offsets=True)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (len(text_of_1_token) + 1, len(text_of_1_token) + 1 + len(text_of_1_token)))
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True, add_prefix_space=True, trim_offsets=False)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (len(text_of_1_token), len(text_of_1_token) + 1 + len(text_of_1_token)))
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True, add_prefix_space=False, trim_offsets=False)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (len(text_of_1_token), len(text_of_1_token) + 1 + len(text_of_1_token)))
                text = f' {text}'
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True, add_prefix_space=False, trim_offsets=True)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (1, 1 + len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (1 + len(text_of_1_token) + 1, 1 + len(text_of_1_token) + 1 + len(text_of_1_token)))
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True, add_prefix_space=True, trim_offsets=False)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (0, 1 + len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (1 + len(text_of_1_token), 1 + len(text_of_1_token) + 1 + len(text_of_1_token)))
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True, add_prefix_space=False, trim_offsets=False)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (0, 1 + len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (1 + len(text_of_1_token), 1 + len(text_of_1_token) + 1 + len(text_of_1_token)))