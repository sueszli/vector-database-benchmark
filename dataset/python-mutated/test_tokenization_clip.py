import json
import os
import unittest
from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers.models.clip.tokenization_clip import VOCAB_FILES_NAMES
from transformers.testing_utils import require_ftfy, require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class CLIPTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = CLIPTokenizer
    rust_tokenizer_class = CLIPTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_kwargs = {}
    test_seq2seq = False

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        vocab = ['l', 'o', 'w', 'e', 'r', 's', 't', 'i', 'd', 'n', 'lo', 'l</w>', 'w</w>', 'r</w>', 't</w>', 'low</w>', 'er</w>', 'lowest</w>', 'newer</w>', 'wider', '<unk>', '<|startoftext|>', '<|endoftext|>']
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ['#version: 0.2', 'l o', 'lo w</w>', 'e r</w>']
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
        return CLIPTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs.update(self.special_tokens_map)
        return CLIPTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            i = 10
            return i + 15
        input_text = 'lower newer'
        output_text = 'lower newer'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = CLIPTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = 'lower newer'
        bpe_tokens = ['lo', 'w', 'er</w>', 'n', 'e', 'w', 'er</w>']
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [10, 2, 16, 9, 3, 2, 16, 20]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @require_ftfy
    def test_check_encoding_slow_fast(self):
        if False:
            for i in range(10):
                print('nop')
        for (tokenizer, pretrained_name, kwargs) in self.tokenizers_list:
            with self.subTest(f'{tokenizer.__class__.__name__} ({pretrained_name})'):
                tokenizer_s = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                text = "A\n'll 11p223RF☆ho!!to?'d'd''d of a cat to-$''d."
                text_tokenized_s = tokenizer_s.tokenize(text)
                text_tokenized_r = tokenizer_r.tokenize(text)
                self.assertListEqual(text_tokenized_s, text_tokenized_r)
                text = 'xãy' + ' ' + 'xãy'
                text_tokenized_s = tokenizer_s.tokenize(text)
                text_tokenized_r = tokenizer_r.tokenize(text)
                self.assertListEqual(text_tokenized_s, text_tokenized_r)
                spaces_unicodes = ['\t', '\x0b', '\x0c', ' ', '\u200e', '\u200f']
                for unicode_seq in spaces_unicodes:
                    text_tokenized_s = tokenizer_s.tokenize(unicode_seq)
                    text_tokenized_r = tokenizer_r.tokenize(unicode_seq)
                    self.assertListEqual(text_tokenized_s, text_tokenized_r)
                line_break_unicodes = ['\n', '\r\n', '\r', '\r', '\r', '\u2028', '\u2029']
                for unicode_seq in line_break_unicodes:
                    text_tokenized_s = tokenizer_s.tokenize(unicode_seq)
                    text_tokenized_r = tokenizer_r.tokenize(unicode_seq)
                    self.assertListEqual(text_tokenized_s, text_tokenized_r)

    def test_offsets_mapping_with_different_add_prefix_space_argument(self):
        if False:
            print('Hello World!')
        for (tokenizer, pretrained_name, kwargs) in self.tokenizers_list:
            with self.subTest(f'{tokenizer.__class__.__name__} ({pretrained_name})'):
                text_of_1_token = 'hello'
                text = f'{text_of_1_token} {text_of_1_token}'
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (len(text_of_1_token) + 1, len(text_of_1_token) + 1 + len(text_of_1_token)))
                text = f' {text}'
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, use_fast=True)
                encoding = tokenizer_r(text, return_offsets_mapping=True, add_special_tokens=False)
                self.assertEqual(encoding.offset_mapping[0], (1, 1 + len(text_of_1_token)))
                self.assertEqual(encoding.offset_mapping[1], (1 + len(text_of_1_token) + 1, 1 + len(text_of_1_token) + 1 + len(text_of_1_token)))

    def test_log_warning(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError) as context:
            self.rust_tokenizer_class.from_pretrained('robot-test/old-clip-tokenizer')
        self.assertTrue(context.exception.args[0].startswith('The `backend_tokenizer` provided does not match the expected format.'))

    @require_ftfy
    def test_tokenization_python_rust_equals(self):
        if False:
            i = 10
            return i + 15
        super().test_tokenization_python_rust_equals()

    def test_added_tokens_do_lower_case(self):
        if False:
            while True:
                i = 10
        pass