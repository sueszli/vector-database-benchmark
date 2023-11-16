import json
import os
import re
import shutil
import tempfile
import unittest
from typing import Tuple
from transformers import AddedToken, BatchEncoding, ByT5Tokenizer
from transformers.utils import cached_property, is_tf_available, is_torch_available
from ...test_tokenization_common import TokenizerTesterMixin
if is_torch_available():
    FRAMEWORK = 'pt'
elif is_tf_available():
    FRAMEWORK = 'tf'
else:
    FRAMEWORK = 'jax'

class ByT5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = ByT5Tokenizer
    test_rust_tokenizer = False

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        tokenizer = ByT5Tokenizer()
        tokenizer.save_pretrained(self.tmpdirname)

    @cached_property
    def t5_base_tokenizer(self):
        if False:
            while True:
                i = 10
        return ByT5Tokenizer.from_pretrained('google/byt5-small')

    def get_tokenizer(self, **kwargs) -> ByT5Tokenizer:
        if False:
            for i in range(10):
                print('nop')
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5) -> Tuple[str, list]:
        if False:
            while True:
                i = 10
        toks = []
        for i in range(len(tokenizer)):
            try:
                tok = tokenizer.decode([i], clean_up_tokenization_spaces=False)
            except UnicodeDecodeError:
                pass
            toks.append((i, tok))
        toks = list(filter(lambda t: re.match('^[ a-zA-Z]+$', t[1]), toks))
        toks = list(filter(lambda t: [t[0]] == tokenizer.encode(t[1], add_special_tokens=False), toks))
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and (len(toks) > 0):
            while len(toks) < min_length:
                toks = toks + toks
        toks_ids = [t[0] for t in toks]
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if ' ' not in output_txt and len(toks_ids) > 1:
            output_txt = tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False) + ' ' + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
        if with_prefix_space:
            output_txt = ' ' + output_txt
        output_ids = tokenizer.encode(output_txt, add_special_tokens=False)
        return (output_txt, output_ids)

    def test_eos_treatment(self):
        if False:
            print('Hello World!')
        tokenizer = self.t5_base_tokenizer
        batch_with_eos_added = tokenizer(['hi</s>', 'I went to the gym</s>', '</s>'])
        batch_without_eos_added = tokenizer(['hi', 'I went to the gym', ''])
        self.assertListEqual(batch_with_eos_added['input_ids'], batch_without_eos_added['input_ids'])

    def test_multibytes_char(self):
        if False:
            while True:
                i = 10
        tokenizer = self.t5_base_tokenizer
        src_text = 'Unicode €.'
        encoded = tokenizer(src_text)
        encoded_ids = [88, 113, 108, 102, 114, 103, 104, 35, 229, 133, 175, 49, 1]
        self.assertEqual(encoded['input_ids'], encoded_ids)
        decoded = tokenizer.decode(encoded_ids)
        self.assertEqual(decoded, 'Unicode €.</s>')
        encoded = tokenizer('e è é ê ë')
        encoded_ids = [104, 35, 198, 171, 35, 198, 172, 35, 198, 173, 35, 198, 174, 1]
        self.assertEqual(encoded['input_ids'], encoded_ids)
        decoded = tokenizer.decode(encoded_ids)
        self.assertEqual(decoded, 'e è é ê ë</s>')
        self.assertEqual(tokenizer.decode(tokenizer.encode('e è é ê ë')), 'e è é ê ë</s>')

    def test_prepare_batch_integration(self):
        if False:
            print('Hello World!')
        tokenizer = self.t5_base_tokenizer
        src_text = ['A long paragraph for summarization.', 'Another paragraph for summarization.']
        expected_src_tokens = [68, 35, 111, 114, 113, 106, 35, 115, 100, 117, 100, 106, 117, 100, 115, 107, 35, 105, 114, 117, 35, 118, 120, 112, 112, 100, 117, 108, 125, 100, 119, 108, 114, 113, 49, 1, 0]
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        self.assertIsInstance(batch, BatchEncoding)
        if FRAMEWORK != 'jax':
            result = list(batch.input_ids.numpy()[0])
        else:
            result = list(batch.input_ids.tolist()[0])
        self.assertListEqual(expected_src_tokens, result)
        self.assertEqual((2, 37), batch.input_ids.shape)
        self.assertEqual((2, 37), batch.attention_mask.shape)

    def test_empty_target_text(self):
        if False:
            while True:
                i = 10
        tokenizer = self.t5_base_tokenizer
        src_text = ['A long paragraph for summarization.', 'Another paragraph for summarization.']
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        self.assertNotIn('decoder_input_ids', batch)
        self.assertNotIn('decoder_attention_mask', batch)

    def test_max_length_integration(self):
        if False:
            return 10
        tokenizer = self.t5_base_tokenizer
        tgt_text = ['Summary of the text.', 'Another summary.']
        targets = tokenizer(text_target=tgt_text, max_length=32, padding='max_length', truncation=True, return_tensors=FRAMEWORK)
        self.assertEqual(32, targets['input_ids'].shape[1])

    def test_eos_in_input(self):
        if False:
            while True:
                i = 10
        tokenizer = self.t5_base_tokenizer
        src_text = ['A long paragraph for summarization. </s>']
        tgt_text = ['Summary of the text. </s>']
        expected_src_tokens = [68, 35, 111, 114, 113, 106, 35, 115, 100, 117, 100, 106, 117, 100, 115, 107, 35, 105, 114, 117, 35, 118, 120, 112, 112, 100, 117, 108, 125, 100, 119, 108, 114, 113, 49, 35, 1]
        expected_tgt_tokens = [86, 120, 112, 112, 100, 117, 124, 35, 114, 105, 35, 119, 107, 104, 35, 119, 104, 123, 119, 49, 35, 1]
        batch = tokenizer(src_text, text_target=tgt_text)
        self.assertEqual(expected_src_tokens, batch['input_ids'][0])
        self.assertEqual(expected_tgt_tokens, batch['labels'][0])

    def test_save_and_load_tokenizer(self):
        if False:
            print('Hello World!')
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                self.assertNotEqual(tokenizer.model_max_length, 42)
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                tmpdirname = tempfile.mkdtemp()
                sample_text = ' He is very happy, UNwantéd,running'
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                tokenizer.save_pretrained(tmpdirname)
                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                self.assertListEqual(before_tokens, after_tokens)
                shutil.rmtree(tmpdirname)
        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                tmpdirname = tempfile.mkdtemp()
                sample_text = ' He is very happy, UNwantéd,running'
                tokenizer.add_tokens(['bim', 'bambam'])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append('new_additional_special_token')
                tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens}, replace_additional_special_tokens=False)
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                tokenizer.save_pretrained(tmpdirname)
                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                self.assertListEqual(before_tokens, after_tokens)
                self.assertIn('new_additional_special_token', after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)
                tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)
                shutil.rmtree(tmpdirname)

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        if False:
            return 10
        tokenizer_list = []
        if self.test_slow_tokenizer:
            tokenizer_list.append((self.tokenizer_class, self.get_tokenizer()))
        if self.test_rust_tokenizer:
            tokenizer_list.append((self.rust_tokenizer_class, self.get_rust_tokenizer()))
        for (tokenizer_class, tokenizer_utils) in tokenizer_list:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer_utils.save_pretrained(tmp_dir)
                with open(os.path.join(tmp_dir, 'special_tokens_map.json'), encoding='utf-8') as json_file:
                    special_tokens_map = json.load(json_file)
                with open(os.path.join(tmp_dir, 'tokenizer_config.json'), encoding='utf-8') as json_file:
                    tokenizer_config = json.load(json_file)
                added_tokens_extra_ids = [f'<extra_id_{i}>' for i in range(125)]
                special_tokens_map['additional_special_tokens'] = added_tokens_extra_ids + ['an_additional_special_token']
                tokenizer_config['additional_special_tokens'] = added_tokens_extra_ids + ['an_additional_special_token']
                with open(os.path.join(tmp_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as outfile:
                    json.dump(special_tokens_map, outfile)
                with open(os.path.join(tmp_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as outfile:
                    json.dump(tokenizer_config, outfile)
                tokenizer_without_change_in_init = tokenizer_class.from_pretrained(tmp_dir)
                self.assertIn('an_additional_special_token', tokenizer_without_change_in_init.additional_special_tokens)
                self.assertEqual(['an_additional_special_token'], tokenizer_without_change_in_init.convert_ids_to_tokens(tokenizer_without_change_in_init.convert_tokens_to_ids(['an_additional_special_token'])))
                new_added_tokens = added_tokens_extra_ids + [AddedToken('a_new_additional_special_token', lstrip=True)]
                tokenizer = tokenizer_class.from_pretrained(tmp_dir, additional_special_tokens=new_added_tokens)
                self.assertIn('a_new_additional_special_token', tokenizer.additional_special_tokens)
                self.assertEqual(['a_new_additional_special_token'], tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(['a_new_additional_special_token'])))

    def test_decode_single_bytes(self):
        if False:
            while True:
                i = 10
        tokenizer_list = []
        if self.test_slow_tokenizer:
            tokenizer_list.append((self.tokenizer_class, self.get_tokenizer()))
        if self.test_rust_tokenizer:
            tokenizer_list.append((self.rust_tokenizer_class, self.get_rust_tokenizer()))
        for (tokenizer_class, tokenizer_utils) in tokenizer_list:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer_utils.save_pretrained(tmp_dir)
                tokenizer = tokenizer_class.from_pretrained(tmp_dir)
                self.assertTrue(tokenizer.decode([255]) == '')

    def test_pretrained_model_lists(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_get_vocab(self):
        if False:
            while True:
                i = 10
        pass

    def test_pretokenized_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_conversion_reversible(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_convert_tokens_to_string_format(self):
        if False:
            i = 10
            return i + 15
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                tokens = ['t', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 't', 'e', 'x', 't', '</s>']
                string = tokenizer.convert_tokens_to_string(tokens)
                self.assertIsInstance(string, str)

    def test_tokenizers_common_ids_setters(self):
        if False:
            while True:
                i = 10
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f'{tokenizer.__class__.__name__}'):
                attributes_list = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token']
                token_id_to_test_setters = 0
                token_to_test_setters = tokenizer.convert_ids_to_tokens(token_id_to_test_setters, skip_special_tokens=False)
                for attr in attributes_list:
                    setattr(tokenizer, attr + '_id', None)
                    self.assertEqual(getattr(tokenizer, attr), None)
                    self.assertEqual(getattr(tokenizer, attr + '_id'), None)
                    setattr(tokenizer, attr + '_id', token_id_to_test_setters)
                    self.assertEqual(getattr(tokenizer, attr), token_to_test_setters)
                    self.assertEqual(getattr(tokenizer, attr + '_id'), token_id_to_test_setters)
                setattr(tokenizer, 'additional_special_tokens_ids', [])
                self.assertListEqual(getattr(tokenizer, 'additional_special_tokens'), [])
                self.assertListEqual(getattr(tokenizer, 'additional_special_tokens_ids'), [])
                setattr(tokenizer, 'additional_special_tokens_ids', [token_id_to_test_setters])
                self.assertListEqual(getattr(tokenizer, 'additional_special_tokens'), [token_to_test_setters])
                self.assertListEqual(getattr(tokenizer, 'additional_special_tokens_ids'), [token_id_to_test_setters])