import json
import os
import unittest
from transformers.models.gpt_neox_japanese.tokenization_gpt_neox_japanese import VOCAB_FILES_NAMES, GPTNeoXJapaneseTokenizer
from transformers.testing_utils import require_tokenizers, slow
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class GPTNeoXJapaneseTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = GPTNeoXJapaneseTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {'do_clean_text': False, 'add_prefix_space': False}

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        vocab_tokens = ['こん', 'こんに', 'にちは', 'ばんは', '世界,㔺界', '、', '。', '<BR>', '<SP>', '<TAB>', '<URL>', '<EMAIL>', '<TEL>', '<DATE>', '<PRICE>', '<BLOCK>', '<KIGOU>', '<U2000U2BFF>', '<|emoji1|>', '<unk>', '<|startoftext|>', '<|endoftext|>']
        emoji_tokens = {'emoji': {'\ud83d\ude00': '<|emoji1|>'}, 'emoji_inv': {'<|emoji1|>': '\ud83d\ude00'}}
        self.special_tokens_map = {'unk_token': '<unk>'}
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        self.emoji_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['emoji_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as vocab_writer:
            vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))
        with open(self.emoji_file, 'w') as emoji_writer:
            emoji_writer.write(json.dumps(emoji_tokens))

    def get_tokenizer(self, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.update(self.special_tokens_map)
        return GPTNeoXJapaneseTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            return 10
        input_text = 'こんにちは、世界。 \nこんばんは、㔺界。😀'
        output_text = 'こんにちは、世界。 \nこんばんは、世界。😀'
        return (input_text, output_text)

    def get_clean_sequence(self, tokenizer):
        if False:
            i = 10
            return i + 15
        (input_text, output_text) = self.get_input_output_texts(tokenizer)
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        return (text, ids)

    def test_pretokenized_inputs(self):
        if False:
            return 10
        pass

    def test_maximum_encoding_length_pair_input(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_maximum_encoding_length_single_input(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_full_tokenizer(self):
        if False:
            while True:
                i = 10
        tokenizer = self.get_tokenizer()
        input_text = 'こんにちは、世界。\u3000こんばんは、㔺界。'
        expected_token = ['こん', 'にちは', '、', '世界', '。', '<SP>', 'こん', 'ばんは', '、', '㔺界', '。']
        tokens = tokenizer.tokenize(input_text)
        self.assertListEqual(tokens, expected_token)
        expected_ids = [0, 2, 5, 4, 6, 8, 0, 3, 5, 4, 6]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(input_ids, expected_ids)
        input_tokens = tokens + [tokenizer.unk_token]
        expected_ids = [0, 2, 5, 4, 6, 8, 0, 3, 5, 4, 6, 19]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        self.assertListEqual(input_ids, expected_ids)

    @slow
    def test_sequence_builders(self):
        if False:
            return 10
        tokenizer = self.tokenizer_class.from_pretrained('abeja/gpt-neox-japanese-2.7b')
        ids_1 = tokenizer.encode('ありがとう。', add_special_tokens=False)
        ids_2 = tokenizer.encode('どういたしまして。', add_special_tokens=False)
        encoded_sentence = tokenizer.build_inputs_with_special_tokens(ids_1)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(ids_1, ids_2)
        assert encoded_sentence == ids_1
        assert encoded_pair == ids_1 + ids_2

    def test_conversion_reversible(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_padding_different_model_input_name(self):
        if False:
            for i in range(10):
                print('nop')
        pass