import json
import os
import unittest
from transformers.models.fsmt.tokenization_fsmt import VOCAB_FILES_NAMES, FSMTTokenizer
from transformers.testing_utils import slow
from transformers.utils import cached_property
from ...test_tokenization_common import TokenizerTesterMixin
FSMT_TINY2 = 'stas/tiny-wmt19-en-ru'

class FSMTTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = FSMTTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        if False:
            return 10
        super().setUp()
        vocab = ['l', 'o', 'w', 'e', 'r', 's', 't', 'i', 'd', 'n', 'w</w>', 'r</w>', 't</w>', 'lo', 'low', 'er</w>', 'low</w>', 'lowest</w>', 'newer</w>', 'wider</w>', '<unk>']
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ['l o 123', 'lo w 1456', 'e r</w> 1789', '']
        self.langs = ['en', 'ru']
        config = {'langs': self.langs, 'src_vocab_size': 10, 'tgt_vocab_size': 20}
        self.src_vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['src_vocab_file'])
        self.tgt_vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['tgt_vocab_file'])
        config_file = os.path.join(self.tmpdirname, 'tokenizer_config.json')
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['merges_file'])
        with open(self.src_vocab_file, 'w') as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.tgt_vocab_file, 'w') as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.merges_file, 'w') as fp:
            fp.write('\n'.join(merges))
        with open(config_file, 'w') as fp:
            fp.write(json.dumps(config))

    @cached_property
    def tokenizer_ru_en(self):
        if False:
            i = 10
            return i + 15
        return FSMTTokenizer.from_pretrained('facebook/wmt19-ru-en')

    @cached_property
    def tokenizer_en_ru(self):
        if False:
            i = 10
            return i + 15
        return FSMTTokenizer.from_pretrained('facebook/wmt19-en-ru')

    def test_online_tokenizer_config(self):
        if False:
            i = 10
            return i + 15
        "this just tests that the online tokenizer files get correctly fetched and\n        loaded via its tokenizer_config.json and it's not slow so it's run by normal CI\n        "
        tokenizer = FSMTTokenizer.from_pretrained(FSMT_TINY2)
        self.assertListEqual([tokenizer.src_lang, tokenizer.tgt_lang], ['en', 'ru'])
        self.assertEqual(tokenizer.src_vocab_size, 21)
        self.assertEqual(tokenizer.tgt_vocab_size, 21)

    def test_full_tokenizer(self):
        if False:
            while True:
                i = 10
        'Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt'
        tokenizer = FSMTTokenizer(self.langs, self.src_vocab_file, self.tgt_vocab_file, self.merges_file)
        text = 'lower'
        bpe_tokens = ['low', 'er</w>']
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + ['<unk>']
        input_bpe_tokens = [14, 15, 20]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @slow
    def test_sequence_builders(self):
        if False:
            while True:
                i = 10
        tokenizer = self.tokenizer_ru_en
        text = tokenizer.encode('sequence builders', add_special_tokens=False)
        text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
        assert encoded_sentence == text + [2]
        assert encoded_pair == text + [2] + text_2 + [2]

    @slow
    def test_match_encode_decode(self):
        if False:
            i = 10
            return i + 15
        tokenizer_enc = self.tokenizer_en_ru
        tokenizer_dec = self.tokenizer_ru_en
        targets = [["Here's a little song I wrote. Don't worry, be happy.", [2470, 39, 11, 2349, 7222, 70, 5979, 7, 8450, 1050, 13160, 5, 26, 6445, 7, 2]], ["This is it. No more. I'm done!", [132, 21, 37, 7, 1434, 86, 7, 70, 6476, 1305, 427, 2]]]
        for (src_text, tgt_input_ids) in targets:
            encoded_ids = tokenizer_enc.encode(src_text, return_tensors=None)
            self.assertListEqual(encoded_ids, tgt_input_ids)
            decoded_text = tokenizer_dec.decode(encoded_ids, skip_special_tokens=True)
            self.assertEqual(decoded_text, src_text)

    @slow
    def test_tokenizer_lower(self):
        if False:
            while True:
                i = 10
        tokenizer = FSMTTokenizer.from_pretrained('facebook/wmt19-ru-en', do_lower_case=True)
        tokens = tokenizer.tokenize('USA is United States of America')
        expected = ['us', 'a</w>', 'is</w>', 'un', 'i', 'ted</w>', 'st', 'ates</w>', 'of</w>', 'am', 'er', 'ica</w>']
        self.assertListEqual(tokens, expected)

    @unittest.skip('FSMTConfig.__init__  requires non-optional args')
    def test_torch_encode_plus_sent_to_model(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip('FSMTConfig.__init__  requires non-optional args')
    def test_np_encode_plus_sent_to_model(self):
        if False:
            return 10
        pass