import concurrent.futures
import json
import os
import shutil
import tempfile
import unittest
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers
from ..test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class PreTrainedTokenizationFastTest(TokenizerTesterMixin, unittest.TestCase):
    rust_tokenizer_class = PreTrainedTokenizerFast
    test_slow_tokenizer = False
    test_rust_tokenizer = True
    from_pretrained_vocab_key = 'tokenizer_file'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_rust_tokenizer = False
        super().setUp()
        self.test_rust_tokenizer = True
        model_paths = ['robot-test/dummy-tokenizer-fast', 'robot-test/dummy-tokenizer-wordlevel']
        self.bytelevel_bpe_model_name = 'SaulLu/dummy-tokenizer-bytelevel-bpe'
        self.tokenizers_list = [(PreTrainedTokenizerFast, model_path, {}) for model_path in model_paths]
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_paths[0])
        tokenizer.save_pretrained(self.tmpdirname)

    def test_tokenizer_mismatch_warning(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip('We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model')
    def test_encode_decode_with_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @unittest.skip('We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model')
    def test_added_tokens_serialization(self):
        if False:
            i = 10
            return i + 15
        pass

    @unittest.skip('We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any model')
    def test_additional_special_tokens_serialization(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_pretrained_model_lists(self):
        if False:
            print('Hello World!')
        pass

    def test_prepare_for_model(self):
        if False:
            return 10
        pass

    def test_rust_tokenizer_signature(self):
        if False:
            print('Hello World!')
        pass

    def test_training_new_tokenizer(self):
        if False:
            print('Hello World!')
        tmpdirname_orig = self.tmpdirname
        for (tokenizer, pretrained_name, kwargs) in self.tokenizers_list:
            with self.subTest(f'{tokenizer.__class__.__name__} ({pretrained_name})'):
                try:
                    self.tmpdirname = tempfile.mkdtemp()
                    tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                    tokenizer.save_pretrained(self.tmpdirname)
                    super().test_training_new_tokenizer()
                finally:
                    shutil.rmtree(self.tmpdirname)
                    self.tmpdirname = tmpdirname_orig

    def test_training_new_tokenizer_with_special_tokens_change(self):
        if False:
            print('Hello World!')
        tmpdirname_orig = self.tmpdirname
        for (tokenizer, pretrained_name, kwargs) in self.tokenizers_list:
            with self.subTest(f'{tokenizer.__class__.__name__} ({pretrained_name})'):
                try:
                    self.tmpdirname = tempfile.mkdtemp()
                    tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                    tokenizer.save_pretrained(self.tmpdirname)
                    super().test_training_new_tokenizer_with_special_tokens_change()
                finally:
                    shutil.rmtree(self.tmpdirname)
                    self.tmpdirname = tmpdirname_orig

    def test_training_new_tokenizer_with_bytelevel(self):
        if False:
            return 10
        tokenizer = self.rust_tokenizer_class.from_pretrained(self.bytelevel_bpe_model_name)
        toy_text_iterator = ('a' for _ in range(1000))
        new_tokenizer = tokenizer.train_new_from_iterator(text_iterator=toy_text_iterator, length=1000, vocab_size=50)
        encoding_ids = new_tokenizer.encode('a🤗')
        self.assertEqual(encoding_ids, [64, 172, 253, 97, 245])

    def test_init_from_tokenizers_model(self):
        if False:
            return 10
        from tokenizers import Tokenizer
        sentences = ["Hello, y'all!", 'How are you 😁 ? There should not be any issue right?']
        tokenizer = Tokenizer.from_pretrained('t5-base')
        tokenizer.enable_padding(pad_id=0, pad_token='<pad>', length=512, pad_to_multiple_of=8)
        self.assertEqual(tokenizer.padding, {'length': 512, 'pad_to_multiple_of': 8, 'pad_id': 0, 'pad_token': '<pad>', 'pad_type_id': 0, 'direction': 'right'})
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tmpdirname = tempfile.mkdtemp()
        fast_tokenizer.save_pretrained(tmpdirname)
        fast_from_saved = PreTrainedTokenizerFast.from_pretrained(tmpdirname)
        for tok in [fast_tokenizer, fast_from_saved]:
            self.assertEqual(tok.pad_token_id, 0)
            self.assertEqual(tok.padding_side, 'right')
            self.assertEqual(tok.pad_token, '<pad>')
            self.assertEqual(tok.init_kwargs['max_length'], 512)
            self.assertEqual(tok.init_kwargs['pad_to_multiple_of'], 8)
            self.assertEqual(tok(sentences, padding=True), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1, 0, 0, 0, 0, 0, 0, 0, 0], [571, 33, 25, 3, 2, 3, 58, 290, 225, 59, 36, 136, 962, 269, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})
        tokenizer.enable_truncation(8, stride=0, strategy='longest_first', direction='right')
        self.assertEqual(tokenizer.truncation, {'max_length': 8, 'stride': 0, 'strategy': 'longest_first', 'direction': 'right'})
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tmpdirname = tempfile.mkdtemp()
        fast_tokenizer.save_pretrained(tmpdirname)
        fast_from_saved = PreTrainedTokenizerFast.from_pretrained(tmpdirname)
        for tok in [fast_tokenizer, fast_from_saved]:
            self.assertEqual(tok.truncation_side, 'right')
            self.assertEqual(tok.init_kwargs['truncation_strategy'], 'longest_first')
            self.assertEqual(tok.init_kwargs['max_length'], 8)
            self.assertEqual(tok.init_kwargs['stride'], 0)
            self.assertEqual(tok(sentences, truncation=True, max_length=8), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1], [571, 33, 25, 3, 2, 3, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]})

@require_tokenizers
class TokenizerVersioningTest(unittest.TestCase):

    def test_local_versioning(self):
        if False:
            i = 10
            return i + 15
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        json_tokenizer = json.loads(tokenizer._tokenizer.to_str())
        json_tokenizer['model']['vocab']['huggingface'] = len(tokenizer)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.init_kwargs['fast_tokenizer_files'] = ['tokenizer.4.0.0.json']
            tokenizer.save_pretrained(tmp_dir)
            json.dump(json_tokenizer, open(os.path.join(tmp_dir, 'tokenizer.4.0.0.json'), 'w'))
            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertEqual(len(new_tokenizer), len(tokenizer) + 1)
            json_tokenizer = json.loads(new_tokenizer._tokenizer.to_str())
            self.assertIn('huggingface', json_tokenizer['model']['vocab'])
            shutil.move(os.path.join(tmp_dir, 'tokenizer.4.0.0.json'), os.path.join(tmp_dir, 'tokenizer.42.0.0.json'))
            tokenizer.init_kwargs['fast_tokenizer_files'] = ['tokenizer.42.0.0.json']
            tokenizer.save_pretrained(tmp_dir)
            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertEqual(len(new_tokenizer), len(tokenizer))
            json_tokenizer = json.loads(new_tokenizer._tokenizer.to_str())
            self.assertNotIn('huggingface', json_tokenizer['model']['vocab'])

    def test_repo_versioning(self):
        if False:
            return 10
        repo = 'hf-internal-testing/test-two-tokenizers'
        tokenizer = AutoTokenizer.from_pretrained(repo)
        self.assertEqual(len(tokenizer), 28997)
        json_tokenizer = json.loads(tokenizer._tokenizer.to_str())
        self.assertIn('huggingface', json_tokenizer['model']['vocab'])
        import transformers as old_transformers
        old_transformers.tokenization_utils_base.__version__ = '3.0.0'
        old_tokenizer = old_transformers.models.auto.AutoTokenizer.from_pretrained(repo)
        self.assertEqual(len(old_tokenizer), 28996)
        json_tokenizer = json.loads(old_tokenizer._tokenizer.to_str())
        self.assertNotIn('huggingface', json_tokenizer['model']['vocab'])

@require_tokenizers
class ReduceMutableBorrowTests(unittest.TestCase):

    def test_async_share_tokenizer(self):
        if False:
            print('Hello World!')
        tokenizer = PreTrainedTokenizerFast.from_pretrained('robot-test/dummy-tokenizer-wordlevel')
        text = 'The Matrix is a 1999 science fiction action film.'
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.fetch, tokenizer, text) for i in range(10)]
            return_value = [future.result() for future in futures]
            self.assertEqual(return_value, [[1, 10, 0, 8, 0, 18, 0, 0, 0, 2] for i in range(10)])

    def fetch(self, tokenizer, text):
        if False:
            while True:
                i = 10
        return tokenizer.encode(text, truncation='longest_first', padding='longest')