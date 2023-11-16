import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from huggingface_hub import HfFolder, delete_repo
from huggingface_hub.file_download import http_get
from requests.exceptions import HTTPError
from transformers import AlbertTokenizer, AutoTokenizer, BertTokenizer, BertTokenizerFast, GPT2TokenizerFast, is_tokenizers_available
from transformers.testing_utils import TOKEN, USER, is_staging_test, require_tokenizers
from transformers.tokenization_utils import Trie
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from test_module.custom_tokenization import CustomTokenizer
if is_tokenizers_available():
    from test_module.custom_tokenization_fast import CustomTokenizerFast

class TokenizerUtilTester(unittest.TestCase):

    def test_cached_files_are_used_when_internet_is_down(self):
        if False:
            i = 10
            return i + 15
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        _ = BertTokenizer.from_pretrained('hf-internal-testing/tiny-random-bert')
        with mock.patch('requests.Session.request', return_value=response_mock) as mock_head:
            _ = BertTokenizer.from_pretrained('hf-internal-testing/tiny-random-bert')
            mock_head.assert_called()

    @require_tokenizers
    def test_cached_files_are_used_when_internet_is_down_missing_files(self):
        if False:
            i = 10
            return i + 15
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        _ = GPT2TokenizerFast.from_pretrained('gpt2')
        with mock.patch('requests.Session.request', return_value=response_mock) as mock_head:
            _ = GPT2TokenizerFast.from_pretrained('gpt2')
            mock_head.assert_called()

    def test_legacy_load_from_one_file(self):
        if False:
            i = 10
            return i + 15
        try:
            tmp_file = tempfile.mktemp()
            with open(tmp_file, 'wb') as f:
                http_get('https://huggingface.co/albert-base-v1/resolve/main/spiece.model', f)
            _ = AlbertTokenizer.from_pretrained(tmp_file)
        finally:
            os.remove(tmp_file)
        if os.path.isfile('tokenizer.json'):
            return
        try:
            with open('tokenizer.json', 'wb') as f:
                http_get('https://huggingface.co/hf-internal-testing/tiny-random-bert/blob/main/tokenizer.json', f)
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')
            self.assertEqual(tokenizer.vocab_size, 1000)
        finally:
            os.remove('tokenizer.json')

    def test_legacy_load_from_url(self):
        if False:
            print('Hello World!')
        _ = AlbertTokenizer.from_pretrained('https://huggingface.co/albert-base-v1/resolve/main/spiece.model')

@is_staging_test
class TokenizerPushToHubTester(unittest.TestCase):
    vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]', 'bla', 'blou']

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        try:
            delete_repo(token=cls._token, repo_id='test-tokenizer')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='valid_org/test-tokenizer-org')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='test-dynamic-tokenizer')
        except HTTPError:
            pass

    def test_push_to_hub(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, 'vocab.txt')
            with open(vocab_file, 'w', encoding='utf-8') as vocab_writer:
                vocab_writer.write(''.join([x + '\n' for x in self.vocab_tokens]))
            tokenizer = BertTokenizer(vocab_file)
        tokenizer.push_to_hub('test-tokenizer', token=self._token)
        new_tokenizer = BertTokenizer.from_pretrained(f'{USER}/test-tokenizer')
        self.assertDictEqual(new_tokenizer.vocab, tokenizer.vocab)
        delete_repo(token=self._token, repo_id='test-tokenizer')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir, repo_id='test-tokenizer', push_to_hub=True, token=self._token)
        new_tokenizer = BertTokenizer.from_pretrained(f'{USER}/test-tokenizer')
        self.assertDictEqual(new_tokenizer.vocab, tokenizer.vocab)

    def test_push_to_hub_in_organization(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, 'vocab.txt')
            with open(vocab_file, 'w', encoding='utf-8') as vocab_writer:
                vocab_writer.write(''.join([x + '\n' for x in self.vocab_tokens]))
            tokenizer = BertTokenizer(vocab_file)
        tokenizer.push_to_hub('valid_org/test-tokenizer-org', token=self._token)
        new_tokenizer = BertTokenizer.from_pretrained('valid_org/test-tokenizer-org')
        self.assertDictEqual(new_tokenizer.vocab, tokenizer.vocab)
        delete_repo(token=self._token, repo_id='valid_org/test-tokenizer-org')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir, repo_id='valid_org/test-tokenizer-org', push_to_hub=True, token=self._token)
        new_tokenizer = BertTokenizer.from_pretrained('valid_org/test-tokenizer-org')
        self.assertDictEqual(new_tokenizer.vocab, tokenizer.vocab)

    @require_tokenizers
    def test_push_to_hub_dynamic_tokenizer(self):
        if False:
            i = 10
            return i + 15
        CustomTokenizer.register_for_auto_class()
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, 'vocab.txt')
            with open(vocab_file, 'w', encoding='utf-8') as vocab_writer:
                vocab_writer.write(''.join([x + '\n' for x in self.vocab_tokens]))
            tokenizer = CustomTokenizer(vocab_file)
        tokenizer.push_to_hub('test-dynamic-tokenizer', token=self._token)
        tokenizer = AutoTokenizer.from_pretrained(f'{USER}/test-dynamic-tokenizer', trust_remote_code=True)
        self.assertEqual(tokenizer.__class__.__name__, 'CustomTokenizer')
        CustomTokenizerFast.register_for_auto_class()
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, 'vocab.txt')
            with open(vocab_file, 'w', encoding='utf-8') as vocab_writer:
                vocab_writer.write(''.join([x + '\n' for x in self.vocab_tokens]))
            bert_tokenizer = BertTokenizerFast.from_pretrained(tmp_dir)
            bert_tokenizer.save_pretrained(tmp_dir)
            tokenizer = CustomTokenizerFast.from_pretrained(tmp_dir)
        tokenizer.push_to_hub('test-dynamic-tokenizer', token=self._token)
        tokenizer = AutoTokenizer.from_pretrained(f'{USER}/test-dynamic-tokenizer', trust_remote_code=True)
        self.assertEqual(tokenizer.__class__.__name__, 'CustomTokenizerFast')
        tokenizer = AutoTokenizer.from_pretrained(f'{USER}/test-dynamic-tokenizer', use_fast=False, trust_remote_code=True)
        self.assertEqual(tokenizer.__class__.__name__, 'CustomTokenizer')

class TrieTest(unittest.TestCase):

    def test_trie(self):
        if False:
            return 10
        trie = Trie()
        trie.add('Hello 友達')
        self.assertEqual(trie.data, {'H': {'e': {'l': {'l': {'o': {' ': {'友': {'達': {'': 1}}}}}}}}})
        trie.add('Hello')
        trie.data
        self.assertEqual(trie.data, {'H': {'e': {'l': {'l': {'o': {'': 1, ' ': {'友': {'達': {'': 1}}}}}}}}})

    def test_trie_split(self):
        if False:
            i = 10
            return i + 15
        trie = Trie()
        self.assertEqual(trie.split('[CLS] This is a extra_id_100'), ['[CLS] This is a extra_id_100'])
        trie.add('[CLS]')
        trie.add('extra_id_1')
        trie.add('extra_id_100')
        self.assertEqual(trie.split('[CLS] This is a extra_id_100'), ['[CLS]', ' This is a ', 'extra_id_100'])

    def test_trie_single(self):
        if False:
            for i in range(10):
                print('nop')
        trie = Trie()
        trie.add('A')
        self.assertEqual(trie.split('ABC'), ['A', 'BC'])
        self.assertEqual(trie.split('BCA'), ['BC', 'A'])

    def test_trie_final(self):
        if False:
            i = 10
            return i + 15
        trie = Trie()
        trie.add('TOKEN]')
        trie.add('[SPECIAL_TOKEN]')
        self.assertEqual(trie.split('This is something [SPECIAL_TOKEN]'), ['This is something ', '[SPECIAL_TOKEN]'])

    def test_trie_subtokens(self):
        if False:
            while True:
                i = 10
        trie = Trie()
        trie.add('A')
        trie.add('P')
        trie.add('[SPECIAL_TOKEN]')
        self.assertEqual(trie.split('This is something [SPECIAL_TOKEN]'), ['This is something ', '[SPECIAL_TOKEN]'])

    def test_trie_suffix_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        trie = Trie()
        trie.add('AB')
        trie.add('B')
        trie.add('C')
        self.assertEqual(trie.split('ABC'), ['AB', 'C'])

    def test_trie_skip(self):
        if False:
            return 10
        trie = Trie()
        trie.add('ABC')
        trie.add('B')
        trie.add('CD')
        self.assertEqual(trie.split('ABCD'), ['ABC', 'D'])

    def test_cut_text_hardening(self):
        if False:
            print('Hello World!')
        trie = Trie()
        parts = trie.cut_text('ABC', [0, 0, 2, 1, 2, 3])
        self.assertEqual(parts, ['AB', 'C'])