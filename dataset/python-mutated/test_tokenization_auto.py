import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import pytest
import transformers
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoTokenizer, BertConfig, BertTokenizer, BertTokenizerFast, CTRLTokenizer, GPT2Tokenizer, GPT2TokenizerFast, PreTrainedTokenizerFast, RobertaTokenizer, RobertaTokenizerFast, is_tokenizers_available
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING, get_tokenizer_config, tokenizer_class_from_name
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.testing_utils import DUMMY_DIFF_TOKENIZER_IDENTIFIER, DUMMY_UNKNOWN_IDENTIFIER, SMALL_MODEL_IDENTIFIER, RequestCounter, require_tokenizers, slow
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from test_module.custom_configuration import CustomConfig
from test_module.custom_tokenization import CustomTokenizer
if is_tokenizers_available():
    from test_module.custom_tokenization_fast import CustomTokenizerFast

class AutoTokenizerTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    @slow
    def test_tokenizer_from_pretrained(self):
        if False:
            for i in range(10):
                print('nop')
        for model_name in (x for x in BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys() if 'japanese' not in x):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
            self.assertGreater(len(tokenizer), 0)
        for model_name in GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP.keys():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast))
            self.assertGreater(len(tokenizer), 0)

    def test_tokenizer_from_pretrained_identifier(self):
        if False:
            while True:
                i = 10
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 12)

    def test_tokenizer_from_model_type(self):
        if False:
            print('Hello World!')
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 20)

    def test_tokenizer_from_tokenizer_class(self):
        if False:
            return 10
        config = AutoConfig.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER)
        self.assertIsInstance(config, RobertaConfig)
        tokenizer = AutoTokenizer.from_pretrained(DUMMY_DIFF_TOKENIZER_IDENTIFIER, config=config)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.assertEqual(tokenizer.vocab_size, 12)

    def test_tokenizer_from_type(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy('./tests/fixtures/vocab.txt', os.path.join(tmp_dir, 'vocab.txt'))
            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type='bert', use_fast=False)
            self.assertIsInstance(tokenizer, BertTokenizer)
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy('./tests/fixtures/vocab.json', os.path.join(tmp_dir, 'vocab.json'))
            shutil.copy('./tests/fixtures/merges.txt', os.path.join(tmp_dir, 'merges.txt'))
            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type='gpt2', use_fast=False)
            self.assertIsInstance(tokenizer, GPT2Tokenizer)

    @require_tokenizers
    def test_tokenizer_from_type_fast(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy('./tests/fixtures/vocab.txt', os.path.join(tmp_dir, 'vocab.txt'))
            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type='bert')
            self.assertIsInstance(tokenizer, BertTokenizerFast)
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy('./tests/fixtures/vocab.json', os.path.join(tmp_dir, 'vocab.json'))
            shutil.copy('./tests/fixtures/merges.txt', os.path.join(tmp_dir, 'merges.txt'))
            tokenizer = AutoTokenizer.from_pretrained(tmp_dir, tokenizer_type='gpt2')
            self.assertIsInstance(tokenizer, GPT2TokenizerFast)

    def test_tokenizer_from_type_incorrect_name(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            AutoTokenizer.from_pretrained('./', tokenizer_type='xxx')

    @require_tokenizers
    def test_tokenizer_identifier_with_correct_config(self):
        if False:
            while True:
                i = 10
        for tokenizer_class in [BertTokenizer, BertTokenizerFast, AutoTokenizer]:
            tokenizer = tokenizer_class.from_pretrained('wietsedv/bert-base-dutch-cased')
            self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
            if isinstance(tokenizer, BertTokenizer):
                self.assertEqual(tokenizer.basic_tokenizer.do_lower_case, False)
            else:
                self.assertEqual(tokenizer.do_lower_case, False)
            self.assertEqual(tokenizer.model_max_length, 512)

    @require_tokenizers
    def test_tokenizer_identifier_non_existent(self):
        if False:
            return 10
        for tokenizer_class in [BertTokenizer, BertTokenizerFast, AutoTokenizer]:
            with self.assertRaisesRegex(EnvironmentError, 'julien-c/herlolip-not-exists is not a local folder and is not a valid model identifier'):
                _ = tokenizer_class.from_pretrained('julien-c/herlolip-not-exists')

    def test_model_name_edge_cases_in_mappings(self):
        if False:
            print('Hello World!')
        tokenizers = TOKENIZER_MAPPING.values()
        tokenizer_names = []
        for (slow_tok, fast_tok) in tokenizers:
            if slow_tok is not None:
                tokenizer_names.append(slow_tok.__name__)
            if fast_tok is not None:
                tokenizer_names.append(fast_tok.__name__)
        for tokenizer_name in tokenizer_names:
            tokenizer_class_from_name(tokenizer_name)

    @require_tokenizers
    def test_from_pretrained_use_fast_toggle(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(AutoTokenizer.from_pretrained('bert-base-cased', use_fast=False), BertTokenizer)
        self.assertIsInstance(AutoTokenizer.from_pretrained('bert-base-cased'), BertTokenizerFast)

    @require_tokenizers
    def test_do_lower_case(self):
        if False:
            i = 10
            return i + 15
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=False)
        sample = 'Hello, world. How are you?'
        tokens = tokenizer.tokenize(sample)
        self.assertEqual('[UNK]', tokens[0])
        tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base', do_lower_case=False)
        tokens = tokenizer.tokenize(sample)
        self.assertEqual('[UNK]', tokens[0])

    @require_tokenizers
    def test_PreTrainedTokenizerFast_from_pretrained(self):
        if False:
            i = 10
            return i + 15
        tokenizer = AutoTokenizer.from_pretrained('robot-test/dummy-tokenizer-fast-with-model-config')
        self.assertEqual(type(tokenizer), PreTrainedTokenizerFast)
        self.assertEqual(tokenizer.model_max_length, 512)
        self.assertEqual(tokenizer.vocab_size, 30000)
        self.assertEqual(tokenizer.unk_token, '[UNK]')
        self.assertEqual(tokenizer.padding_side, 'right')
        self.assertEqual(tokenizer.truncation_side, 'right')

    def test_auto_tokenizer_from_local_folder(self):
        if False:
            i = 10
            return i + 15
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            tokenizer2 = AutoTokenizer.from_pretrained(tmp_dir)
        self.assertIsInstance(tokenizer2, tokenizer.__class__)
        self.assertEqual(tokenizer2.vocab_size, 12)

    def test_auto_tokenizer_fast_no_slow(self):
        if False:
            i = 10
            return i + 15
        tokenizer = AutoTokenizer.from_pretrained('ctrl')
        self.assertIsInstance(tokenizer, CTRLTokenizer)

    def test_get_tokenizer_config(self):
        if False:
            print('Hello World!')
        config = get_tokenizer_config('bert-base-cased')
        _ = config.pop('_commit_hash', None)
        self.assertEqual(config, {'do_lower_case': False})
        config = get_tokenizer_config(SMALL_MODEL_IDENTIFIER)
        self.assertDictEqual(config, {})
        tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            config = get_tokenizer_config(tmp_dir)
        self.assertEqual(config['tokenizer_class'], 'BertTokenizer')

    def test_new_tokenizer_registration(self):
        if False:
            return 10
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, slow_tokenizer_class=BertTokenizer)
            tokenizer = CustomTokenizer.from_pretrained(SMALL_MODEL_IDENTIFIER)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)
                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
                self.assertIsInstance(new_tokenizer, CustomTokenizer)
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    @require_tokenizers
    def test_new_tokenizer_fast_registration(self):
        if False:
            i = 10
            return i + 15
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, None))
            AutoTokenizer.register(CustomConfig, fast_tokenizer_class=CustomTokenizerFast)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, CustomTokenizerFast))
            del TOKENIZER_MAPPING._extra_content[CustomConfig]
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=CustomTokenizer, fast_tokenizer_class=CustomTokenizerFast)
            self.assertEqual(TOKENIZER_MAPPING[CustomConfig], (CustomTokenizer, CustomTokenizerFast))
            with self.assertRaises(ValueError):
                AutoTokenizer.register(BertConfig, fast_tokenizer_class=BertTokenizerFast)
            with tempfile.TemporaryDirectory() as tmp_dir:
                bert_tokenizer = BertTokenizerFast.from_pretrained(SMALL_MODEL_IDENTIFIER)
                bert_tokenizer.save_pretrained(tmp_dir)
                tokenizer = CustomTokenizerFast.from_pretrained(tmp_dir)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)
                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
                self.assertIsInstance(new_tokenizer, CustomTokenizerFast)
                new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, use_fast=False)
                self.assertIsInstance(new_tokenizer, CustomTokenizer)
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_tokenizer(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer')
        with self.assertRaises(ValueError):
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer', trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer', trust_remote_code=True)
        self.assertTrue(tokenizer.special_attribute_present)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertTrue(reloaded_tokenizer.special_attribute_present)
        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizerFast')
            self.assertEqual(reloaded_tokenizer.__class__.__name__, 'NewTokenizerFast')
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer', trust_remote_code=True, use_fast=False)
            self.assertTrue(tokenizer.special_attribute_present)
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizer')
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer.save_pretrained(tmp_dir)
                reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True, use_fast=False)
            self.assertEqual(reloaded_tokenizer.__class__.__name__, 'NewTokenizer')
            self.assertTrue(reloaded_tokenizer.special_attribute_present)
        else:
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizer')
            self.assertEqual(reloaded_tokenizer.__class__.__name__, 'NewTokenizer')

    @require_tokenizers
    def test_from_pretrained_dynamic_tokenizer_conflict(self):
        if False:
            i = 10
            return i + 15

        class NewTokenizer(BertTokenizer):
            special_attribute_present = False

        class NewTokenizerFast(BertTokenizerFast):
            slow_tokenizer_class = NewTokenizer
            special_attribute_present = False
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoTokenizer.register(CustomConfig, slow_tokenizer_class=NewTokenizer)
            AutoTokenizer.register(CustomConfig, fast_tokenizer_class=NewTokenizerFast)
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer')
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizerFast')
            self.assertFalse(tokenizer.special_attribute_present)
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer', use_fast=False)
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizer')
            self.assertFalse(tokenizer.special_attribute_present)
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer', trust_remote_code=False)
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizerFast')
            self.assertFalse(tokenizer.special_attribute_present)
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer', trust_remote_code=False, use_fast=False)
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizer')
            self.assertFalse(tokenizer.special_attribute_present)
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer', trust_remote_code=True)
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizerFast')
            self.assertTrue(tokenizer.special_attribute_present)
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer', trust_remote_code=True, use_fast=False)
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizer')
            self.assertTrue(tokenizer.special_attribute_present)
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in TOKENIZER_MAPPING._extra_content:
                del TOKENIZER_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_tokenizer_legacy_format(self):
        if False:
            while True:
                i = 10
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer_legacy', trust_remote_code=True)
        self.assertTrue(tokenizer.special_attribute_present)
        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizerFast')
            tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/test_dynamic_tokenizer_legacy', trust_remote_code=True, use_fast=False)
            self.assertTrue(tokenizer.special_attribute_present)
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizer')
        else:
            self.assertEqual(tokenizer.__class__.__name__, 'NewTokenizer')

    def test_repo_not_found(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(EnvironmentError, 'bert-base is not a local folder and is not a valid model identifier'):
            _ = AutoTokenizer.from_pretrained('bert-base')

    def test_revision_not_found(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(EnvironmentError, 'aaaaaa is not a valid git identifier \\(branch name, tag name or commit id\\)'):
            _ = AutoTokenizer.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision='aaaaaa')

    def test_cached_tokenizer_has_minimum_calls_to_head(self):
        if False:
            i = 10
            return i + 15
        _ = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-bert')
        with RequestCounter() as counter:
            _ = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-bert')
        self.assertEqual(counter['GET'], 0)
        self.assertEqual(counter['HEAD'], 1)
        self.assertEqual(counter.total_calls, 1)