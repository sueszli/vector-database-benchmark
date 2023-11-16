import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from huggingface_hub import HfFolder, delete_repo
from requests.exceptions import HTTPError
from transformers import AutoConfig, BertConfig, GPT2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import TOKEN, USER, is_staging_test
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from test_module.custom_configuration import CustomConfig
config_common_kwargs = {'return_dict': False, 'output_hidden_states': True, 'output_attentions': True, 'torchscript': True, 'torch_dtype': 'float16', 'use_bfloat16': True, 'tf_legacy_loss': True, 'pruned_heads': {'a': 1}, 'tie_word_embeddings': False, 'is_decoder': True, 'cross_attention_hidden_size': 128, 'add_cross_attention': True, 'tie_encoder_decoder': True, 'max_length': 50, 'min_length': 3, 'do_sample': True, 'early_stopping': True, 'num_beams': 3, 'num_beam_groups': 3, 'diversity_penalty': 0.5, 'temperature': 2.0, 'top_k': 10, 'top_p': 0.7, 'typical_p': 0.2, 'repetition_penalty': 0.8, 'length_penalty': 0.8, 'no_repeat_ngram_size': 5, 'encoder_no_repeat_ngram_size': 5, 'bad_words_ids': [1, 2, 3], 'num_return_sequences': 3, 'chunk_size_feed_forward': 5, 'output_scores': True, 'return_dict_in_generate': True, 'forced_bos_token_id': 2, 'forced_eos_token_id': 3, 'remove_invalid_values': True, 'architectures': ['BertModel'], 'finetuning_task': 'translation', 'id2label': {0: 'label'}, 'label2id': {'label': '0'}, 'tokenizer_class': 'BertTokenizerFast', 'prefix': 'prefix', 'bos_token_id': 6, 'pad_token_id': 7, 'eos_token_id': 8, 'sep_token_id': 9, 'decoder_start_token_id': 10, 'exponential_decay_length_penalty': (5, 1.01), 'suppress_tokens': [0, 1], 'begin_suppress_tokens': 2, 'task_specific_params': {'translation': 'some_params'}, 'problem_type': 'regression'}

@is_staging_test
class ConfigPushToHubTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        try:
            delete_repo(token=cls._token, repo_id='test-config')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='valid_org/test-config-org')
        except HTTPError:
            pass
        try:
            delete_repo(token=cls._token, repo_id='test-dynamic-config')
        except HTTPError:
            pass

    def test_push_to_hub(self):
        if False:
            while True:
                i = 10
        config = BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37)
        config.push_to_hub('test-config', token=self._token)
        new_config = BertConfig.from_pretrained(f'{USER}/test-config')
        for (k, v) in config.to_dict().items():
            if k != 'transformers_version':
                self.assertEqual(v, getattr(new_config, k))
        delete_repo(token=self._token, repo_id='test-config')
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir, repo_id='test-config', push_to_hub=True, token=self._token)
        new_config = BertConfig.from_pretrained(f'{USER}/test-config')
        for (k, v) in config.to_dict().items():
            if k != 'transformers_version':
                self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization(self):
        if False:
            i = 10
            return i + 15
        config = BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37)
        config.push_to_hub('valid_org/test-config-org', token=self._token)
        new_config = BertConfig.from_pretrained('valid_org/test-config-org')
        for (k, v) in config.to_dict().items():
            if k != 'transformers_version':
                self.assertEqual(v, getattr(new_config, k))
        delete_repo(token=self._token, repo_id='valid_org/test-config-org')
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir, repo_id='valid_org/test-config-org', push_to_hub=True, token=self._token)
        new_config = BertConfig.from_pretrained('valid_org/test-config-org')
        for (k, v) in config.to_dict().items():
            if k != 'transformers_version':
                self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_dynamic_config(self):
        if False:
            while True:
                i = 10
        CustomConfig.register_for_auto_class()
        config = CustomConfig(attribute=42)
        config.push_to_hub('test-dynamic-config', token=self._token)
        self.assertDictEqual(config.auto_map, {'AutoConfig': 'custom_configuration.CustomConfig'})
        new_config = AutoConfig.from_pretrained(f'{USER}/test-dynamic-config', trust_remote_code=True)
        self.assertEqual(new_config.__class__.__name__, 'CustomConfig')
        self.assertEqual(new_config.attribute, 42)

class ConfigTestUtils(unittest.TestCase):

    def test_config_from_string(self):
        if False:
            while True:
                i = 10
        c = GPT2Config()
        n_embd = c.n_embd + 1
        resid_pdrop = c.resid_pdrop + 1.0
        scale_attn_weights = not c.scale_attn_weights
        summary_type = c.summary_type + 'foo'
        c.update_from_string(f'n_embd={n_embd},resid_pdrop={resid_pdrop},scale_attn_weights={scale_attn_weights},summary_type={summary_type}')
        self.assertEqual(n_embd, c.n_embd, 'mismatch for key: n_embd')
        self.assertEqual(resid_pdrop, c.resid_pdrop, 'mismatch for key: resid_pdrop')
        self.assertEqual(scale_attn_weights, c.scale_attn_weights, 'mismatch for key: scale_attn_weights')
        self.assertEqual(summary_type, c.summary_type, 'mismatch for key: summary_type')

    def test_config_common_kwargs_is_complete(self):
        if False:
            i = 10
            return i + 15
        base_config = PretrainedConfig()
        missing_keys = [key for key in base_config.__dict__ if key not in config_common_kwargs]
        self.assertListEqual(missing_keys, ['is_encoder_decoder', '_name_or_path', '_commit_hash', 'transformers_version'])
        keys_with_defaults = [key for (key, value) in config_common_kwargs.items() if value == getattr(base_config, key)]
        if len(keys_with_defaults) > 0:
            raise ValueError(f"The following keys are set with the default values in `test_configuration_common.config_common_kwargs` pick another value for them: {', '.join(keys_with_defaults)}.")

    def test_nested_config_load_from_dict(self):
        if False:
            return 10
        config = AutoConfig.from_pretrained('hf-internal-testing/tiny-random-CLIPModel', text_config={'num_hidden_layers': 2})
        self.assertNotIsInstance(config.text_config, dict)
        self.assertEqual(config.text_config.__class__.__name__, 'CLIPTextConfig')

    def test_from_pretrained_subfolder(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(OSError):
            _ = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert-subfolder')
        config = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert-subfolder', subfolder='bert')
        self.assertIsNotNone(config)

    def test_cached_files_are_used_when_internet_is_down(self):
        if False:
            for i in range(10):
                print('nop')
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        _ = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert')
        with mock.patch('requests.Session.request', return_value=response_mock) as mock_head:
            _ = BertConfig.from_pretrained('hf-internal-testing/tiny-random-bert')
            mock_head.assert_called()

    def test_legacy_load_from_url(self):
        if False:
            i = 10
            return i + 15
        _ = BertConfig.from_pretrained('https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/config.json')

    def test_local_versioning(self):
        if False:
            i = 10
            return i + 15
        configuration = AutoConfig.from_pretrained('bert-base-cased')
        configuration.configuration_files = ['config.4.0.0.json']
        with tempfile.TemporaryDirectory() as tmp_dir:
            configuration.save_pretrained(tmp_dir)
            configuration.hidden_size = 2
            json.dump(configuration.to_dict(), open(os.path.join(tmp_dir, 'config.4.0.0.json'), 'w'))
            new_configuration = AutoConfig.from_pretrained(tmp_dir)
            self.assertEqual(new_configuration.hidden_size, 2)
            configuration.configuration_files = ['config.42.0.0.json']
            configuration.hidden_size = 768
            configuration.save_pretrained(tmp_dir)
            shutil.move(os.path.join(tmp_dir, 'config.4.0.0.json'), os.path.join(tmp_dir, 'config.42.0.0.json'))
            new_configuration = AutoConfig.from_pretrained(tmp_dir)
            self.assertEqual(new_configuration.hidden_size, 768)

    def test_repo_versioning_before(self):
        if False:
            while True:
                i = 10
        repo = 'hf-internal-testing/test-two-configs'
        import transformers as new_transformers
        new_transformers.configuration_utils.__version__ = 'v4.0.0'
        (new_configuration, kwargs) = new_transformers.models.auto.AutoConfig.from_pretrained(repo, return_unused_kwargs=True)
        self.assertEqual(new_configuration.hidden_size, 2)
        self.assertDictEqual(kwargs, {})
        import transformers as old_transformers
        old_transformers.configuration_utils.__version__ = 'v3.0.0'
        old_configuration = old_transformers.models.auto.AutoConfig.from_pretrained(repo)
        self.assertEqual(old_configuration.hidden_size, 768)