import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import transformers
import transformers.models.auto
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, get_tests_dir
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from test_module.custom_configuration import CustomConfig
SAMPLE_ROBERTA_CONFIG = get_tests_dir('fixtures/dummy-config.json')

class AutoConfigTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_module_spec(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNotNone(transformers.models.auto.__spec__)
        self.assertIsNotNone(importlib.util.find_spec('transformers.models.auto'))

    def test_config_from_model_shortcut(self):
        if False:
            print('Hello World!')
        config = AutoConfig.from_pretrained('bert-base-uncased')
        self.assertIsInstance(config, BertConfig)

    def test_config_model_type_from_local_file(self):
        if False:
            while True:
                i = 10
        config = AutoConfig.from_pretrained(SAMPLE_ROBERTA_CONFIG)
        self.assertIsInstance(config, RobertaConfig)

    def test_config_model_type_from_model_identifier(self):
        if False:
            print('Hello World!')
        config = AutoConfig.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(config, RobertaConfig)

    def test_config_for_model_str(self):
        if False:
            print('Hello World!')
        config = AutoConfig.for_model('roberta')
        self.assertIsInstance(config, RobertaConfig)

    def test_pattern_matching_fallback(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = os.path.join(tmp_dir, 'fake-roberta')
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, 'config.json'), 'w') as f:
                f.write(json.dumps({}))
            config = AutoConfig.from_pretrained(folder)
            self.assertEqual(type(config), RobertaConfig)

    def test_new_config_registration(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            AutoConfig.register('custom', CustomConfig)
            with self.assertRaises(ValueError):
                AutoConfig.register('model', CustomConfig)
            with self.assertRaises(ValueError):
                AutoConfig.register('bert', BertConfig)
            config = CustomConfig()
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir)
                new_config = AutoConfig.from_pretrained(tmp_dir)
                self.assertIsInstance(new_config, CustomConfig)
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']

    def test_repo_not_found(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(EnvironmentError, 'bert-base is not a local folder and is not a valid model identifier'):
            _ = AutoConfig.from_pretrained('bert-base')

    def test_revision_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(EnvironmentError, 'aaaaaa is not a valid git identifier \\(branch name, tag name or commit id\\)'):
            _ = AutoConfig.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision='aaaaaa')

    def test_configuration_not_found(self):
        if False:
            return 10
        with self.assertRaisesRegex(EnvironmentError, 'hf-internal-testing/no-config-test-repo does not appear to have a file named config.json.'):
            _ = AutoConfig.from_pretrained('hf-internal-testing/no-config-test-repo')

    def test_from_pretrained_dynamic_config(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            config = AutoConfig.from_pretrained('hf-internal-testing/test_dynamic_model')
        with self.assertRaises(ValueError):
            config = AutoConfig.from_pretrained('hf-internal-testing/test_dynamic_model', trust_remote_code=False)
        config = AutoConfig.from_pretrained('hf-internal-testing/test_dynamic_model', trust_remote_code=True)
        self.assertEqual(config.__class__.__name__, 'NewModelConfig')
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            reloaded_config = AutoConfig.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertEqual(reloaded_config.__class__.__name__, 'NewModelConfig')

    def test_from_pretrained_dynamic_config_conflict(self):
        if False:
            return 10

        class NewModelConfigLocal(BertConfig):
            model_type = 'new-model'
        try:
            AutoConfig.register('new-model', NewModelConfigLocal)
            config = AutoConfig.from_pretrained('hf-internal-testing/test_dynamic_model')
            self.assertEqual(config.__class__.__name__, 'NewModelConfigLocal')
            config = AutoConfig.from_pretrained('hf-internal-testing/test_dynamic_model', trust_remote_code=False)
            self.assertEqual(config.__class__.__name__, 'NewModelConfigLocal')
            config = AutoConfig.from_pretrained('hf-internal-testing/test_dynamic_model', trust_remote_code=True)
            self.assertEqual(config.__class__.__name__, 'NewModelConfig')
        finally:
            if 'new-model' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['new-model']