import json
import sys
import tempfile
import unittest
from pathlib import Path
import transformers
from transformers import CONFIG_MAPPING, IMAGE_PROCESSOR_MAPPING, AutoConfig, AutoImageProcessor, CLIPConfig, CLIPImageProcessor
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from test_module.custom_configuration import CustomConfig
from test_module.custom_image_processing import CustomImageProcessor

class AutoImageProcessorTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_image_processor_from_model_shortcut(self):
        if False:
            while True:
                i = 10
        config = AutoImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_directory_from_key(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / 'preprocessor_config.json'
            config_tmpfile = Path(tmpdirname) / 'config.json'
            json.dump({'image_processor_type': 'CLIPImageProcessor', 'processor_class': 'CLIPProcessor'}, open(processor_tmpfile, 'w'))
            json.dump({'model_type': 'clip'}, open(config_tmpfile, 'w'))
            config = AutoImageProcessor.from_pretrained(tmpdirname)
            self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_directory_from_feature_extractor_key(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / 'preprocessor_config.json'
            config_tmpfile = Path(tmpdirname) / 'config.json'
            json.dump({'feature_extractor_type': 'CLIPFeatureExtractor', 'processor_class': 'CLIPProcessor'}, open(processor_tmpfile, 'w'))
            json.dump({'model_type': 'clip'}, open(config_tmpfile, 'w'))
            config = AutoImageProcessor.from_pretrained(tmpdirname)
            self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_directory_from_config(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = CLIPConfig()
            processor_tmpfile = Path(tmpdirname) / 'preprocessor_config.json'
            config_tmpfile = Path(tmpdirname) / 'config.json'
            json.dump({'image_processor_type': 'CLIPImageProcessor', 'processor_class': 'CLIPProcessor'}, open(processor_tmpfile, 'w'))
            json.dump({'model_type': 'clip'}, open(config_tmpfile, 'w'))
            config_dict = AutoImageProcessor.from_pretrained(tmpdirname).to_dict()
            config_dict.pop('image_processor_type')
            config = CLIPImageProcessor(**config_dict)
            model_config.save_pretrained(tmpdirname)
            config.save_pretrained(tmpdirname)
            config = AutoImageProcessor.from_pretrained(tmpdirname)
            dict_as_saved = json.loads(config.to_json_string())
            self.assertTrue('_processor_class' not in dict_as_saved)
        self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_file(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / 'preprocessor_config.json'
            json.dump({'image_processor_type': 'CLIPImageProcessor', 'processor_class': 'CLIPProcessor'}, open(processor_tmpfile, 'w'))
            config = AutoImageProcessor.from_pretrained(processor_tmpfile)
            self.assertIsInstance(config, CLIPImageProcessor)

    def test_repo_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(EnvironmentError, 'clip-base is not a local folder and is not a valid model identifier'):
            _ = AutoImageProcessor.from_pretrained('clip-base')

    def test_revision_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(EnvironmentError, 'aaaaaa is not a valid git identifier \\(branch name, tag name or commit id\\)'):
            _ = AutoImageProcessor.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision='aaaaaa')

    def test_image_processor_not_found(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(EnvironmentError, 'hf-internal-testing/config-no-model does not appear to have a file named preprocessor_config.json.'):
            _ = AutoImageProcessor.from_pretrained('hf-internal-testing/config-no-model')

    def test_from_pretrained_dynamic_image_processor(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            image_processor = AutoImageProcessor.from_pretrained('hf-internal-testing/test_dynamic_image_processor')
        with self.assertRaises(ValueError):
            image_processor = AutoImageProcessor.from_pretrained('hf-internal-testing/test_dynamic_image_processor', trust_remote_code=False)
        image_processor = AutoImageProcessor.from_pretrained('hf-internal-testing/test_dynamic_image_processor', trust_remote_code=True)
        self.assertEqual(image_processor.__class__.__name__, 'NewImageProcessor')
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_processor.save_pretrained(tmp_dir)
            reloaded_image_processor = AutoImageProcessor.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertEqual(reloaded_image_processor.__class__.__name__, 'NewImageProcessor')

    def test_new_image_processor_registration(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoImageProcessor.register(CustomConfig, CustomImageProcessor)
            with self.assertRaises(ValueError):
                AutoImageProcessor.register(CLIPConfig, CLIPImageProcessor)
            with tempfile.TemporaryDirectory() as tmpdirname:
                processor_tmpfile = Path(tmpdirname) / 'preprocessor_config.json'
                config_tmpfile = Path(tmpdirname) / 'config.json'
                json.dump({'feature_extractor_type': 'CLIPFeatureExtractor', 'processor_class': 'CLIPProcessor'}, open(processor_tmpfile, 'w'))
                json.dump({'model_type': 'clip'}, open(config_tmpfile, 'w'))
                image_processor = CustomImageProcessor.from_pretrained(tmpdirname)
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_processor.save_pretrained(tmp_dir)
                new_image_processor = AutoImageProcessor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_image_processor, CustomImageProcessor)
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in IMAGE_PROCESSOR_MAPPING._extra_content:
                del IMAGE_PROCESSOR_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_image_processor_conflict(self):
        if False:
            i = 10
            return i + 15

        class NewImageProcessor(CLIPImageProcessor):
            is_local = True
        try:
            AutoConfig.register('custom', CustomConfig)
            AutoImageProcessor.register(CustomConfig, NewImageProcessor)
            image_processor = AutoImageProcessor.from_pretrained('hf-internal-testing/test_dynamic_image_processor')
            self.assertEqual(image_processor.__class__.__name__, 'NewImageProcessor')
            self.assertTrue(image_processor.is_local)
            image_processor = AutoImageProcessor.from_pretrained('hf-internal-testing/test_dynamic_image_processor', trust_remote_code=False)
            self.assertEqual(image_processor.__class__.__name__, 'NewImageProcessor')
            self.assertTrue(image_processor.is_local)
            image_processor = AutoImageProcessor.from_pretrained('hf-internal-testing/test_dynamic_image_processor', trust_remote_code=True)
            self.assertEqual(image_processor.__class__.__name__, 'NewImageProcessor')
            self.assertTrue(not hasattr(image_processor, 'is_local'))
        finally:
            if 'custom' in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content['custom']
            if CustomConfig in IMAGE_PROCESSOR_MAPPING._extra_content:
                del IMAGE_PROCESSOR_MAPPING._extra_content[CustomConfig]