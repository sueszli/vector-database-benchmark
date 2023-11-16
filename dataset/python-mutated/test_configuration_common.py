import copy
import json
import os
import tempfile
from transformers import is_torch_available
from .test_configuration_utils import config_common_kwargs

class ConfigTester(object):

    def __init__(self, parent, config_class=None, has_text_modality=True, common_properties=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.config_class = config_class
        self.has_text_modality = has_text_modality
        self.inputs_dict = kwargs
        self.common_properties = common_properties

    def create_and_test_config_common_properties(self):
        if False:
            while True:
                i = 10
        config = self.config_class(**self.inputs_dict)
        common_properties = ['hidden_size', 'num_attention_heads', 'num_hidden_layers'] if self.common_properties is None else self.common_properties
        if self.has_text_modality:
            common_properties.extend(['vocab_size'])
        for prop in common_properties:
            self.parent.assertTrue(hasattr(config, prop), msg=f'`{prop}` does not exist')
        for (idx, name) in enumerate(common_properties):
            try:
                setattr(config, name, idx)
                self.parent.assertEqual(getattr(config, name), idx, msg=f'`{name} value {idx} expected, but was {getattr(config, name)}')
            except NotImplementedError:
                pass
        for (idx, name) in enumerate(common_properties):
            try:
                config = self.config_class(**{name: idx})
                self.parent.assertEqual(getattr(config, name), idx, msg=f'`{name} value {idx} expected, but was {getattr(config, name)}')
            except NotImplementedError:
                pass

    def create_and_test_config_to_json_string(self):
        if False:
            while True:
                i = 10
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        for (key, value) in self.inputs_dict.items():
            self.parent.assertEqual(obj[key], value)

    def create_and_test_config_to_json_file(self):
        if False:
            for i in range(10):
                print('nop')
        config_first = self.config_class(**self.inputs_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, 'config.json')
            config_first.to_json_file(json_file_path)
            config_second = self.config_class.from_json_file(json_file_path)
        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained(self):
        if False:
            print('Hello World!')
        config_first = self.config_class(**self.inputs_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname)
        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())
        with self.parent.assertRaises(OSError):
            self.config_class.from_pretrained(f'.{tmpdirname}')

    def create_and_test_config_from_and_save_pretrained_subfolder(self):
        if False:
            while True:
                i = 10
        config_first = self.config_class(**self.inputs_dict)
        subfolder = 'test'
        with tempfile.TemporaryDirectory() as tmpdirname:
            sub_tmpdirname = os.path.join(tmpdirname, subfolder)
            config_first.save_pretrained(sub_tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname, subfolder=subfolder)
        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_with_num_labels(self):
        if False:
            print('Hello World!')
        config = self.config_class(**self.inputs_dict, num_labels=5)
        self.parent.assertEqual(len(config.id2label), 5)
        self.parent.assertEqual(len(config.label2id), 5)
        config.num_labels = 3
        self.parent.assertEqual(len(config.id2label), 3)
        self.parent.assertEqual(len(config.label2id), 3)

    def check_config_can_be_init_without_params(self):
        if False:
            for i in range(10):
                print('nop')
        if self.config_class.is_composition:
            with self.parent.assertRaises(ValueError):
                config = self.config_class()
        else:
            config = self.config_class()
            self.parent.assertIsNotNone(config)

    def check_config_arguments_init(self):
        if False:
            for i in range(10):
                print('nop')
        kwargs = copy.deepcopy(config_common_kwargs)
        config = self.config_class(**kwargs)
        wrong_values = []
        for (key, value) in config_common_kwargs.items():
            if key == 'torch_dtype':
                if not is_torch_available():
                    continue
                else:
                    import torch
                    if config.torch_dtype != torch.float16:
                        wrong_values.append(('torch_dtype', config.torch_dtype, torch.float16))
            elif getattr(config, key) != value:
                wrong_values.append((key, getattr(config, key), value))
        if len(wrong_values) > 0:
            errors = '\n'.join([f'- {v[0]}: got {v[1]} instead of {v[2]}' for v in wrong_values])
            raise ValueError(f'The following keys were not properly set in the config:\n{errors}')

    def run_common_tests(self):
        if False:
            while True:
                i = 10
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()
        self.create_and_test_config_from_and_save_pretrained()
        self.create_and_test_config_from_and_save_pretrained_subfolder()
        self.create_and_test_config_with_num_labels()
        self.check_config_can_be_init_without_params()
        self.check_config_arguments_init()