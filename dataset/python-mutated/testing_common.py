import json
import os
import pickle
import re
import tempfile
from collections import OrderedDict
from dataclasses import replace
import torch
import yaml
from diffusers import StableDiffusionPipeline
from peft import AdaLoraConfig, IA3Config, LoraConfig, PeftModel, PeftType, PrefixTuningConfig, PromptEncoderConfig, PromptLearningConfig, PromptTuningConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training
from peft.tuners.lora import LoraLayer
from peft.utils import _get_submodules, infer_device
from .testing_utils import get_state_dict
CONFIG_TESTING_KWARGS = ({'target_modules': None, 'feedforward_modules': None}, {'r': 8, 'lora_alpha': 32, 'target_modules': None, 'lora_dropout': 0.05, 'bias': 'none'}, {'num_virtual_tokens': 10}, {'num_virtual_tokens': 10, 'encoder_hidden_size': 32}, {'num_virtual_tokens': 10}, {'target_modules': None})
CLASSES_MAPPING = {'ia3': (IA3Config, CONFIG_TESTING_KWARGS[0]), 'lora': (LoraConfig, CONFIG_TESTING_KWARGS[1]), 'prefix_tuning': (PrefixTuningConfig, CONFIG_TESTING_KWARGS[2]), 'prompt_encoder': (PromptEncoderConfig, CONFIG_TESTING_KWARGS[3]), 'prompt_tuning': (PromptTuningConfig, CONFIG_TESTING_KWARGS[4]), 'adalora': (AdaLoraConfig, CONFIG_TESTING_KWARGS[5])}

class ClassInstantier(OrderedDict):

    def __getitem__(self, key, *args, **kwargs):
        if False:
            print('Hello World!')
        if any((kwarg in self[key][1] for kwarg in kwargs)):
            new_config_kwargs = self[key][1].copy()
            new_config_kwargs.update(kwargs)
            return (self[key][0], new_config_kwargs)
        return super().__getitem__(key, *args, **kwargs)

    def get_grid_parameters(self, grid_parameters, filter_params_func=None):
        if False:
            while True:
                i = 10
        '\n        Returns a list of all possible combinations of the parameters in the config classes.\n\n        Args:\n            grid_parameters (`dict`):\n                A dictionary containing the parameters to be tested. There should be at least the key "model_ids" which\n                contains a list of model ids to be tested. The other keys should be the name of the config class\n                post-fixed with "_kwargs" and the value should be a dictionary containing the parameters to be tested\n                for that config class.\n            filter_params_func (`callable`, `optional`):\n                A function that takes a list of tuples and returns a list of tuples. This function is used to filter\n                out the tests that needs for example to be skipped.\n\n        Returns:\n            generated_tests (`list`):\n                A list of tuples containing the name of the test, the model id, the config class and the config class\n                kwargs.\n        '
        generated_tests = []
        model_list = grid_parameters['model_ids']
        task_type = grid_parameters['task_type'] if 'task_type' in grid_parameters else None
        for model_id in model_list:
            for (key, value) in self.items():
                if '{}_kwargs'.format(key) in grid_parameters:
                    peft_configs = []
                    current_peft_config = value[1].copy()
                    for (current_key, current_value) in grid_parameters[f'{key}_kwargs'].items():
                        for kwarg in current_value:
                            current_peft_config.update({current_key: kwarg})
                            if task_type is not None:
                                current_peft_config.update({'task_type': task_type})
                            peft_configs.append(current_peft_config.copy())
                else:
                    current_peft_config = value[1].copy()
                    if task_type is not None:
                        current_peft_config.update({'task_type': task_type})
                    peft_configs = [current_peft_config]
                for peft_config in peft_configs:
                    generated_tests.append((f'test_{model_id}_{key}', model_id, value[0], peft_config))
        if filter_params_func is not None:
            generated_tests = filter_params_func(generated_tests)
        return generated_tests
PeftTestConfigManager = ClassInstantier(CLASSES_MAPPING)

class PeftCommonTester:
    """
    A large testing suite for testing common functionality of the PEFT models.

    Attributes:
        torch_device (`torch.device`):
            The device on which the tests will be run.
        transformers_class (`transformers.PreTrainedModel`):
            The transformers class that is being tested.
    """
    torch_device = infer_device()
    transformers_class = None

    def prepare_inputs_for_common(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def check_modelcard(self, tmp_dirname, model):
        if False:
            print('Hello World!')
        filename = os.path.join(tmp_dirname, 'README.md')
        self.assertTrue(os.path.exists(filename))
        with open(filename, 'r', encoding='utf-8') as f:
            readme = f.read()
        metainfo = re.search('---\\n(.*?)\\n---', readme, re.DOTALL).group(1)
        dct = yaml.safe_load(metainfo)
        self.assertEqual(dct['library_name'], 'peft')
        model_config = model.config if isinstance(model.config, dict) else model.config.to_dict()
        if model_config['model_type'] != 'custom':
            self.assertEqual(dct['base_model'], model_config['_name_or_path'])
        else:
            self.assertTrue('base_model' not in dct)

    def check_config_json(self, tmp_dirname, model):
        if False:
            return 10
        filename = os.path.join(tmp_dirname, 'adapter_config.json')
        self.assertTrue(os.path.exists(filename))
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model_config = model.config if isinstance(model.config, dict) else model.config.to_dict()
        if model_config['model_type'] != 'custom':
            self.assertEqual(config['base_model_name_or_path'], model_config['_name_or_path'])

    def _test_model_attr(self, model_id, config_cls, config_kwargs):
        if False:
            i = 10
            return i + 15
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        self.assertTrue(hasattr(model, 'save_pretrained'))
        self.assertTrue(hasattr(model, 'from_pretrained'))
        self.assertTrue(hasattr(model, 'push_to_hub'))

    def _test_adapter_name(self, model_id, config_cls, config_kwargs):
        if False:
            for i in range(10):
                print('nop')
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config, adapter_name='test-adapter')
        correctly_converted = False
        for (n, _) in model.named_parameters():
            if 'test-adapter' in n:
                correctly_converted = True
                break
        self.assertTrue(correctly_converted)

    def _test_prepare_for_training(self, model_id, config_cls, config_kwargs):
        if False:
            for i in range(10):
                print('nop')
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        dummy_input = self.prepare_inputs_for_testing()
        dummy_output = model.get_input_embeddings()(dummy_input['input_ids'])
        self.assertFalse(dummy_output.requires_grad)
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        model = prepare_model_for_int8_training(model)
        for param in model.parameters():
            self.assertFalse(param.requires_grad)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                if False:
                    print('Hello World!')
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        dummy_input = self.prepare_inputs_for_testing()
        dummy_output = model.get_input_embeddings()(dummy_input['input_ids'])
        self.assertTrue(dummy_output.requires_grad)

    def _test_save_pretrained(self, model_id, config_cls, config_kwargs, safe_serialization=True):
        if False:
            return 10
        if issubclass(config_cls, LoraConfig):
            config_kwargs = config_kwargs.copy()
            config_kwargs['init_lora_weights'] = False
        if issubclass(config_cls, IA3Config):
            config_kwargs = config_kwargs.copy()
            config_kwargs['init_ia3_weights'] = False
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            if safe_serialization:
                model.save_pretrained(tmp_dirname)
            else:
                model.save_pretrained(tmp_dirname, safe_serialization=False)
            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            if issubclass(config_cls, PromptEncoderConfig):
                state_dict = get_peft_model_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained, unwrap_compiled=True)
            else:
                state_dict = get_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_state_dict(model_from_pretrained, unwrap_compiled=True)
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)))
            target_adapter_filename = 'adapter_model.safetensors' if safe_serialization else 'adapter_model.bin'
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, target_adapter_filename)))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_config.json')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'model.safetensors')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'config.json')))
            self.check_modelcard(tmp_dirname, model)
            self.check_config_json(tmp_dirname, model)

    def _test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs, safe_serialization=True):
        if False:
            print('Hello World!')
        if issubclass(config_cls, AdaLoraConfig):
            return
        if issubclass(config_cls, LoraConfig):
            config_kwargs = config_kwargs.copy()
            config_kwargs['init_lora_weights'] = False
        if issubclass(config_cls, IA3Config):
            config_kwargs = config_kwargs.copy()
            config_kwargs['init_ia3_weights'] = False
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        new_adapter_config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model.add_adapter('new_adapter', new_adapter_config)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            if safe_serialization:
                model.save_pretrained(tmp_dirname)
            else:
                model.save_pretrained(tmp_dirname, safe_serialization=False)
            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            new_adapter_dir = os.path.join(tmp_dirname, 'new_adapter')
            model_from_pretrained.load_adapter(new_adapter_dir, 'new_adapter')
            if issubclass(config_cls, PromptEncoderConfig):
                state_dict = get_peft_model_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained, unwrap_compiled=True)
            else:
                state_dict = get_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_state_dict(model_from_pretrained, unwrap_compiled=True)
            self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)))
            target_adapter_filename = 'adapter_model.safetensors' if safe_serialization else 'adapter_model.bin'
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, target_adapter_filename)))
            self.assertTrue(os.path.exists(os.path.join(new_adapter_dir, target_adapter_filename)))
            self.assertTrue(os.path.exists(os.path.join(tmp_dirname, 'adapter_config.json')))
            self.assertTrue(os.path.exists(os.path.join(new_adapter_dir, 'adapter_config.json')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'model.safetensors')))
            self.assertFalse(os.path.exists(os.path.join(new_adapter_dir, 'model.safetensors')))
            self.assertFalse(os.path.exists(os.path.join(tmp_dirname, 'config.json')))
            self.assertFalse(os.path.exists(os.path.join(new_adapter_dir, 'config.json')))
            self.check_modelcard(tmp_dirname, model)
            self.check_config_json(tmp_dirname, model)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname, selected_adapters=['default'])
            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            self.assertTrue('default' in model_from_pretrained.peft_config.keys())
            self.assertTrue('new_adapter' not in model_from_pretrained.peft_config.keys())

    def _test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        if False:
            i = 10
            return i + 15
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname, is_trainable=False, config=config)
            self.assertTrue(model_from_pretrained.peft_config['default'].inference_mode)
            self.assertIs(model_from_pretrained.peft_config['default'], config)

    def _test_merge_layers_fp16(self, model_id, config_cls, config_kwargs):
        if False:
            return 10
        if config_cls not in (LoraConfig,):
            return
        if 'gpt2' in model_id.lower() and config_cls != LoraConfig:
            self.skipTest('Merging GPT2 adapters not supported for IA³ (yet)')
        model = self.transformers_class.from_pretrained(model_id, torch_dtype=torch.float16)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(device='cpu', dtype=torch.float16)
        model.eval()
        _ = model.merge_and_unload()

    def _test_merge_layers_nan(self, model_id, config_cls, config_kwargs):
        if False:
            while True:
                i = 10
        if config_cls not in (LoraConfig, IA3Config, AdaLoraConfig):
            return
        if 'gpt2' in model_id.lower() and config_cls != LoraConfig:
            self.skipTest('Merging GPT2 adapters not supported for IA³ (yet)')
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        dummy_input = self.prepare_inputs_for_testing()
        model.eval()
        logits_unmerged = model(**dummy_input)[0]
        model = model.merge_and_unload()
        logits_merged = model(**dummy_input)[0]
        self.assertTrue(torch.allclose(logits_unmerged, logits_merged, atol=0.001, rtol=0.001))
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        for (name, module) in model.named_parameters():
            if 'lora_A' in name or 'ia3' in name or 'lora_E' in name or ('lora_B' in name):
                module.data[0] = torch.nan
        with self.assertRaises(ValueError) as error_context:
            model = model.merge_and_unload(safe_merge=True)
        self.assertEqual(str(error_context.exception), 'NaNs detected in the merged weights. The adapter default seems to be broken')
        for (name, module) in model.named_parameters():
            if 'lora_A' in name or 'ia3' in name or 'lora_E' in name or ('lora_B' in name):
                module.data[0] = torch.inf
        with self.assertRaises(ValueError) as error_context:
            model = model.merge_and_unload(safe_merge=True)
        self.assertEqual(str(error_context.exception), 'NaNs detected in the merged weights. The adapter default seems to be broken')

    def _test_merge_layers(self, model_id, config_cls, config_kwargs):
        if False:
            while True:
                i = 10
        if config_cls not in (LoraConfig, IA3Config):
            return
        if 'gpt2' in model_id.lower() and config_cls != LoraConfig:
            self.skipTest('Merging GPT2 adapters not supported for IA³ (yet)')
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        if config.peft_type not in ('IA3', 'LORA'):
            with self.assertRaises(AttributeError):
                model = model.merge_and_unload()
        dummy_input = self.prepare_inputs_for_testing()
        model.eval()
        logits = model(**dummy_input)[0]
        model.merge_adapter()
        logits_merged = model(**dummy_input)[0]
        model.unmerge_adapter()
        logits_unmerged = model(**dummy_input)[0]
        model = model.merge_and_unload()
        logits_merged_unloaded = model(**dummy_input)[0]
        (atol, rtol) = (0.0001, 0.0001)
        if config.peft_type == 'IA3' and model_id == 'Conv2d':
            (atol, rtol) = (0.3, 0.01)
        self.assertTrue(torch.allclose(logits, logits_merged, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(logits, logits_unmerged, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(logits, logits_merged_unloaded, atol=atol, rtol=rtol))
        transformers_model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        logits_transformers = transformers_model(**dummy_input)[0]
        self.assertFalse(torch.allclose(logits_merged, logits_transformers, atol=1e-10, rtol=1e-10))
        if hasattr(model, 'save_pretrained'):
            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)
                model_from_pretrained = self.transformers_class.from_pretrained(tmp_dirname).to(self.torch_device)
        else:
            model_from_pretrained = pickle.loads(pickle.dumps(model))
        logits_merged_from_pretrained = model_from_pretrained(**dummy_input)[0]
        self.assertTrue(torch.allclose(logits_merged, logits_merged_from_pretrained, atol=atol, rtol=rtol))

    def _test_generate(self, model_id, config_cls, config_kwargs):
        if False:
            i = 10
            return i + 15
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        inputs = self.prepare_inputs_for_testing()
        _ = model.generate(**inputs)
        with self.assertRaises(TypeError):
            _ = model.generate(inputs['input_ids'])

    def _test_generate_half_prec(self, model_id, config_cls, config_kwargs):
        if False:
            print('Hello World!')
        if config_cls not in (IA3Config, LoraConfig, PrefixTuningConfig):
            return
        model = self.transformers_class.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        with self.assertRaises(TypeError):
            _ = model.generate(input_ids, attention_mask=attention_mask)

    def _test_prefix_tuning_half_prec_conversion(self, model_id, config_cls, config_kwargs):
        if False:
            return 10
        if config_cls not in (PrefixTuningConfig,):
            return
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)
        model = model.half()
        self.assertEqual(model.base_model_torch_dtype, torch.float16)

    def _test_training(self, model_id, config_cls, config_kwargs):
        if False:
            while True:
                i = 10
        if config_cls not in (IA3Config, LoraConfig):
            return
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        inputs = self.prepare_inputs_for_testing()
        output = model(**inputs)[0]
        loss = output.sum()
        loss.backward()
        parameter_prefix = 'ia3' if config_cls == IA3Config else 'lora'
        for (n, param) in model.named_parameters():
            if parameter_prefix in n or 'modules_to_save' in n:
                self.assertIsNotNone(param.grad)
            else:
                self.assertIsNone(param.grad)

    def _test_inference_safetensors(self, model_id, config_cls, config_kwargs):
        if False:
            i = 10
            return i + 15
        if config_cls not in (LoraConfig,):
            return
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        inputs = self.prepare_inputs_for_testing()
        output = model(**inputs)[0]
        logits = output[0]
        loss = output.sum()
        loss.backward()
        model.eval()
        logits = model(**inputs)[0][0]
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname, safe_serialization=True)
            self.assertTrue('adapter_model.safetensors' in os.listdir(tmp_dirname))
            self.assertTrue('adapter_model.bin' not in os.listdir(tmp_dirname))
            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname).to(self.torch_device)
            logits_from_pretrained = model_from_pretrained(**inputs)[0][0]
            self.assertTrue(torch.allclose(logits, logits_from_pretrained, atol=0.0001, rtol=0.0001))

    def _test_training_layer_indexing(self, model_id, config_cls, config_kwargs):
        if False:
            return 10
        if config_cls not in (LoraConfig,):
            return
        config = config_cls(base_model_name_or_path=model_id, layers_to_transform=[0], **config_kwargs)
        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        inputs = self.prepare_inputs_for_testing()
        output = model(**inputs)[0]
        logits = output[0]
        loss = output.sum()
        loss.backward()
        nb_trainable = 0
        for (n, param) in model.named_parameters():
            if 'lora' in n:
                self.assertIsNotNone(param.grad)
                nb_trainable += 1
            else:
                self.assertIsNone(param.grad)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname).to(self.torch_device)
            logits_from_pretrained = model_from_pretrained(**inputs)[0][0]
            self.assertTrue(torch.allclose(logits, logits_from_pretrained, atol=0.0001, rtol=0.0001))
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        nb_trainable_all = 0
        for (n, param) in model.named_parameters():
            if 'lora' in n:
                nb_trainable_all += 1
        self.assertLess(nb_trainable, nb_trainable_all)

    def _test_training_gradient_checkpointing(self, model_id, config_cls, config_kwargs):
        if False:
            return 10
        if config_cls not in (LoraConfig, IA3Config):
            return
        model = self.transformers_class.from_pretrained(model_id)
        if not getattr(model, 'supports_gradient_checkpointing', False):
            return
        model.gradient_checkpointing_enable()
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        inputs = self.prepare_inputs_for_testing()
        output = model(**inputs)[0]
        loss = output.sum()
        loss.backward()
        parameter_prefix = 'ia3' if config_cls == IA3Config else 'lora'
        for (n, param) in model.named_parameters():
            if parameter_prefix in n:
                self.assertIsNotNone(param.grad)
            else:
                self.assertIsNone(param.grad)

    def _test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        if False:
            i = 10
            return i + 15
        if config_cls not in (LoraConfig,):
            return
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            _ = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname, device_map={'': 'cpu'}).to(self.torch_device)

    def _test_training_prompt_learning_tasks(self, model_id, config_cls, config_kwargs):
        if False:
            while True:
                i = 10
        if not issubclass(config_cls, PromptLearningConfig):
            return
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        inputs = self.prepare_inputs_for_testing()
        output = model(**inputs)[0]
        loss = output.sum()
        loss.backward()
        for param in model.prompt_encoder.parameters():
            self.assertIsNotNone(param.grad)

    def _test_delete_adapter(self, model_id, config_cls, config_kwargs):
        if False:
            for i in range(10):
                print('nop')
        supported_peft_types = [PeftType.LORA, PeftType.LOHA, PeftType.LOKR]
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        if config.peft_type not in supported_peft_types:
            return
        model = self.transformers_class.from_pretrained(model_id)
        adapter_to_delete = 'delete_me'
        model = get_peft_model(model, config)
        model.add_adapter(adapter_to_delete, config)
        model.set_adapter(adapter_to_delete)
        model = model.to(self.torch_device)
        model.delete_adapter(adapter_to_delete)
        self.assertFalse(adapter_to_delete in model.peft_config)
        self.assertEqual(model.active_adapters, ['default'])
        key_list = [key for (key, _) in model.named_modules() if 'lora' not in key]
        for key in key_list:
            (_, target, _) = _get_submodules(model, key)
            attributes_to_check = getattr(target, 'adapter_layer_names', []) + getattr(target, 'other_param_names', [])
            for attr in attributes_to_check:
                self.assertFalse(adapter_to_delete in getattr(target, attr))
        model.delete_adapter('default')
        self.assertFalse('default' in model.peft_config)
        self.assertEqual(model.active_adapters, [])
        input = self.prepare_inputs_for_testing()
        model.base_model(**input)

    def _test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        if False:
            i = 10
            return i + 15
        supported_peft_types = [PeftType.LORA, PeftType.LOHA, PeftType.LOKR]
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        if config.peft_type not in supported_peft_types:
            return
        model = self.transformers_class.from_pretrained(model_id)
        adapter_to_delete = 'delete_me'
        model = get_peft_model(model, config)
        model.add_adapter(adapter_to_delete, config)
        model = model.to(self.torch_device)
        model.delete_adapter(adapter_to_delete)
        self.assertFalse(adapter_to_delete in model.peft_config)
        self.assertEqual(model.active_adapters, ['default'])
        key_list = [key for (key, _) in model.named_modules() if 'lora' not in key]
        for key in key_list:
            (_, target, _) = _get_submodules(model, key)
            attributes_to_check = getattr(target, 'adapter_layer_names', []) + getattr(target, 'other_param_names', [])
            for attr in attributes_to_check:
                self.assertFalse(adapter_to_delete in getattr(target, attr))
        model.delete_adapter('default')
        self.assertFalse('default' in model.peft_config)
        self.assertEqual(model.active_adapters, [])
        input = self.prepare_inputs_for_testing()
        model.base_model(**input)

    def _test_unload_adapter(self, model_id, config_cls, config_kwargs):
        if False:
            return 10
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        if config.peft_type not in ('LORA', 'ADALORA'):
            with self.assertRaises(AttributeError):
                model = model.unload()
        else:
            dummy_input = self.prepare_inputs_for_testing()
            logits_with_lora = model(**dummy_input)[0]
            transformers_model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            logits_transformers = transformers_model(**dummy_input)[0]
            model.eval()
            model = model.unload()
            logits_unload = model(**dummy_input)[0]
            self.assertFalse(torch.allclose(logits_with_lora, logits_unload, atol=1e-10, rtol=1e-10))
            self.assertTrue(torch.allclose(logits_transformers, logits_unload, atol=0.0001, rtol=0.0001))

    def _test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        if False:
            while True:
                i = 10
        if issubclass(config_cls, AdaLoraConfig):
            return
        adapter_list = ['adapter1', 'adapter_2', 'adapter_3']
        weight_list = [0.5, 1.5, 1.5]
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        if not isinstance(config, LoraConfig):
            return
        model = get_peft_model(model, config, adapter_list[0])
        model.add_adapter(adapter_list[1], config)
        model.add_adapter(adapter_list[2], replace(config, r=20))
        model = model.to(self.torch_device)
        model.add_weighted_adapter([adapter_list[0]], [weight_list[0]], 'single_adapter_reweighting')
        model.add_weighted_adapter(adapter_list[1:], weight_list[1:], 'multi_adapter_svd_reweighting')
        model.add_weighted_adapter(adapter_list[1:], weight_list[1:], 'multi_adapter_cat_reweighting', combination_type='cat')
        model.add_weighted_adapter(adapter_list[:2], weight_list[:2], 'multi_adapter_linear_reweighting', combination_type='linear')
        with self.assertRaises(ValueError):
            model.add_weighted_adapter(adapter_list[1:], weight_list[1:], 'multi_adapter_linear_reweighting_uneven_r', combination_type='linear')
        new_adapters = ['single_adapter_reweighting', 'multi_adapter_svd_reweighting', 'multi_adapter_cat_reweighting', 'multi_adapter_linear_reweighting']
        for new_adapter in new_adapters:
            self.assertTrue(new_adapter in model.peft_config)
        key_list = [key for (key, _) in model.named_modules() if 'lora' not in key]
        for key in key_list:
            (_, target, _) = _get_submodules(model, key)
            if isinstance(target, LoraLayer):
                for adapter_name in new_adapters:
                    if 'single' in adapter_name:
                        new_delta_weight = target.get_delta_weight(adapter_name)
                        weighted_original_delta_weights = target.get_delta_weight(adapter_list[0]) * weight_list[0]
                        self.assertTrue(torch.allclose(new_delta_weight, weighted_original_delta_weights, atol=0.0001, rtol=0.0001))
                    elif 'svd' in adapter_name:
                        self.assertTrue(target.r[adapter_name] == 20)
                    elif 'linear' in adapter_name:
                        self.assertTrue(target.r[adapter_name] == 8)
                    elif 'cat' in adapter_name:
                        self.assertTrue(target.r[adapter_name] == 28)
        dummy_input = self.prepare_inputs_for_testing()
        model.eval()
        for adapter_name in new_adapters:
            model.set_adapter(adapter_name)
            self.assertTrue(model.active_adapter == adapter_name)
            self.assertTrue(model.active_adapters == [adapter_name])
            model(**dummy_input)[0]

    def _test_disable_adapter(self, model_id, config_cls, config_kwargs):
        if False:
            return 10
        task_type = config_kwargs.get('task_type')
        if task_type == 'SEQ_2_SEQ_LM' and config_cls in (PromptTuningConfig, PromptEncoderConfig):
            self.skipTest('Seq2Seq + prompt tuning/prompt encoder does not work with disabling adapters')

        def get_output(model):
            if False:
                while True:
                    i = 10
            torch.manual_seed(0)
            if hasattr(model, 'generate'):
                output = model.generate(**input, return_dict_in_generate=True, output_scores=True).scores[0]
            else:
                output = model(**input)
            if hasattr(output, 'images'):
                import numpy as np
                img = output.images[0]
                return torch.from_numpy(np.array(img))
            return output
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        input = self.prepare_inputs_for_testing()
        output_before = get_output(model)
        if hasattr(self, 'instantiate_sd_peft'):
            peft_model = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)
        else:
            config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
            peft_model = get_peft_model(model, config)
        output_peft = get_output(peft_model)
        if isinstance(peft_model, StableDiffusionPipeline):
            self.assertTrue((output_before != output_peft).float().mean() > 0.9)
        else:
            self.assertFalse(torch.allclose(output_before, output_peft))
        if isinstance(peft_model, StableDiffusionPipeline):
            with peft_model.unet.disable_adapter():
                with peft_model.text_encoder.disable_adapter():
                    output_peft_disabled = get_output(peft_model)
            self.assertTrue((output_before != output_peft_disabled).float().mean() < 0.0001)
        else:
            with peft_model.disable_adapter():
                output_peft_disabled = get_output(peft_model)
            self.assertTrue(torch.allclose(output_before, output_peft_disabled, atol=1e-06, rtol=1e-06))

    def _test_adding_multiple_adapters_with_bias_raises(self, model_id, config_cls, config_kwargs):
        if False:
            return 10
        if not issubclass(config_cls, (LoraConfig, AdaLoraConfig)):
            return
        config_kwargs = config_kwargs.copy()
        config_kwargs['bias'] = 'all'
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config, 'adapter0')
        with self.assertRaises(ValueError):
            model.add_adapter('adapter1', replace(config, r=20))
        self.assertFalse('adapter1' in model.peft_config)
        self.assertFalse('adapter1' in model.base_model.peft_config)

    def _test_passing_input_embeds_works(self, test_name, model_id, config_cls, config_kwargs):
        if False:
            return 10
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config, adapter_name='test-adapter').to(self.torch_device)
        dummy_input = self.prepare_inputs_for_testing()
        inputs_embeds = model.get_input_embeddings()(dummy_input['input_ids'])
        model.forward(inputs_embeds=inputs_embeds)