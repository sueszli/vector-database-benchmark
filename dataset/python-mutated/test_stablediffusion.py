from dataclasses import asdict, replace
from unittest import TestCase
import numpy as np
from diffusers import StableDiffusionPipeline
from parameterized import parameterized
from peft import LoHaConfig, LoraConfig, get_peft_model
from .testing_common import ClassInstantier, PeftCommonTester
from .testing_utils import temp_seed
PEFT_DIFFUSERS_SD_MODELS_TO_TEST = ['hf-internal-testing/tiny-stable-diffusion-torch']
CONFIG_TESTING_KWARGS = ({'text_encoder': {'r': 8, 'lora_alpha': 32, 'target_modules': ['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'], 'lora_dropout': 0.0, 'bias': 'none'}, 'unet': {'r': 8, 'lora_alpha': 32, 'target_modules': ['proj_in', 'proj_out', 'to_k', 'to_q', 'to_v', 'to_out.0', 'ff.net.0.proj', 'ff.net.2'], 'lora_dropout': 0.0, 'bias': 'none'}}, {'text_encoder': {'r': 8, 'alpha': 32, 'target_modules': ['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'], 'rank_dropout': 0.0, 'module_dropout': 0.0}, 'unet': {'r': 8, 'alpha': 32, 'target_modules': ['proj_in', 'proj_out', 'to_k', 'to_q', 'to_v', 'to_out.0', 'ff.net.0.proj', 'ff.net.2'], 'rank_dropout': 0.0, 'module_dropout': 0.0}})
CLASSES_MAPPING = {'lora': (LoraConfig, CONFIG_TESTING_KWARGS[0]), 'loha': (LoHaConfig, CONFIG_TESTING_KWARGS[1]), 'lokr': (LoHaConfig, CONFIG_TESTING_KWARGS[1])}
PeftStableDiffusionTestConfigManager = ClassInstantier(CLASSES_MAPPING)

class StableDiffusionModelTester(TestCase, PeftCommonTester):
    """
    Tests that diffusers StableDiffusion model works with PEFT as expected.

    """
    transformers_class = StableDiffusionPipeline

    def instantiate_sd_peft(self, model_id, config_cls, config_kwargs):
        if False:
            print('Hello World!')
        model = self.transformers_class.from_pretrained(model_id)
        config_kwargs = config_kwargs.copy()
        text_encoder_kwargs = config_kwargs.pop('text_encoder')
        unet_kwargs = config_kwargs.pop('unet')
        for (key, val) in config_kwargs.items():
            text_encoder_kwargs[key] = val
            unet_kwargs[key] = val
        config_text_encoder = config_cls(**text_encoder_kwargs)
        model.text_encoder = get_peft_model(model.text_encoder, config_text_encoder)
        config_unet = config_cls(**unet_kwargs)
        model.unet = get_peft_model(model.unet, config_unet)
        model = model.to(self.torch_device)
        return model

    def prepare_inputs_for_testing(self):
        if False:
            return 10
        return {'prompt': 'a high quality digital photo of a cute corgi', 'num_inference_steps': 20}

    @parameterized.expand(PeftStableDiffusionTestConfigManager.get_grid_parameters({'model_ids': PEFT_DIFFUSERS_SD_MODELS_TO_TEST, 'lora_kwargs': {'init_lora_weights': [False]}, 'loha_kwargs': {'init_weights': [False]}}))
    def test_merge_layers(self, test_name, model_id, config_cls, config_kwargs):
        if False:
            for i in range(10):
                print('nop')
        if config_cls == LoHaConfig:
            self.skipTest('LoHaConfig test is flaky')
        model = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)
        dummy_input = self.prepare_inputs_for_testing()
        with temp_seed(seed=42):
            peft_output = np.array(model(**dummy_input).images[0]).astype(np.float32)
        model.text_encoder = model.text_encoder.merge_and_unload()
        model.unet = model.unet.merge_and_unload()
        with temp_seed(seed=42):
            merged_output = np.array(model(**dummy_input).images[0]).astype(np.float32)
        self.assertTrue(np.allclose(peft_output, merged_output, atol=1.0))

    @parameterized.expand(PeftStableDiffusionTestConfigManager.get_grid_parameters({'model_ids': PEFT_DIFFUSERS_SD_MODELS_TO_TEST, 'lora_kwargs': {'init_lora_weights': [False]}}, filter_params_func=lambda tests: [x for x in tests if all((s not in x[0] for s in ['loha', 'lokr']))]))
    def test_add_weighted_adapter_base_unchanged(self, test_name, model_id, config_cls, config_kwargs):
        if False:
            for i in range(10):
                print('nop')
        model = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)
        text_encoder_adapter_name = next(iter(model.text_encoder.peft_config.keys()))
        unet_adapter_name = next(iter(model.unet.peft_config.keys()))
        text_encoder_adapter_config = replace(model.text_encoder.peft_config[text_encoder_adapter_name])
        unet_adapter_config = replace(model.unet.peft_config[unet_adapter_name])
        model.text_encoder.add_weighted_adapter([unet_adapter_name], [0.5], 'weighted_adapter_test')
        model.unet.add_weighted_adapter([unet_adapter_name], [0.5], 'weighted_adapter_test')
        self.assertTrue(asdict(text_encoder_adapter_config) == asdict(model.text_encoder.peft_config[text_encoder_adapter_name]))
        self.assertTrue(asdict(unet_adapter_config) == asdict(model.unet.peft_config[unet_adapter_name]))

    @parameterized.expand(PeftStableDiffusionTestConfigManager.get_grid_parameters({'model_ids': PEFT_DIFFUSERS_SD_MODELS_TO_TEST, 'lora_kwargs': {'init_lora_weights': [False]}, 'loha_kwargs': {'init_weights': [False]}, 'lokr_kwargs': {'init_weights': [False]}}))
    def test_disable_adapter(self, test_name, model_id, config_cls, config_kwargs):
        if False:
            return 10
        self._test_disable_adapter(model_id, config_cls, config_kwargs)