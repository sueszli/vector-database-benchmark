import tempfile
import unittest
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from transformers.testing_utils import is_torch_available, require_accelerate, require_auto_gptq, require_optimum, require_torch_gpu, require_torch_multi_gpu, slow
if is_torch_available():
    import torch

class GPTQConfigTest(unittest.TestCase):

    def test_bits(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            GPTQConfig(bits='')
            GPTQConfig(bits=1)
        GPTQConfig(bits=2)
        GPTQConfig(bits=4)

    def test_dataset(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            GPTQConfig(bits=2, dataset='auto_gpt')
        GPTQConfig(bits=2, dataset='c4')
        GPTQConfig(bits=2, dataset='ptb-new')

    def test_damp_percent(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            GPTQConfig(bits=2, damp_percent=10)
            GPTQConfig(bits=2, damp_percent=-1)
            GPTQConfig(bits=2, damp_percent='0')
        GPTQConfig(bits=2, damp_percent=0.01)

    def test_to_dict(self):
        if False:
            i = 10
            return i + 15
        quantization_config = GPTQConfig(bits=2)
        quantization_config.to_dict()

    def test_from_dict(self):
        if False:
            print('Hello World!')
        dict = {'bits': 2}
        quantization_config = GPTQConfig.from_dict(dict)
        self.assertEqual(dict['bits'], quantization_config.bits)

    @require_optimum
    def test_optimum_config(self):
        if False:
            return 10
        from optimum.gptq import GPTQQuantizer
        config = GPTQConfig(bits=2)
        optimum_config = GPTQQuantizer.from_dict(config.to_dict_optimum())
        self.assertEqual(optimum_config.bits, config.bits)
        new_config = GPTQConfig.from_dict_optimum(optimum_config.to_dict())
        self.assertEqual(optimum_config.bits, new_config.bits)

@slow
@require_optimum
@require_auto_gptq
@require_torch_gpu
class GPTQTest(unittest.TestCase):
    model_name = 'bigscience/bloom-560m'
    input_text = 'Hello my name is'
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add('Hello my name is John and I am a professional photographer. I')
    EXPECTED_OUTPUTS.add('Hello my name is John, I am a professional photographer and I')
    EXPECTED_OUTPUTS.add('Hello my name is John, I am a student in the University of')
    EXPECTED_OUTPUTS.add('Hello my name is John and I am a very good looking man.')
    EXPECTED_OUTPUTS.add('Hello my name is Alyson, I am a student in the')
    EXPECTED_OUTPUTS.add('Hello my name is Alyson and I am a very sweet,')
    EXPECTED_RELATIVE_DIFFERENCE = 1.664253062
    bits = 4
    group_size = 128
    desc_act = False
    use_exllama = False
    dataset = ['auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.']
    device_map = None

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        '\n        Setup quantized model\n        '
        cls.model_fp16 = AutoModelForCausalLM.from_pretrained(cls.model_name, torch_dtype=torch.float16, device_map=cls.device_map)
        cls.mem_fp16 = cls.model_fp16.get_memory_footprint()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)
        quantization_config = GPTQConfig(bits=cls.bits, dataset=cls.dataset, tokenizer=cls.tokenizer, group_size=cls.group_size, desc_act=cls.desc_act, use_exllama=cls.use_exllama)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(cls.model_name, torch_dtype=torch.float16, device_map=cls.device_map, quantization_config=quantization_config)

    def test_memory_footprint(self):
        if False:
            print('Hello World!')
        '\n        A simple test to check if the model conversion has been done correctly by checking on the\n        memory footprint of the converted model\n        '
        mem_quantized = self.quantized_model.get_memory_footprint()
        self.assertAlmostEqual(self.mem_fp16 / mem_quantized, self.EXPECTED_RELATIVE_DIFFERENCE)

    def test_device_and_dtype_assignment(self):
        if False:
            while True:
                i = 10
        '\n        Test whether trying to cast (or assigning a device to) a model after quantization will throw an error.\n        Checks also if other models are casted correctly.\n        '
        if self.device_map is None:
            _ = self.quantized_model.to(0)
        with self.assertRaises(ValueError):
            self.quantized_model.to(torch.float16)

    def test_original_dtype(self):
        if False:
            print('Hello World!')
        '\n        A simple test to check if the model succesfully stores the original dtype\n        '
        self.assertTrue(hasattr(self.quantized_model.config, '_pre_quantization_dtype'))
        self.assertFalse(hasattr(self.model_fp16.config, '_pre_quantization_dtype'))
        self.assertTrue(self.quantized_model.config._pre_quantization_dtype == torch.float16)

    def test_quantized_layers_class(self):
        if False:
            i = 10
            return i + 15
        '\n        Simple test to check if the model conversion has been done correctly by checking on\n        the class type of the linear layers of the converted models\n        '
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
        QuantLinear = dynamically_import_QuantLinear(use_triton=False, desc_act=self.desc_act, group_size=self.group_size, bits=self.bits, disable_exllama=not self.use_exllama, disable_exllamav2=True)
        self.assertTrue(self.quantized_model.transformer.h[0].mlp.dense_4h_to_h.__class__ == QuantLinear)

    def check_inference_correctness(self, model):
        if False:
            i = 10
            return i + 15
        "\n        Test the generation quality of the quantized model and see that we are matching the expected output.\n        Given that we are operating on small numbers + the testing model is relatively small, we might not get\n        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.\n        "
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = model.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def check_quantized_layers_type(self, model, value):
        if False:
            while True:
                i = 10
        self.assertTrue(model.transformer.h[0].mlp.dense_4h_to_h.QUANT_TYPE == value)

    def test_generate_quality(self):
        if False:
            while True:
                i = 10
        '\n        Simple test to check the quality of the model by comparing the generated tokens with the expected tokens\n        '
        if self.device_map is None:
            self.check_inference_correctness(self.quantized_model.to(0))
        else:
            self.check_inference_correctness(self.quantized_model)

    def test_serialization(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the serialization of the model and the loading of the quantized weights works\n        '
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            if not self.use_exllama:
                quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname).to(0)
                self.check_quantized_layers_type(quantized_model_from_saved, 'cuda-old')
            else:
                quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={'': 0})
                self.check_quantized_layers_type(quantized_model_from_saved, 'exllama')
            self.check_inference_correctness(quantized_model_from_saved)

    @require_accelerate
    def test_serialization_big_model_inference(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the serialization of the model and the loading of the quantized weights with big model inference\n        '
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map='auto')
            self.check_inference_correctness(quantized_model_from_saved)

    def test_change_loading_attributes(self):
        if False:
            print('Hello World!')
        '\n        Test the serialization of the model and the loading of the quantized weights works with another config file\n        '
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            if not self.use_exllama:
                self.assertEqual(self.quantized_model.config.quantization_config.use_exllama, False)
                quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, quantization_config=GPTQConfig(use_exllama=True, bits=4), device_map={'': 0})
                self.assertEqual(quantized_model_from_saved.config.quantization_config.use_exllama, True)
                self.assertEqual(quantized_model_from_saved.config.quantization_config.bits, self.bits)
                self.check_quantized_layers_type(quantized_model_from_saved, 'exllama')
                self.check_inference_correctness(quantized_model_from_saved)

@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMap(GPTQTest):
    device_map = 'auto'

@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMapExllama(GPTQTest):
    device_map = 'auto'
    use_exllama = True

@slow
@require_optimum
@require_auto_gptq
@require_torch_gpu
@require_accelerate
class GPTQTestActOrderExllama(unittest.TestCase):
    """
    Test GPTQ model with exllama kernel and desc_act=True (also known as act-order).
    More information on those arguments here:
    https://huggingface.co/docs/transformers/main_classes/quantization#transformers.GPTQConfig
    """
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add('Hello my name is Katie and I am a 20 year')
    model_name = 'hf-internal-testing/Llama-2-7B-GPTQ'
    revision = 'gptq-4bit-128g-actorder_True'
    input_text = 'Hello my name is'

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setup quantized model\n        '
        cls.quantization_config = GPTQConfig(bits=4, max_input_length=4028)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(cls.model_name, revision=cls.revision, torch_dtype=torch.float16, device_map={'': 0}, quantization_config=cls.quantization_config)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

    def check_inference_correctness(self, model):
        if False:
            while True:
                i = 10
        "\n        Test the generation quality of the quantized model and see that we are matching the expected output.\n        Given that we are operating on small numbers + the testing model is relatively small, we might not get\n        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.\n        "
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = model.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_quantized_layers_type(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.quantized_model.model.layers[0].self_attn.k_proj.QUANT_TYPE == 'exllama')

    def test_generate_quality(self):
        if False:
            return 10
        '\n        Simple test to check the quality of the model by comparing the generated tokens with the expected tokens\n        '
        self.check_inference_correctness(self.quantized_model)

    def test_max_input_length(self):
        if False:
            return 10
        '\n        Test if the max_input_length works. It modifies the maximum input length that of the model that runs with exllama backend.\n        '
        prompt = 'I am in Paris and' * 1000
        inp = self.tokenizer(prompt, return_tensors='pt').to(0)
        self.assertTrue(inp['input_ids'].shape[1] > 4028)
        with self.assertRaises(RuntimeError) as cm:
            self.quantized_model.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)
            self.assertTrue('temp_state buffer is too small' in str(cm.exception))
        prompt = 'I am in Paris and' * 500
        inp = self.tokenizer(prompt, return_tensors='pt').to(0)
        self.assertTrue(inp['input_ids'].shape[1] < 4028)
        self.quantized_model.generate(**inp, num_beams=1, min_new_tokens=3, max_new_tokens=3)

@slow
@require_optimum
@require_auto_gptq
@require_torch_gpu
@require_accelerate
class GPTQTestExllamaV2(unittest.TestCase):
    """
    Test GPTQ model with exllamav2 kernel and desc_act=True (also known as act-order).
    More information on those arguments here:
    https://huggingface.co/docs/transformers/main_classes/quantization#transformers.GPTQConfig
    """
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add('Hello my name is Katie and I am a 20 year')
    model_name = 'hf-internal-testing/Llama-2-7B-GPTQ'
    revision = 'gptq-4bit-128g-actorder_True'
    input_text = 'Hello my name is'

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setup quantized model\n        '
        cls.quantization_config = GPTQConfig(bits=4, exllama_config={'version': 2})
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(cls.model_name, revision=cls.revision, torch_dtype=torch.float16, device_map={'': 0}, quantization_config=cls.quantization_config)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

    def test_quantized_layers_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.quantized_model.model.layers[0].self_attn.k_proj.QUANT_TYPE == 'exllamav2')

    def check_inference_correctness(self, model):
        if False:
            return 10
        "\n        Test the generation quality of the quantized model and see that we are matching the expected output.\n        Given that we are operating on small numbers + the testing model is relatively small, we might not get\n        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.\n        "
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = model.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality(self):
        if False:
            return 10
        '\n        Simple test to check the quality of the model by comapring the the generated tokens with the expected tokens\n        '
        self.check_inference_correctness(self.quantized_model)

@pytest.mark.skip
@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMapCPUOffload(GPTQTest):
    device_map = {'transformer.word_embeddings': 0, 'transformer.word_embeddings_layernorm': 0, 'lm_head': 0, 'transformer.h.0': 0, 'transformer.h.1': 0, 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0, 'transformer.h.5': 0, 'transformer.h.6': 0, 'transformer.h.7': 0, 'transformer.h.8': 0, 'transformer.h.9': 0, 'transformer.h.10': 1, 'transformer.h.11': 1, 'transformer.h.12': 1, 'transformer.h.13': 1, 'transformer.h.14': 1, 'transformer.h.15': 1, 'transformer.h.16': 1, 'transformer.h.17': 0, 'transformer.h.18': 'cpu', 'transformer.h.19': 'cpu', 'transformer.h.20': 'cpu', 'transformer.h.21': 'cpu', 'transformer.h.22': 'cpu', 'transformer.h.23': 1, 'transformer.ln_f': 0}