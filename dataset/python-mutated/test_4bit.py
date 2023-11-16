import gc
import importlib.metadata
import tempfile
import unittest
from packaging import version
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers.testing_utils import is_torch_available, require_accelerate, require_bitsandbytes, require_torch, require_torch_gpu, require_torch_multi_gpu, slow

def get_some_linear_layer(model):
    if False:
        for i in range(10):
            print('nop')
    if model.config.model_type == 'gpt2':
        return model.transformer.h[0].mlp.c_fc
    return model.transformer.h[0].mlp.dense_4h_to_h
if is_torch_available():
    import torch
    import torch.nn as nn

    class LoRALayer(nn.Module):
        """Wraps a linear layer with LoRA-like adapter - Used for testing purposes only"""

        def __init__(self, module: nn.Module, rank: int):
            if False:
                return 10
            super().__init__()
            self.module = module
            self.adapter = nn.Sequential(nn.Linear(module.in_features, rank, bias=False), nn.Linear(rank, module.out_features, bias=False))
            small_std = (2.0 / (5 * min(module.in_features, module.out_features))) ** 0.5
            nn.init.normal_(self.adapter[0].weight, std=small_std)
            nn.init.zeros_(self.adapter[1].weight)
            self.adapter.to(module.weight.device)

        def forward(self, input, *args, **kwargs):
            if False:
                print('Hello World!')
            return self.module(input, *args, **kwargs) + self.adapter(input)

@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class Base4bitTest(unittest.TestCase):
    model_name = 'bigscience/bloom-1b7'
    EXPECTED_RELATIVE_DIFFERENCE = 2.109659552692574
    input_text = 'Hello my name is'
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add('Hello my name is John and I am a professional photographer. I')
    EXPECTED_OUTPUTS.add('Hello my name is John.\nI am a friend of your father.\n')
    EXPECTED_OUTPUTS.add('Hello my name is John Doe, I am a student at the University')
    MAX_NEW_TOKENS = 10

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

class Bnb4BitTest(Base4bitTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map='auto')
        self.model_4bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True, device_map='auto')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to\n        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27\n        '
        del self.model_fp16
        del self.model_4bit
        gc.collect()
        torch.cuda.empty_cache()

    def test_quantization_num_parameters(self):
        if False:
            print('Hello World!')
        '\n        Test if the number of returned parameters is correct\n\n        See: https://github.com/huggingface/transformers/issues/25978\n        '
        num_params_4bit = self.model_4bit.num_parameters()
        num_params_fp16 = self.model_fp16.num_parameters()
        self.assertEqual(num_params_4bit, num_params_fp16)

    def test_quantization_config_json_serialization(self):
        if False:
            return 10
        '\n        A simple test to check if the quantization config is correctly serialized and deserialized\n        '
        config = self.model_4bit.config
        self.assertTrue(hasattr(config, 'quantization_config'))
        _ = config.to_dict()
        _ = config.to_diff_dict()
        _ = config.to_json_string()

    def test_memory_footprint(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A simple test to check if the model conversion has been done correctly by checking on the\n        memory footprint of the converted model and the class type of the linear layers of the converted models\n        '
        from bitsandbytes.nn import Params4bit
        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_4bit = self.model_4bit.get_memory_footprint()
        self.assertAlmostEqual(mem_fp16 / mem_4bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        linear = get_some_linear_layer(self.model_4bit)
        self.assertTrue(linear.weight.__class__ == Params4bit)

    def test_original_dtype(self):
        if False:
            while True:
                i = 10
        '\n        A simple test to check if the model succesfully stores the original dtype\n        '
        self.assertTrue(hasattr(self.model_4bit.config, '_pre_quantization_dtype'))
        self.assertFalse(hasattr(self.model_fp16.config, '_pre_quantization_dtype'))
        self.assertTrue(self.model_4bit.config._pre_quantization_dtype == torch.float16)

    def test_linear_are_4bit(self):
        if False:
            while True:
                i = 10
        '\n        A simple test to check if the model conversion has been done correctly by checking on the\n        memory footprint of the converted model and the class type of the linear layers of the converted models\n        '
        from transformers import T5PreTrainedModel
        self.model_fp16.get_memory_footprint()
        self.model_4bit.get_memory_footprint()
        for (name, module) in self.model_4bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in ['lm_head'] + T5PreTrainedModel._keep_in_fp32_modules:
                    self.assertTrue(module.weight.dtype == torch.uint8)

    def test_rwkv_4bit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A simple test to check if 4-bit RWKV inference works as expected.\n        '
        model_id = 'RWKV/rwkv-4-169m-pile'
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        tok = AutoTokenizer.from_pretrained(model_id)
        text = 'Hello my name is'
        input_ids = tok.encode(text, return_tensors='pt').to(0)
        _ = model.generate(input_ids, max_new_tokens=30)

    def test_generate_quality(self):
        if False:
            i = 10
            return i + 15
        "\n        Test the generation quality of the quantized model and see that we are matching the expected output.\n        Given that we are operating on small numbers + the testing model is relatively small, we might not get\n        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.\n        "
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = self.model_4bit.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality_config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that loading the model with the config is equivalent\n        '
        bnb_config = BitsAndBytesConfig()
        bnb_config.load_in_4bit = True
        model_4bit_from_config = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, device_map='auto')
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = model_4bit_from_config.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_raise_on_save_pretrained(self):
        if False:
            while True:
                i = 10
        '\n        Test whether trying to save a model after converting it in 8-bit will throw a warning.\n        '
        with self.assertRaises(NotImplementedError), tempfile.TemporaryDirectory() as tmpdirname:
            self.model_4bit.save_pretrained(tmpdirname)

    def test_raise_if_config_and_load_in_4bit(self):
        if False:
            return 10
        '\n        Test that loading the model with the config and `load_in_4bit` raises an error\n        '
        bnb_config = BitsAndBytesConfig()
        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, load_in_4bit=True, device_map='auto', bnb_4bit_quant_type='nf4')

    def test_device_and_dtype_assignment(self):
        if False:
            return 10
        '\n        Test whether trying to cast (or assigning a device to) a model after converting it in 8-bit will throw an error.\n        Checks also if other models are casted correctly.\n        '
        with self.assertRaises(ValueError):
            self.model_4bit.to('cpu')
        with self.assertRaises(ValueError):
            self.model_4bit.to(torch.float16)
        with self.assertRaises(ValueError):
            self.model_4bit.to(torch.device('cuda:0'))
        with self.assertRaises(ValueError):
            self.model_4bit.float()
        with self.assertRaises(ValueError):
            self.model_4bit.half()
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        self.model_fp16 = self.model_fp16.to(torch.float32)
        _ = self.model_fp16.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        _ = self.model_fp16.to('cpu')
        _ = self.model_fp16.half()
        _ = self.model_fp16.float()

    def test_fp32_4bit_conversion(self):
        if False:
            print('Hello World!')
        '\n        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.\n        '
        model = AutoModelForSeq2SeqLM.from_pretrained('t5-small', load_in_4bit=True, device_map='auto')
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.float32)

@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class Bnb4BitT5Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.model_name = 't5-small'
        cls.dense_act_model_name = 'google/flan-t5-small'
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = 'Translate in German: Hello, my dog is cute'

    def tearDown(self):
        if False:
            while True:
                i = 10
        '\n        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to\n        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27\n        '
        gc.collect()
        torch.cuda.empty_cache()

    def test_inference_without_keep_in_fp32(self):
        if False:
            return 10
        '\n        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.\n        `flan-t5-small` uses `T5DenseGatedActDense` whereas `t5-small` uses `T5DenseReluDense`. We need to test\n        both cases.\n        '
        from transformers import T5ForConditionalGeneration
        modules = T5ForConditionalGeneration._keep_in_fp32_modules
        T5ForConditionalGeneration._keep_in_fp32_modules = None
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_4bit=True, device_map='auto')
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
        _ = model.generate(**encoded_input)
        model = T5ForConditionalGeneration.from_pretrained(self.dense_act_model_name, load_in_4bit=True, device_map='auto')
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
        _ = model.generate(**encoded_input)
        T5ForConditionalGeneration._keep_in_fp32_modules = modules

    def test_inference_with_keep_in_fp32(self):
        if False:
            while True:
                i = 10
        '\n        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.\n        `flan-t5-small` uses `T5DenseGatedActDense` whereas `t5-small` uses `T5DenseReluDense`. We need to test\n        both cases.\n        '
        import bitsandbytes as bnb
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_4bit=True, device_map='auto')
        self.assertTrue(isinstance(model.decoder.block[0].layer[0].SelfAttention.q, bnb.nn.Linear4bit))
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
        _ = model.generate(**encoded_input)
        model = T5ForConditionalGeneration.from_pretrained(self.dense_act_model_name, load_in_4bit=True, device_map='auto')
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
        _ = model.generate(**encoded_input)

class Classes4BitModelTest(Base4bitTest):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.model_name = 'bigscience/bloom-560m'
        self.seq_to_seq_name = 't5-small'
        self.base_model = AutoModel.from_pretrained(self.model_name, load_in_4bit=True, device_map='auto')
        self.sequence_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, load_in_4bit=True, device_map='auto')
        self.model_4bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True, device_map='auto')
        self.seq_to_seq_model = AutoModelForSeq2SeqLM.from_pretrained(self.seq_to_seq_name, load_in_4bit=True, device_map='auto')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to\n        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27\n        '
        del self.base_model
        del self.sequence_model
        del self.model_4bit
        del self.seq_to_seq_model
        gc.collect()
        torch.cuda.empty_cache()

    def test_correct_head_class(self):
        if False:
            while True:
                i = 10
        '\n        A simple test to check if the last modules for some classes (AutoModelForCausalLM or SequenceClassification)\n        are kept in their native class.\n        '
        from bitsandbytes.nn import Params4bit
        self.assertTrue(self.base_model.h[-1].mlp.dense_4h_to_h.weight.__class__ == Params4bit)
        self.assertTrue(self.model_4bit.lm_head.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.sequence_model.score.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.seq_to_seq_model.lm_head.weight.__class__ == torch.nn.Parameter)

class Pipeline4BitTest(Base4bitTest):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()

    def tearDown(self):
        if False:
            while True:
                i = 10
        '\n        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to\n        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27\n        '
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()

    def test_pipeline(self):
        if False:
            print('Hello World!')
        '\n        The aim of this test is to verify that the mixed 4bit is compatible with `pipeline` from transformers. Since\n        we used pipline for inference speed benchmarking we want to make sure that this feature does not break anything\n        on pipline.\n        '
        self.pipe = pipeline('text-generation', model=self.model_name, model_kwargs={'device_map': 'auto', 'load_in_4bit': True, 'torch_dtype': torch.float16}, max_new_tokens=self.MAX_NEW_TOKENS)
        pipeline_output = self.pipe(self.input_text)
        self.assertIn(pipeline_output[0]['generated_text'], self.EXPECTED_OUTPUTS)

@require_torch_multi_gpu
class Bnb4bitTestMultiGpu(Base4bitTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()

    def test_multi_gpu_loading(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        This tests that the model has been loaded and can be used correctly on a multi-GPU setup.\n        Let's just try to load a model on 2 GPUs and see if it works. The model we test has ~2GB of total, 3GB should suffice\n        "
        model_parallel = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True, device_map='balanced')
        self.assertEqual(set(model_parallel.hf_device_map.values()), {0, 1})
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_parallel = model_parallel.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertIn(self.tokenizer.decode(output_parallel[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

class Bnb4BitTestTraining(Base4bitTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model_name = 'facebook/opt-350m'
        super().setUp()

    def test_training(self):
        if False:
            return 10
        if version.parse(importlib.metadata.version('bitsandbytes')) < version.parse('0.37.0'):
            return
        model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True)
        self.assertEqual(set(model.hf_device_map.values()), {torch.cuda.current_device()})
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        for (_, module) in model.named_modules():
            if 'OPTAttention' in repr(type(module)):
                module.q_proj = LoRALayer(module.q_proj, rank=16)
                module.k_proj = LoRALayer(module.k_proj, rank=16)
                module.v_proj = LoRALayer(module.v_proj, rank=16)
        batch = self.tokenizer('Test batch ', return_tensors='pt').to(0)
        with torch.cuda.amp.autocast():
            out = model.forward(**batch)
            out.logits.norm().backward()
        for module in model.modules():
            if isinstance(module, LoRALayer):
                self.assertTrue(module.adapter[1].weight.grad is not None)
                self.assertTrue(module.adapter[1].weight.grad.norm().item() > 0)
            elif isinstance(module, nn.Embedding):
                self.assertTrue(module.weight.grad is None)

class Bnb4BitGPT2Test(Bnb4BitTest):
    model_name = 'gpt2-xl'
    EXPECTED_RELATIVE_DIFFERENCE = 3.3191854854152187