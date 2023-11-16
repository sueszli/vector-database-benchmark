import gc
import importlib.metadata
import tempfile
import unittest
from packaging import version
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers.testing_utils import is_accelerate_available, is_torch_available, require_accelerate, require_bitsandbytes, require_torch, require_torch_gpu, require_torch_multi_gpu, slow

def get_some_linear_layer(model):
    if False:
        while True:
            i = 10
    if model.config.model_type == 'gpt2':
        return model.transformer.h[0].mlp.c_fc
    return model.transformer.h[0].mlp.dense_4h_to_h
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.logging import get_logger
    logger = get_logger(__name__)
    _ = PartialState()
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
                for i in range(10):
                    print('nop')
            return self.module(input, *args, **kwargs) + self.adapter(input)

@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class BaseMixedInt8Test(unittest.TestCase):
    model_name = 'bigscience/bloom-1b7'
    EXPECTED_RELATIVE_DIFFERENCE = 1.540025
    input_text = 'Hello my name is'
    EXPECTED_OUTPUT = 'Hello my name is John.\nI am a friend of the family.\n'
    MAX_NEW_TOKENS = 10

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

class MixedInt8Test(BaseMixedInt8Test):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map='auto')
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map='auto')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        '\n        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to\n        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27\n        '
        del self.model_fp16
        del self.model_8bit
        gc.collect()
        torch.cuda.empty_cache()

    def test_get_keys_to_not_convert_trust_remote_code(self):
        if False:
            print('Hello World!')
        '\n        Test the `get_keys_to_not_convert` function with `trust_remote_code` models.\n        '
        from accelerate import init_empty_weights
        from transformers.integrations.bitsandbytes import get_keys_to_not_convert
        model_id = 'mosaicml/mpt-7b'
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision='ada218f9a93b5f1c6dce48a4cc9ff01fcba431e7')
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, code_revision='ada218f9a93b5f1c6dce48a4cc9ff01fcba431e7')
        self.assertEqual(get_keys_to_not_convert(model), ['transformer.wte'])

    def test_get_keys_to_not_convert(self):
        if False:
            while True:
                i = 10
        '\n        Test the `get_keys_to_not_convert` function.\n        '
        from accelerate import init_empty_weights
        from transformers import AutoModelForMaskedLM, Blip2ForConditionalGeneration, MptForCausalLM, OPTForCausalLM
        from transformers.integrations.bitsandbytes import get_keys_to_not_convert
        model_id = 'mosaicml/mpt-7b'
        config = AutoConfig.from_pretrained(model_id, revision='72e5f594ce36f9cabfa2a9fd8f58b491eb467ee7')
        with init_empty_weights():
            model = MptForCausalLM(config)
        self.assertEqual(get_keys_to_not_convert(model).sort(), ['lm_head', 'transformer.wte'].sort())
        model_id = 'Salesforce/blip2-opt-2.7b'
        config = AutoConfig.from_pretrained(model_id, revision='1ef7f63a8f0a144c13fdca8103eb7b4691c74cec')
        with init_empty_weights():
            model = Blip2ForConditionalGeneration(config)
        self.assertEqual(get_keys_to_not_convert(model).sort(), ['language_model.lm_head', 'language_model.model.decoder.embed_tokens'].sort())
        model_id = 'facebook/opt-350m'
        config = AutoConfig.from_pretrained(model_id, revision='cb32f77e905cccbca1d970436fb0f5e6b58ee3c5')
        with init_empty_weights():
            model = OPTForCausalLM(config)
        self.assertEqual(get_keys_to_not_convert(model).sort(), ['lm_head', 'model.decoder.embed_tokens'].sort())
        model_id = 'roberta-large'
        config = AutoConfig.from_pretrained(model_id, revision='716877d372b884cad6d419d828bac6c85b3b18d9')
        with init_empty_weights():
            model = AutoModelForMaskedLM.from_config(config)
        self.assertEqual(get_keys_to_not_convert(model).sort(), ["'roberta.embeddings.word_embeddings', 'lm_head', 'lm_head.decoder"].sort())

    def test_quantization_config_json_serialization(self):
        if False:
            while True:
                i = 10
        '\n        A simple test to check if the quantization config is correctly serialized and deserialized\n        '
        config = self.model_8bit.config
        self.assertTrue(hasattr(config, 'quantization_config'))
        _ = config.to_dict()
        _ = config.to_diff_dict()
        _ = config.to_json_string()

    def test_original_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A simple test to check if the model succesfully stores the original dtype\n        '
        self.assertTrue(hasattr(self.model_8bit.config, '_pre_quantization_dtype'))
        self.assertFalse(hasattr(self.model_fp16.config, '_pre_quantization_dtype'))
        self.assertTrue(self.model_8bit.config._pre_quantization_dtype == torch.float16)

    def test_memory_footprint(self):
        if False:
            while True:
                i = 10
        '\n        A simple test to check if the model conversion has been done correctly by checking on the\n        memory footprint of the converted model and the class type of the linear layers of the converted models\n        '
        from bitsandbytes.nn import Int8Params
        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_8bit = self.model_8bit.get_memory_footprint()
        self.assertAlmostEqual(mem_fp16 / mem_8bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        self.assertTrue(get_some_linear_layer(self.model_8bit).weight.__class__ == Int8Params)

    def test_linear_are_8bit(self):
        if False:
            print('Hello World!')
        '\n        A simple test to check if the model conversion has been done correctly by checking on the\n        memory footprint of the converted model and the class type of the linear layers of the converted models\n        '
        from transformers import T5PreTrainedModel
        self.model_fp16.get_memory_footprint()
        self.model_8bit.get_memory_footprint()
        for (name, module) in self.model_8bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in ['lm_head'] + T5PreTrainedModel._keep_in_fp32_modules:
                    self.assertTrue(module.weight.dtype == torch.int8)

    def test_llm_skip(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A simple test to check if `llm_int8_skip_modules` works as expected\n        '
        import bitsandbytes as bnb
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=['classifier'])
        seq_classification_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli', quantization_config=quantization_config)
        self.assertTrue(seq_classification_model.roberta.encoder.layer[0].output.dense.weight.dtype == torch.int8)
        self.assertTrue(isinstance(seq_classification_model.roberta.encoder.layer[0].output.dense, bnb.nn.Linear8bitLt))
        self.assertTrue(isinstance(seq_classification_model.classifier.dense, nn.Linear))
        self.assertTrue(seq_classification_model.classifier.dense.weight.dtype != torch.int8)
        self.assertTrue(isinstance(seq_classification_model.classifier.out_proj, nn.Linear))
        self.assertTrue(seq_classification_model.classifier.out_proj != torch.int8)

    def test_generate_quality(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test the generation quality of the quantized model and see that we are matching the expected output.\n        Given that we are operating on small numbers + the testing model is relatively small, we might not get\n        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.\n        "
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = self.model_8bit.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_generate_quality_config(self):
        if False:
            print('Hello World!')
        '\n        Test that loading the model with the config is equivalent\n        '
        bnb_config = BitsAndBytesConfig()
        bnb_config.load_in_8bit = True
        model_8bit_from_config = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, device_map='auto')
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = model_8bit_from_config.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_raise_if_config_and_load_in_8bit(self):
        if False:
            return 10
        '\n        Test that loading the model with the config and `load_in_8bit` raises an error\n        '
        bnb_config = BitsAndBytesConfig()
        with self.assertRaises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, load_in_8bit=True, device_map='auto', llm_int8_enable_fp32_cpu_offload=True)

    def test_device_and_dtype_assignment(self):
        if False:
            while True:
                i = 10
        '\n        Test whether trying to cast (or assigning a device to) a model after converting it in 8-bit will throw an error.\n        Checks also if other models are casted correctly.\n        '
        with self.assertRaises(ValueError):
            self.model_8bit.to('cpu')
        with self.assertRaises(ValueError):
            self.model_8bit.to(torch.float16)
        with self.assertRaises(ValueError):
            self.model_8bit.to(torch.device('cuda:0'))
        with self.assertRaises(ValueError):
            self.model_8bit.float()
        with self.assertRaises(ValueError):
            self.model_8bit.half()
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        self.model_fp16 = self.model_fp16.to(torch.float32)
        _ = self.model_fp16.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        _ = self.model_fp16.to('cpu')
        _ = self.model_fp16.half()
        _ = self.model_fp16.float()

    def test_fp32_int8_conversion(self):
        if False:
            while True:
                i = 10
        '\n        Test whether it is possible to mix both `int8` and `fp32` weights when using `keep_in_fp32_modules` correctly.\n        '
        model = AutoModelForSeq2SeqLM.from_pretrained('t5-small', load_in_8bit=True, device_map='auto')
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.float32)

    def test_int8_serialization(self):
        if False:
            print('Hello World!')
        '\n        Test whether it is possible to serialize a model in 8-bit.\n        '
        from bitsandbytes.nn import Int8Params
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.model_8bit.save_pretrained(tmpdirname)
            config = AutoConfig.from_pretrained(tmpdirname)
            self.assertTrue(hasattr(config, 'quantization_config'))
            model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, load_in_8bit=True, device_map='auto')
            linear = get_some_linear_layer(model_from_saved)
            self.assertTrue(linear.weight.__class__ == Int8Params)
            self.assertTrue(hasattr(linear.weight, 'SCB'))
            encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
            output_sequences = model_from_saved.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
            self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_int8_serialization_regression(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test whether it is possible to serialize a model in 8-bit - using not safetensors\n        '
        from bitsandbytes.nn import Int8Params
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.model_8bit.save_pretrained(tmpdirname, safe_serialization=False)
            config = AutoConfig.from_pretrained(tmpdirname)
            self.assertTrue(hasattr(config, 'quantization_config'))
            model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, load_in_8bit=True, device_map='auto')
            linear = get_some_linear_layer(model_from_saved)
            self.assertTrue(linear.weight.__class__ == Int8Params)
            self.assertTrue(hasattr(linear.weight, 'SCB'))
            encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
            output_sequences = model_from_saved.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
            self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_int8_serialization_sharded(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test whether it is possible to serialize a model in 8-bit - sharded version.\n        '
        from bitsandbytes.nn import Int8Params
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.model_8bit.save_pretrained(tmpdirname, max_shard_size='200MB')
            config = AutoConfig.from_pretrained(tmpdirname)
            self.assertTrue(hasattr(config, 'quantization_config'))
            model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname)
            linear = get_some_linear_layer(model_from_saved)
            self.assertTrue(linear.weight.__class__ == Int8Params)
            self.assertTrue(hasattr(linear.weight, 'SCB'))
            encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
            output_sequences = model_from_saved.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
            self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

    def test_int8_from_pretrained(self):
        if False:
            print('Hello World!')
        '\n        Test whether loading a 8bit model from the Hub works as expected\n        '
        from bitsandbytes.nn import Int8Params
        model_id = 'ybelkada/bloom-1b7-8bit'
        model = AutoModelForCausalLM.from_pretrained(model_id)
        linear = get_some_linear_layer(model)
        self.assertTrue(linear.weight.__class__ == Int8Params)
        self.assertTrue(hasattr(linear.weight, 'SCB'))
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = model.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

@require_bitsandbytes
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class MixedInt8T5Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.model_name = 't5-small'
        cls.dense_act_model_name = 'google/flan-t5-small'
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = 'Translate in German: Hello, my dog is cute'

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        '\n        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to\n        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27\n        '
        gc.collect()
        torch.cuda.empty_cache()

    def test_inference_without_keep_in_fp32(self):
        if False:
            i = 10
            return i + 15
        '\n        Test whether it is possible to mix both `int8` and `fp32` weights when using `keep_in_fp32_modules` correctly.\n        `flan-t5-small` uses `T5DenseGatedActDense` whereas `t5-small` uses `T5DenseReluDense`. We need to test\n        both cases.\n        '
        from transformers import T5ForConditionalGeneration
        modules = T5ForConditionalGeneration._keep_in_fp32_modules
        T5ForConditionalGeneration._keep_in_fp32_modules = None
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_8bit=True, device_map='auto')
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
        _ = model.generate(**encoded_input)
        model = T5ForConditionalGeneration.from_pretrained(self.dense_act_model_name, load_in_8bit=True, device_map='auto')
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
        _ = model.generate(**encoded_input)
        T5ForConditionalGeneration._keep_in_fp32_modules = modules

    def test_inference_with_keep_in_fp32(self):
        if False:
            return 10
        '\n        Test whether it is possible to mix both `int8` and `fp32` weights when using `keep_in_fp32_modules` correctly.\n        `flan-t5-small` uses `T5DenseGatedActDense` whereas `t5-small` uses `T5DenseReluDense`. We need to test\n        both cases.\n        '
        import bitsandbytes as bnb
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_8bit=True, device_map='auto')
        self.assertTrue(isinstance(model.decoder.block[0].layer[0].SelfAttention.q, bnb.nn.Linear8bitLt))
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
        _ = model.generate(**encoded_input)
        model = T5ForConditionalGeneration.from_pretrained(self.dense_act_model_name, load_in_8bit=True, device_map='auto')
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
        _ = model.generate(**encoded_input)

    def test_inference_with_keep_in_fp32_serialized(self):
        if False:
            return 10
        '\n        Test whether it is possible to mix both `int8` and `fp32` weights when using `keep_in_fp32_modules` correctly on\n        a serialized model.\n        `flan-t5-small` uses `T5DenseGatedActDense` whereas `t5-small` uses `T5DenseReluDense`. We need to test\n        both cases.\n        '
        import bitsandbytes as bnb
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, load_in_8bit=True, device_map='auto')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = T5ForConditionalGeneration.from_pretrained(tmp_dir)
            self.assertTrue(isinstance(model.decoder.block[0].layer[0].SelfAttention.q, bnb.nn.Linear8bitLt))
            encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
            _ = model.generate(**encoded_input)
            model = T5ForConditionalGeneration.from_pretrained(self.dense_act_model_name, load_in_8bit=True, device_map='auto')
            encoded_input = self.tokenizer(self.input_text, return_tensors='pt').to(0)
            _ = model.generate(**encoded_input)

class MixedInt8ModelClassesTest(BaseMixedInt8Test):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.model_name = 'bigscience/bloom-560m'
        self.seq_to_seq_name = 't5-small'
        self.base_model = AutoModel.from_pretrained(self.model_name, load_in_8bit=True, device_map='auto')
        self.sequence_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, load_in_8bit=True, device_map='auto')
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map='auto')
        self.seq_to_seq_model = AutoModelForSeq2SeqLM.from_pretrained(self.seq_to_seq_name, load_in_8bit=True, device_map='auto')

    def tearDown(self):
        if False:
            return 10
        '\n        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to\n        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27\n        '
        del self.base_model
        del self.sequence_model
        del self.model_8bit
        del self.seq_to_seq_model
        gc.collect()
        torch.cuda.empty_cache()

    def test_correct_head_class(self):
        if False:
            while True:
                i = 10
        '\n        A simple test to check if the last modules for some classes (AutoModelForCausalLM or SequenceClassification)\n        are kept in their native class.\n        '
        from bitsandbytes.nn import Int8Params
        self.assertTrue(self.base_model.h[-1].mlp.dense_4h_to_h.weight.__class__ == Int8Params)
        self.assertTrue(self.model_8bit.lm_head.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.sequence_model.score.weight.__class__ == torch.nn.Parameter)
        self.assertTrue(self.seq_to_seq_model.lm_head.weight.__class__ == torch.nn.Parameter)

class MixedInt8TestPipeline(BaseMixedInt8Test):

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
            while True:
                i = 10
        '\n        The aim of this test is to verify that the mixed int8 is compatible with `pipeline` from transformers. Since\n        we used pipline for inference speed benchmarking we want to make sure that this feature does not break anything\n        on pipline.\n        '
        self.pipe = pipeline('text-generation', model=self.model_name, model_kwargs={'device_map': 'auto', 'load_in_8bit': True}, max_new_tokens=self.MAX_NEW_TOKENS)
        pipeline_output = self.pipe(self.input_text)
        self.assertEqual(pipeline_output[0]['generated_text'], self.EXPECTED_OUTPUT)

@require_torch_multi_gpu
class MixedInt8TestMultiGpu(BaseMixedInt8Test):

    def setUp(self):
        if False:
            return 10
        super().setUp()

    def test_multi_gpu_loading(self):
        if False:
            i = 10
            return i + 15
        "\n        This tests that the model has been loaded and can be used correctly on a multi-GPU setup.\n        Let's just try to load a model on 2 GPUs and see if it works. The model we test has ~2GB of total, 3GB should suffice\n        "
        model_parallel = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map='balanced')
        self.assertEqual(set(model_parallel.hf_device_map.values()), {0, 1})
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_parallel = model_parallel.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertEqual(self.tokenizer.decode(output_parallel[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)

@require_torch_multi_gpu
class MixedInt8TestCpuGpu(BaseMixedInt8Test):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()

    def check_inference_correctness(self, model):
        if False:
            while True:
                i = 10
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_parallel = model.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        output_text = self.tokenizer.decode(output_parallel[0], skip_special_tokens=True)
        self.assertEqual(output_text, self.EXPECTED_OUTPUT)

    def test_cpu_gpu_loading_random_device_map(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A test to check is dispatching a model on cpu & gpu works correctly using a random `device_map`.\n        '
        device_map = {'transformer.word_embeddings': 0, 'transformer.word_embeddings_layernorm': 0, 'lm_head': 0, 'transformer.h.0': 'cpu', 'transformer.h.1': 'cpu', 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0, 'transformer.h.5': 0, 'transformer.h.6': 0, 'transformer.h.7': 0, 'transformer.h.8': 0, 'transformer.h.9': 1, 'transformer.h.10': 0, 'transformer.h.11': 1, 'transformer.h.12': 0, 'transformer.h.13': 0, 'transformer.h.14': 1, 'transformer.h.15': 0, 'transformer.h.16': 0, 'transformer.h.17': 1, 'transformer.h.18': 1, 'transformer.h.19': 0, 'transformer.h.20': 1, 'transformer.h.21': 1, 'transformer.h.22': 0, 'transformer.h.23': 0, 'transformer.ln_f': 1}
        bnb_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map, quantization_config=bnb_config)
        self.assertEqual(set(model_8bit.hf_device_map.values()), {0, 1, 'cpu'})
        self.check_inference_correctness(model_8bit)

    def test_cpu_gpu_loading_custom_device_map(self):
        if False:
            return 10
        '\n        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.\n        This time the device map is more organized than the test above and uses the abstraction\n        `transformer.h` to encapsulate all the decoder layers.\n        '
        device_map = {'transformer.word_embeddings': 'cpu', 'transformer.word_embeddings_layernorm': 'cpu', 'lm_head': 'cpu', 'transformer.h': 0, 'transformer.ln_f': 1}
        bnb_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map, quantization_config=bnb_config)
        self.assertEqual(set(model_8bit.hf_device_map.values()), {0, 1, 'cpu'})
        self.check_inference_correctness(model_8bit)

    def test_cpu_gpu_disk_loading_custom_device_map(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.\n        This time we also add `disk` on the device_map.\n        '
        device_map = {'transformer.word_embeddings': 0, 'transformer.word_embeddings_layernorm': 'cpu', 'lm_head': 0, 'transformer.h': 1, 'transformer.ln_f': 'disk'}
        bnb_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map, quantization_config=bnb_config, offload_folder=tmpdirname)
            self.assertEqual(set(model_8bit.hf_device_map.values()), {0, 1, 'cpu', 'disk'})
            self.check_inference_correctness(model_8bit)

    def test_cpu_gpu_disk_loading_custom_device_map_kwargs(self):
        if False:
            i = 10
            return i + 15
        '\n        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.\n        This time we also add `disk` on the device_map - using the kwargs directly instead of the quantization config\n        '
        device_map = {'transformer.word_embeddings': 0, 'transformer.word_embeddings_layernorm': 'cpu', 'lm_head': 0, 'transformer.h': 1, 'transformer.ln_f': 'disk'}
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map, load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True, offload_folder=tmpdirname)
            self.assertEqual(set(model_8bit.hf_device_map.values()), {0, 1, 'cpu', 'disk'})
            self.check_inference_correctness(model_8bit)

class MixedInt8TestTraining(BaseMixedInt8Test):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.model_name = 'facebook/opt-350m'
        super().setUp()

    def test_training(self):
        if False:
            return 10
        if version.parse(importlib.metadata.version('bitsandbytes')) < version.parse('0.37.0'):
            return
        model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True)
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

class MixedInt8GPT2Test(MixedInt8Test):
    model_name = 'gpt2-xl'
    EXPECTED_RELATIVE_DIFFERENCE = 1.8720077507258357
    EXPECTED_OUTPUT = "Hello my name is John Doe, and I'm a big fan of"

    def test_int8_from_pretrained(self):
        if False:
            while True:
                i = 10
        '\n        Test whether loading a 8bit model from the Hub works as expected\n        '
        from bitsandbytes.nn import Int8Params
        model_id = 'ybelkada/gpt2-xl-8bit'
        model = AutoModelForCausalLM.from_pretrained(model_id)
        linear = get_some_linear_layer(model)
        self.assertTrue(linear.weight.__class__ == Int8Params)
        self.assertTrue(hasattr(linear.weight, 'SCB'))
        encoded_input = self.tokenizer(self.input_text, return_tensors='pt')
        output_sequences = model.generate(input_ids=encoded_input['input_ids'].to(0), max_new_tokens=10)
        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)