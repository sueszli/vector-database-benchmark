import pytest
import os
import tempfile
from unittest import TestCase
import shutil
from bigdl.llm import llm_convert
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.optimize import optimize_model, load_low_bit, low_memory_init
llama_model_path = os.environ.get('LLAMA_ORIGIN_PATH')
gptneox_model_path = os.environ.get('GPTNEOX_ORIGIN_PATH')
bloom_model_path = os.environ.get('BLOOM_ORIGIN_PATH')
starcoder_model_path = os.environ.get('STARCODER_ORIGIN_PATH')
output_dir = os.environ.get('INT4_CKPT_DIR')

class TestConvertModel(TestCase):

    def test_convert_llama(self):
        if False:
            print('Hello World!')
        converted_model_path = llm_convert(model=llama_model_path, outfile=output_dir, model_family='llama', model_format='pth', outtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_convert_gptneox(self):
        if False:
            for i in range(10):
                print('nop')
        converted_model_path = llm_convert(model=gptneox_model_path, outfile=output_dir, model_family='gptneox', model_format='pth', outtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_convert_bloom(self):
        if False:
            while True:
                i = 10
        converted_model_path = llm_convert(model=bloom_model_path, outfile=output_dir, model_family='bloom', model_format='pth', outtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_convert_starcoder(self):
        if False:
            i = 10
            return i + 15
        converted_model_path = llm_convert(model=starcoder_model_path, outfile=output_dir, model_family='starcoder', model_format='pth', outtype='int4')
        assert os.path.isfile(converted_model_path)

    def test_transformer_convert_llama(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory(dir=output_dir) as tempdir:
            model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True)
            model.save_low_bit(tempdir)
            newModel = AutoModelForCausalLM.load_low_bit(tempdir)
            assert newModel is not None

    def test_transformer_convert_llama_q5(self):
        if False:
            return 10
        model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_low_bit='sym_int5')

    def test_transformer_convert_llama_q8(self):
        if False:
            return 10
        model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_low_bit='sym_int8')

    def test_transformer_convert_llama_save_load(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory(dir=output_dir) as tempdir:
            model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_low_bit='asym_int4')
            model.save_low_bit(tempdir)
            newModel = AutoModelForCausalLM.load_low_bit(tempdir)
            assert newModel is not None

    def test_optimize_transformers_llama(self):
        if False:
            print('Hello World!')
        from transformers import AutoModelForCausalLM as AutoCLM
        with tempfile.TemporaryDirectory(dir=output_dir) as tempdir:
            model = AutoCLM.from_pretrained(llama_model_path, torch_dtype='auto', low_cpu_mem_usage=True, trust_remote_code=True)
            model = optimize_model(model)
            model.save_low_bit(tempdir)
            with low_memory_init():
                new_model = AutoCLM.from_pretrained(tempdir, torch_dtype='auto', trust_remote_code=True)
            new_model = load_low_bit(new_model, model_path=tempdir)
            assert new_model is not None
if __name__ == '__main__':
    pytest.main([__file__])