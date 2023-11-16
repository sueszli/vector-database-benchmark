import unittest
from transformers import LlamaForCausalLM, LlamaTokenizer
from modelscope import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class HFUtilTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_auto_tokenizer(self):
        if False:
            return 10
        tokenizer = AutoTokenizer.from_pretrained('baichuan-inc/Baichuan2-7B-Chat', trust_remote_code=True, revision='v1.0.3')
        self.assertEqual(tokenizer.vocab_size, 125696)
        self.assertEqual(tokenizer.model_max_length, 4096)
        self.assertFalse(tokenizer.is_fast)

    def test_quantization_import(self):
        if False:
            while True:
                i = 10
        from modelscope import GPTQConfig, BitsAndBytesConfig
        self.assertTrue(BitsAndBytesConfig is not None)

    def test_auto_model(self):
        if False:
            print('Hello World!')
        model = AutoModelForCausalLM.from_pretrained('baichuan-inc/baichuan-7B', trust_remote_code=True)
        self.assertTrue(model is not None)

    def test_auto_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = AutoConfig.from_pretrained('baichuan-inc/Baichuan-13B-Chat', trust_remote_code=True, revision='v1.0.3')
        self.assertEqual(config.model_type, 'baichuan')
        gen_config = GenerationConfig.from_pretrained('baichuan-inc/Baichuan-13B-Chat', trust_remote_code=True, revision='v1.0.3')
        self.assertEqual(gen_config.assistant_token_id, 196)

    def test_transformer_patch(self):
        if False:
            while True:
                i = 10
        tokenizer = LlamaTokenizer.from_pretrained('skyline2006/llama-7b', revision='v1.0.1')
        self.assertIsNotNone(tokenizer)
        model = LlamaForCausalLM.from_pretrained('skyline2006/llama-7b', revision='v1.0.1')
        self.assertIsNotNone(model)
if __name__ == '__main__':
    unittest.main()