import unittest
import torch
from transformers import BitsAndBytesConfig
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class QWenTextGenerationPipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.qwen_base = '../qwen_7b_ckpt_modelscope/'
        self.qwen_chat = '../qwen_7b_ckpt_chat_modelscope/'
        self.qwen_base_input = '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是'
        self.qwen_chat_input = ['今天天气真好，我', 'How do you do? ', "What's your", '今夜阳光明媚', '宫廷玉液酒，', '7 * 8 + 32 =? ', '请问把大象关冰箱总共要几步？', '1+3=?', '请将下面这句话翻译为英文：在哪里跌倒就在哪里趴着']

    def run_pipeline_with_model_id(self, model_id, input, init_kwargs={}, run_kwargs={}):
        if False:
            for i in range(10):
                print('nop')
        pipeline_ins = pipeline(task=Tasks.text_generation, model=model_id, **init_kwargs)
        pipeline_ins._model_prepare = True
        result = pipeline_ins(input, **run_kwargs)
        print(result['text'])

    def run_chat_pipeline_with_model_id(self, model_id, inputs, init_kwargs={}, run_kwargs={}):
        if False:
            i = 10
            return i + 15
        pipeline_ins = pipeline(task=Tasks.chat, model=model_id, **init_kwargs)
        pipeline_ins._model_prepare = True
        history = None
        for (turn_idx, query) in enumerate(inputs, start=1):
            results = pipeline_ins(query, history=history)
            (response, history) = (results['response'], results['history'])
            print(f'===== Turn {turn_idx} ====')
            print('Query:', query, end='\n')
            print('Response:', response, end='\n')

    @unittest.skipUnless(test_level() >= 3, 'skip test in current test level')
    def test_qwen_base_with_text_generation(self):
        if False:
            i = 10
            return i + 15
        self.run_pipeline_with_model_id(self.qwen_base, self.qwen_base_input, init_kwargs={'device_map': 'auto'})

    @unittest.skipUnless(test_level() >= 3, 'skip test in current test level')
    def test_qwen_base_with_text_generation_quant_int8(self):
        if False:
            i = 10
            return i + 15
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.run_pipeline_with_model_id(self.qwen_base, self.qwen_base_input, init_kwargs={'device_map': 'auto', 'use_max_memory': True, 'quantization_config': quantization_config})

    @unittest.skipUnless(test_level() >= 3, 'skip test in current test level')
    def test_qwen_base_with_text_generation_quant_int4(self):
        if False:
            return 10
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
        self.run_pipeline_with_model_id(self.qwen_base, self.qwen_base_input, init_kwargs={'device_map': 'auto', 'use_max_memory': True, 'quantization_config': quantization_config})

    @unittest.skipUnless(test_level() >= 3, 'skip test in current test level')
    def test_qwen_chat_with_chat(self):
        if False:
            i = 10
            return i + 15
        self.run_chat_pipeline_with_model_id(self.qwen_chat, self.qwen_chat_input, init_kwargs={'device_map': 'auto'})

    @unittest.skipUnless(test_level() >= 3, 'skip test in current test level')
    def test_qwen_chat_with_chat_quant_int8(self):
        if False:
            for i in range(10):
                print('nop')
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.run_chat_pipeline_with_model_id(self.qwen_chat, self.qwen_chat_input, init_kwargs={'device_map': 'auto', 'use_max_memory': True, 'quantization_config': quantization_config})

    @unittest.skipUnless(test_level() >= 3, 'skip test in current test level')
    def test_qwen_chat_with_chat_quant_int4(self):
        if False:
            i = 10
            return i + 15
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
        self.run_chat_pipeline_with_model_id(self.qwen_chat, self.qwen_chat_input, init_kwargs={'device_map': 'auto', 'use_max_memory': True, 'quantization_config': quantization_config})
if __name__ == '__main__':
    unittest.main()