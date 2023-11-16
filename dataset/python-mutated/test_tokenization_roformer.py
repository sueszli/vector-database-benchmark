import unittest
from transformers import RoFormerTokenizer, RoFormerTokenizerFast
from transformers.testing_utils import require_rjieba, require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin

@require_rjieba
@require_tokenizers
class RoFormerTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = RoFormerTokenizer
    rust_tokenizer_class = RoFormerTokenizerFast
    space_between_special_tokens = True
    test_rust_tokenizer = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()

    def get_tokenizer(self, **kwargs):
        if False:
            return 10
        return self.tokenizer_class.from_pretrained('junnyu/roformer_chinese_base', **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        if False:
            while True:
                i = 10
        return self.rust_tokenizer_class.from_pretrained('junnyu/roformer_chinese_base', **kwargs)

    def get_chinese_input_output_texts(self):
        if False:
            for i in range(10):
                print('nop')
        input_text = '永和服装饰品有限公司,今天天气非常好'
        output_text = '永和 服装 饰品 有限公司 , 今 天 天 气 非常 好'
        return (input_text, output_text)

    def test_tokenizer(self):
        if False:
            print('Hello World!')
        tokenizer = self.get_tokenizer()
        (input_text, output_text) = self.get_chinese_input_output_texts()
        tokens = tokenizer.tokenize(input_text)
        self.assertListEqual(tokens, output_text.split())
        input_tokens = tokens + [tokenizer.unk_token]
        exp_tokens = [22943, 21332, 34431, 45904, 117, 306, 1231, 1231, 2653, 33994, 1266, 100]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), exp_tokens)

    def test_rust_tokenizer(self):
        if False:
            while True:
                i = 10
        tokenizer = self.get_rust_tokenizer()
        (input_text, output_text) = self.get_chinese_input_output_texts()
        tokens = tokenizer.tokenize(input_text)
        self.assertListEqual(tokens, output_text.split())
        input_tokens = tokens + [tokenizer.unk_token]
        exp_tokens = [22943, 21332, 34431, 45904, 117, 306, 1231, 1231, 2653, 33994, 1266, 100]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), exp_tokens)

    def test_training_new_tokenizer(self):
        if False:
            print('Hello World!')
        pass

    def test_training_new_tokenizer_with_special_tokens_change(self):
        if False:
            return 10
        pass

    def test_save_slow_from_fast_and_reload_fast(self):
        if False:
            i = 10
            return i + 15
        pass