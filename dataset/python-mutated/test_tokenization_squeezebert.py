from transformers import SqueezeBertTokenizer, SqueezeBertTokenizerFast
from transformers.testing_utils import require_tokenizers, slow
from ..bert.test_tokenization_bert import BertTokenizationTest

@require_tokenizers
class SqueezeBertTokenizationTest(BertTokenizationTest):
    tokenizer_class = SqueezeBertTokenizer
    rust_tokenizer_class = SqueezeBertTokenizerFast
    test_rust_tokenizer = True

    def get_rust_tokenizer(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return SqueezeBertTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    @slow
    def test_sequence_builders(self):
        if False:
            i = 10
            return i + 15
        tokenizer = SqueezeBertTokenizer.from_pretrained('squeezebert/squeezebert-mnli-headless')
        text = tokenizer.encode('sequence builders', add_special_tokens=False)
        text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [tokenizer.sep_token_id]