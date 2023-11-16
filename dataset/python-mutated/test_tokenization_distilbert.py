from transformers import DistilBertTokenizer, DistilBertTokenizerFast
from transformers.testing_utils import require_tokenizers, slow
from ..bert.test_tokenization_bert import BertTokenizationTest

@require_tokenizers
class DistilBertTokenizationTest(BertTokenizationTest):
    tokenizer_class = DistilBertTokenizer
    rust_tokenizer_class = DistilBertTokenizerFast
    test_rust_tokenizer = True

    @slow
    def test_sequence_builders(self):
        if False:
            while True:
                i = 10
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        text = tokenizer.encode('sequence builders', add_special_tokens=False)
        text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [tokenizer.sep_token_id]