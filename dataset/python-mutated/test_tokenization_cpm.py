from transformers.models.cpm.tokenization_cpm import CpmTokenizer
from transformers.testing_utils import custom_tokenizers
from ..xlnet.test_modeling_xlnet import XLNetModelTest

@custom_tokenizers
class CpmTokenizationTest(XLNetModelTest):

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            return 10
        return True

    def test_pre_tokenization(self):
        if False:
            return 10
        tokenizer = CpmTokenizer.from_pretrained('TsinghuaAI/CPM-Generate')
        text = 'Hugging Face大法好，谁用谁知道。'
        normalized_text = 'Hugging Face大法好,谁用谁知道。<unk>'
        bpe_tokens = '▁Hu gg ing ▁ ▂ ▁F ace ▁大法 ▁好 ▁ , ▁谁 ▁用 ▁谁 ▁知 道 ▁ 。'.split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [13789, 13283, 1421, 8, 10, 1164, 13608, 16528, 63, 8, 9, 440, 108, 440, 121, 90, 8, 12, 0]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
        reconstructed_text = tokenizer.decode(input_bpe_tokens)
        self.assertEqual(reconstructed_text, normalized_text)