import unittest
from transformers import AutoConfig, AutoTokenizer, BertConfig, TensorType, is_flax_available
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, require_flax, slow
if is_flax_available():
    import jax
    from transformers.models.auto.modeling_flax_auto import FlaxAutoModel
    from transformers.models.bert.modeling_flax_bert import FlaxBertModel
    from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModel

@require_flax
class FlaxAutoModelTest(unittest.TestCase):

    @slow
    def test_bert_from_pretrained(self):
        if False:
            while True:
                i = 10
        for model_name in ['bert-base-cased', 'bert-large-uncased']:
            with self.subTest(model_name):
                config = AutoConfig.from_pretrained(model_name)
                self.assertIsNotNone(config)
                self.assertIsInstance(config, BertConfig)
                model = FlaxAutoModel.from_pretrained(model_name)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, FlaxBertModel)

    @slow
    def test_roberta_from_pretrained(self):
        if False:
            return 10
        for model_name in ['roberta-base', 'roberta-large']:
            with self.subTest(model_name):
                config = AutoConfig.from_pretrained(model_name)
                self.assertIsNotNone(config)
                self.assertIsInstance(config, BertConfig)
                model = FlaxAutoModel.from_pretrained(model_name)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, FlaxRobertaModel)

    @slow
    def test_bert_jax_jit(self):
        if False:
            while True:
                i = 10
        for model_name in ['bert-base-cased', 'bert-large-uncased']:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = FlaxBertModel.from_pretrained(model_name)
            tokens = tokenizer('Do you support jax jitted function?', return_tensors=TensorType.JAX)

            @jax.jit
            def eval(**kwargs):
                if False:
                    return 10
                return model(**kwargs)
            eval(**tokens).block_until_ready()

    @slow
    def test_roberta_jax_jit(self):
        if False:
            return 10
        for model_name in ['roberta-base', 'roberta-large']:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = FlaxRobertaModel.from_pretrained(model_name)
            tokens = tokenizer('Do you support jax jitted function?', return_tensors=TensorType.JAX)

            @jax.jit
            def eval(**kwargs):
                if False:
                    print('Hello World!')
                return model(**kwargs)
            eval(**tokens).block_until_ready()

    def test_repo_not_found(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(EnvironmentError, 'bert-base is not a local folder and is not a valid model identifier'):
            _ = FlaxAutoModel.from_pretrained('bert-base')

    def test_revision_not_found(self):
        if False:
            return 10
        with self.assertRaisesRegex(EnvironmentError, 'aaaaaa is not a valid git identifier \\(branch name, tag name or commit id\\)'):
            _ = FlaxAutoModel.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision='aaaaaa')

    def test_model_file_not_found(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(EnvironmentError, 'hf-internal-testing/config-no-model does not appear to have a file named flax_model.msgpack'):
            _ = FlaxAutoModel.from_pretrained('hf-internal-testing/config-no-model')

    def test_model_from_pt_suggestion(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(EnvironmentError, 'Use `from_pt=True` to load this model'):
            _ = FlaxAutoModel.from_pretrained('hf-internal-testing/tiny-bert-pt-only')