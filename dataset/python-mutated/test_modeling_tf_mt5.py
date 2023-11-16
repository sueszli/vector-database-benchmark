from __future__ import annotations
import unittest
from transformers import is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow
if is_tf_available():
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

@require_tf
@require_sentencepiece
@require_tokenizers
class TFMT5ModelIntegrationTest(unittest.TestCase):

    @slow
    def test_small_integration_test(self):
        if False:
            return 10
        '\n        For comparision run:\n        >>> import t5  # pip install t5==0.7.1\n        >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary\n\n        >>> path_to_mtf_small_mt5_checkpoint = \'<fill_in>\'\n        >>> path_to_mtf_small_mt5_spm_model_path = \'<fill_in>\'\n        >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_mt5_checkpoint, batch_size=1, tpu=None)\n        >>> vocab = SentencePieceVocabulary(path_to_mtf_small_mt5_spm_model_path, extra_ids=100)\n        >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)\n        '
        model = TFAutoModelForSeq2SeqLM.from_pretrained('google/mt5-small')
        tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
        input_ids = tokenizer('Hello there', return_tensors='tf').input_ids
        labels = tokenizer('Hi I am', return_tensors='tf').input_ids
        loss = model(input_ids, labels=labels).loss
        mtf_score = -tf.math.reduce_mean(loss).numpy()
        EXPECTED_SCORE = -21.228168
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 0.0002)