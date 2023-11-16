import unittest
from tests.speech import TestFairseqSpeech

class TestXMTransformer(TestFairseqSpeech):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_up_sotasty_es_en()

    def test_sotasty_es_en_600m_checkpoint(self):
        if False:
            print('Hello World!')
        self.base_test(ckpt_name='xm_transformer_600m_es_en_md.pt', reference_score=31.74, score_delta=0.2, max_tokens=3000000, max_positions=(1000000, 1024), dataset='sotasty_es_en_test_ted', arg_overrides={'config_yaml': 'cfg_es_en.yaml'}, score_type='bleu')
if __name__ == '__main__':
    unittest.main()