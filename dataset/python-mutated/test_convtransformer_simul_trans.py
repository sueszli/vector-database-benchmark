import unittest
from tests.speech import TestFairseqSpeech
S3_BASE_URL = 'https://dl.fbaipublicfiles.com/fairseq/'

class TestConvtransformerSimulTrans(TestFairseqSpeech):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._set_up('simul', 'speech_tests/simul', ['config_gcmvn_specaug.yaml', 'dict.txt', 'dev.tsv'])

    def test_waitk_checkpoint(self):
        if False:
            return 10
        "Only test model loading since fairseq currently doesn't support inference of simultaneous models"
        (_, _, _, _) = self.download_and_load_checkpoint('checkpoint_best.pt', arg_overrides={'config_yaml': 'config_gcmvn_specaug.yaml', 'load_pretrained_encoder_from': None})
        return
if __name__ == '__main__':
    unittest.main()