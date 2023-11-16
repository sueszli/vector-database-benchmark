import unittest
from tests.speech import TestFairseqSpeech

class TestS2TConformer(TestFairseqSpeech):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_up_librispeech()

    def test_librispeech_s2t_conformer_s_checkpoint(self):
        if False:
            for i in range(10):
                print('nop')
        self.base_test(ckpt_name='librispeech_conformer_rel_pos_s.pt', reference_score=12, arg_overrides={'config_yaml': 'cfg_librispeech.yaml'})
if __name__ == '__main__':
    unittest.main()