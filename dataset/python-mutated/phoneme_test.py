import os
import unittest
from pocketsphinx import Decoder, get_model_path
DATADIR = os.path.join(os.path.dirname(__file__), '../../test/data')

class PhonemeTest(unittest.TestCase):

    def test_phoneme(self):
        if False:
            while True:
                i = 10
        decoder = Decoder(allphone=get_model_path('en-us/en-us-phone.lm.bin'), lw=2.0, pip=0.3, beam=1e-200, pbeam=1e-20)
        with open(os.path.join(DATADIR, 'goforward.raw'), 'rb') as stream:
            decoder.start_utt()
            while True:
                buf = stream.read(1024)
                if buf:
                    decoder.process_raw(buf, False, False)
                else:
                    break
            decoder.end_utt()
            print('Best phonemes: ', [seg.word for seg in decoder.seg()])
if __name__ == '__main__':
    unittest.main()