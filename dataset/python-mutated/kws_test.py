import unittest
import sys, os
from pocketsphinx import Decoder
DATADIR = os.path.join(os.path.dirname(__file__), '../../test/data')

class TestKWS(unittest.TestCase):

    def test_kws(self):
        if False:
            print('Hello World!')
        stream = open(os.path.join(DATADIR, 'goforward.raw'), 'rb')
        decoder = Decoder(kws=os.path.join(DATADIR, 'goforward.kws'), loglevel='INFO', lm=None)
        decoder.start_utt()
        keywords = ['forward', 'meters']
        while keywords:
            buf = stream.read(1024)
            if buf:
                decoder.process_raw(buf)
            else:
                break
            if decoder.hyp() != None:
                print([(seg.word, seg.prob, seg.start_frame, seg.end_frame) for seg in decoder.seg()])
                print('Detected keyphrase, restarting search')
                for seg in decoder.seg():
                    self.assertTrue(seg.end_frame > seg.start_frame)
                    self.assertEqual(seg.word, keywords.pop(0))
                decoder.end_utt()
                decoder.start_utt()
        stream.close()
        decoder.end_utt()
        stream = open(os.path.join(DATADIR, 'goforward.raw'), 'rb')
        decoder.start_utt()
        decoder.process_raw(stream.read(), full_utt=True)
        decoder.end_utt()
        print([(seg.word, seg.prob, seg.start_frame, seg.end_frame) for seg in decoder.seg()])
        self.assertEqual(decoder.hyp().hypstr, 'forward meters')
        self.assertEqual(['forward', 'meters'], [seg.word for seg in decoder.seg()])
if __name__ == '__main__':
    unittest.main()