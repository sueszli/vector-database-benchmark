import unittest
import os
from pocketsphinx import Decoder, Jsgf
DATADIR = os.path.join(os.path.dirname(__file__), '../../test/data')

class TestJsgf(unittest.TestCase):

    def test_create_jsgf(self):
        if False:
            print('Hello World!')
        jsgf = Jsgf(os.path.join(DATADIR, 'goforward.gram'))
        del jsgf

    def test_jsgf(self):
        if False:
            return 10
        decoder = Decoder(lm=os.path.join(DATADIR, 'turtle.lm.bin'), dict=os.path.join(DATADIR, 'turtle.dic'))
        decoder.start_utt()
        with open(os.path.join(DATADIR, 'goforward.raw'), 'rb') as stream:
            while True:
                buf = stream.read(1024)
                if buf:
                    decoder.process_raw(buf, False, False)
                else:
                    break
        decoder.end_utt()
        print('Decoding with "turtle" language:', decoder.hyp().hypstr)
        self.assertEqual('go forward ten meters', decoder.hyp().hypstr)
        jsgf = Jsgf(os.path.join(DATADIR, 'goforward.gram'))
        rule = jsgf.get_rule('goforward.move2')
        fsg = jsgf.build_fsg(rule, decoder.logmath, 7.5)
        fsg.writefile('goforward.fsg')
        self.assertTrue(os.path.exists('goforward.fsg'))
        os.remove('goforward.fsg')
        decoder.add_fsg('goforward', fsg)
        self.assertNotEqual(decoder.current_search(), 'goforward')
        decoder.activate_search('goforward')
        self.assertEqual(decoder.current_search(), 'goforward')
        self.assertTrue(decoder.get_fsg())
        self.assertTrue(decoder.get_fsg('goforward'))
        self.assertIsNone(decoder.get_lm('foobiebletch'))
        decoder.start_utt()
        with open(os.path.join(DATADIR, 'goforward.raw'), 'rb') as stream:
            while True:
                buf = stream.read(1024)
                if buf:
                    decoder.process_raw(buf, False, False)
                else:
                    break
        decoder.end_utt()
        print('Decoding with "goforward" grammar:', decoder.hyp().hypstr)
        self.assertEqual('go forward ten meters', decoder.hyp().hypstr)
if __name__ == '__main__':
    unittest.main()