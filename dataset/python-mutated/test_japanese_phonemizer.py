import unittest
from TTS.tts.utils.text.japanese.phonemizer import japanese_text_to_phonemes
_TEST_CASES = '\nどちらに行きますか？/dochiraniikimasuka?\n今日は温泉に、行きます。/kyo:waoNseNni,ikimasu.\n「A」から「Z」までです。/e:karazeqtomadedesu.\nそうですね！/so:desune!\nクジラは哺乳類です。/kujirawahonyu:ruidesu.\nヴィディオを見ます。/bidioomimasu.\n今日は８月22日です/kyo:wahachigatsuniju:ninichidesu\nxyzとαβγ/eqkusuwaizeqtotoarufabe:tagaNma\n値段は$12.34です/nedaNwaju:niteNsaNyoNdorudesu\n'

class TestText(unittest.TestCase):

    def test_japanese_text_to_phonemes(self):
        if False:
            i = 10
            return i + 15
        for line in _TEST_CASES.strip().split('\n'):
            (text, phone) = line.split('/')
            self.assertEqual(japanese_text_to_phonemes(text), phone)
if __name__ == '__main__':
    unittest.main()