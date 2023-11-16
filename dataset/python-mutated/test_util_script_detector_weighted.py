from test.picardtestcase import PicardTestCase
from picard.util.script_detector_weighted import detect_script_weighted, list_script_weighted

class WeightedScriptDetectionTest(PicardTestCase):

    def test_detect_script_weighted(self):
        if False:
            for i in range(10):
                print('nop')
        scripts = detect_script_weighted('Latin, кириллический, Ελληνική')
        self.assertAlmostEqual(scripts['LATIN'], 0.195, 3)
        self.assertAlmostEqual(scripts['CYRILLIC'], 0.518, 3)
        self.assertAlmostEqual(scripts['GREEK'], 0.287, 3)
        scripts = detect_script_weighted('Latin, кириллический, Ελληνική', threshold=0.5)
        script_keys = list(scripts.keys())
        self.assertEqual(script_keys, ['CYRILLIC'])
        scripts = detect_script_weighted('Latin')
        self.assertEqual(scripts['LATIN'], 1)
        scripts = detect_script_weighted('привет')
        self.assertEqual(scripts['CYRILLIC'], 1)
        scripts = detect_script_weighted('ελληνικά?')
        self.assertEqual(scripts['GREEK'], 1)
        scripts = detect_script_weighted('سماوي يدور')
        self.assertEqual(scripts['ARABIC'], 1)
        scripts = detect_script_weighted('שלום')
        self.assertEqual(scripts['HEBREW'], 1)
        scripts = detect_script_weighted('汉字')
        self.assertEqual(scripts['CJK'], 1)
        scripts = detect_script_weighted('한글')
        self.assertEqual(scripts['HANGUL'], 1)
        scripts = detect_script_weighted('ひらがな')
        self.assertEqual(scripts['HIRAGANA'], 1)
        scripts = detect_script_weighted('カタカナ')
        self.assertEqual(scripts['KATAKANA'], 1)
        scripts = detect_script_weighted('พยัญชนะ')
        self.assertEqual(scripts['THAI'], 1)
        scripts = detect_script_weighted('1234567890+-/*=,./!?')
        self.assertEqual(scripts, {})
        scripts = detect_script_weighted('')
        self.assertEqual(scripts, {})

class ListScriptWeightedTest(PicardTestCase):

    def test_list_script_weighted(self):
        if False:
            i = 10
            return i + 15
        scripts = list_script_weighted('Cyrillic, кириллический, 汉字')
        self.assertEqual(scripts, ['CYRILLIC', 'LATIN', 'CJK'])
        scripts = list_script_weighted('Cyrillic, кириллический, 汉字', threshold=0.3)
        self.assertEqual(scripts, ['CYRILLIC', 'LATIN'])