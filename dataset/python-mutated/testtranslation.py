"""
Translation module tests
"""
import os
import unittest
from txtai.pipeline import Translation

class TestTranslation(unittest.TestCase):
    """
    Translation tests.
    """

    def testDetect(self):
        if False:
            print('Hello World!')
        '\n        Test language detection\n        '
        translate = Translation()
        test = ['This is a test language detection.']
        language = translate.detect(test)
        self.assertListEqual(language, ['en'])

    def testDetectWithCustomFunc(self):
        if False:
            print('Hello World!')
        '\n        Test language detection with custom function\n        '

        def dummy_func(text):
            if False:
                while True:
                    i = 10
            return ['en' for x in text]
        translate = Translation(langdetect=dummy_func)
        test = ['This is a test language detection.']
        language = translate.detect(test)
        self.assertListEqual(language, ['en'])

    def testLongTranslation(self):
        if False:
            print('Hello World!')
        '\n        Test a translation longer than max tokenization length\n        '
        translate = Translation()
        text = 'This is a test translation to Spanish. ' * 100
        translation = translate(text, 'es')
        self.assertIsNotNone(translation)

    @unittest.skipIf(os.name == 'nt', 'M2M100 skipped on Windows')
    def testM2M100Translation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a translation using M2M100 models\n        '
        translate = Translation()
        text = translate('This is a test translation to Croatian', 'hr')
        self.assertEqual(text, 'Ovo je testni prijevod na hrvatski')

    def testMarianTranslation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a translation using Marian models\n        '
        translate = Translation()
        text = 'This is a test translation into Spanish'
        translation = translate(text, 'es')
        self.assertEqual(translation, 'Esta es una traducción de prueba al español')
        translation = translate(translation, 'en')
        self.assertEqual(translation, text)

    def testNoLang(self):
        if False:
            while True:
                i = 10
        '\n        Test no matching language id\n        '
        translate = Translation()
        self.assertIsNone(translate.langid([], 'zz'))

    def testNoModel(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test no known available model found\n        '
        translate = Translation()
        self.assertEqual(translate.modelpath('zz', 'en'), 'Helsinki-NLP/opus-mt-mul-en')

    def testNoTranslation(self):
        if False:
            i = 10
            return i + 15
        '\n        Test translation skipped when text already in destination language\n        '
        translate = Translation()
        text = 'This is a test translation to English'
        translation = translate(text, 'en')
        self.assertEqual(text, translation)

    def testTranslationWithShowmodels(self):
        if False:
            print('Hello World!')
        '\n        Test a translation using Marian models and showmodels flag to return\n        model and language.\n        '
        translate = Translation()
        text = 'This is a test translation into Spanish'
        result = translate(text, 'es', showmodels=True)
        (translation, language, modelpath) = result
        self.assertEqual(translation, 'Esta es una traducción de prueba al español')
        self.assertEqual(language, 'en')
        self.assertEqual(modelpath, 'Helsinki-NLP/opus-mt-en-es')
        result = translate(translation, 'en', showmodels=True)
        (translation, language, modelpath) = result
        self.assertEqual(translation, text)
        self.assertEqual(language, 'es')
        self.assertEqual(modelpath, 'Helsinki-NLP/opus-mt-es-en')