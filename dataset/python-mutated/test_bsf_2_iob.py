"""
Tests the conversion code for the lang_uk NER dataset
"""
import unittest
from stanza.utils.datasets.ner.convert_bsf_to_beios import convert_bsf, parse_bsf, BsfInfo
import pytest
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

class TestBsf2Iob(unittest.TestCase):

    def test_1line_follow_markup_iob(self):
        if False:
            i = 10
            return i + 15
        data = 'тележурналіст Василь .'
        bsf_markup = 'T1\tPERS 14 20\tВасиль'
        expected = 'тележурналіст O\nВасиль B-PERS\n. O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))

    def test_1line_2tok_markup_iob(self):
        if False:
            for i in range(10):
                print('nop')
        data = 'тележурналіст Василь Нагірний .'
        bsf_markup = 'T1\tPERS 14 29\tВасиль Нагірний'
        expected = 'тележурналіст O\nВасиль B-PERS\nНагірний I-PERS\n. O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))

    def test_1line_Long_tok_markup_iob(self):
        if False:
            print('Hello World!')
        data = 'А в музеї Гуцульщини і Покуття можна '
        bsf_markup = 'T12\tORG 4 30\tмузеї Гуцульщини і Покуття'
        expected = 'А O\nв O\nмузеї B-ORG\nГуцульщини I-ORG\nі I-ORG\nПокуття I-ORG\nможна O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))

    def test_2line_2tok_markup_iob(self):
        if False:
            print('Hello World!')
        data = 'тележурналіст Василь Нагірний .\nВ івано-франківському видавництві «Лілея НВ» вийшла друком'
        bsf_markup = 'T1\tPERS 14 29\tВасиль Нагірний\nT2\tORG 67 75\tЛілея НВ'
        expected = 'тележурналіст O\nВасиль B-PERS\nНагірний I-PERS\n. O\n\n\nВ O\nівано-франківському O\nвидавництві O\n« O\nЛілея B-ORG\nНВ I-ORG\n» O\nвийшла O\nдруком O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))

    def test_all_multiline_iob(self):
        if False:
            for i in range(10):
                print('nop')
        data = 'його книжечка «А .\nKubler .\nСвітло і тіні маестро» .\nПричому'
        bsf_markup = 'T4\tMISC 15 49\tА .\nKubler .\nСвітло і тіні маестро\n'
        expected = 'його O\nкнижечка O\n« O\nА B-MISC\n. I-MISC\nKubler I-MISC\n. I-MISC\nСвітло I-MISC\nі I-MISC\nтіні I-MISC\nмаестро I-MISC\n» O\n. O\n\n\nПричому O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup, converter='iob'))
if __name__ == '__main__':
    unittest.main()