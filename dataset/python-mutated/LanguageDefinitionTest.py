import unittest
from coalib.bearlib.languages.LanguageDefinition import LanguageDefinition
from coalib.settings.Section import Section
from coalib.settings.Setting import Setting

class LanguageDefinitionTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.section = Section('any')
        self.section.append(Setting('language', 'CPP'))

    def test_key_contains(self):
        if False:
            print('Hello World!')
        uut = LanguageDefinition.from_section(self.section)
        self.assertIn('extensions', uut)
        self.assertNotIn('randomstuff', uut)