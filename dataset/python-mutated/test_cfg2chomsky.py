import unittest
import nltk
from nltk.grammar import CFG

class ChomskyNormalFormForCFGTest(unittest.TestCase):

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        grammar = CFG.fromstring("\n          S -> NP VP\n          PP -> P NP\n          NP -> Det N | NP PP P\n          VP -> V NP | VP PP\n          VP -> Det\n          Det -> 'a' | 'the'\n          N -> 'dog' | 'cat'\n          V -> 'chased' | 'sat'\n          P -> 'on' | 'in'\n        ")
        self.assertFalse(grammar.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar.is_chomsky_normal_form())
        grammar = grammar.chomsky_normal_form(flexible=True)
        self.assertTrue(grammar.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar.is_chomsky_normal_form())
        grammar2 = CFG.fromstring("\n          S -> NP VP\n          NP -> VP N P\n          VP -> P\n          N -> 'dog' | 'cat'\n          P -> 'on' | 'in'\n        ")
        self.assertFalse(grammar2.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar2.is_chomsky_normal_form())
        grammar2 = grammar2.chomsky_normal_form()
        self.assertTrue(grammar2.is_flexible_chomsky_normal_form())
        self.assertTrue(grammar2.is_chomsky_normal_form())

    def test_complex(self):
        if False:
            print('Hello World!')
        grammar = nltk.data.load('grammars/large_grammars/atis.cfg')
        self.assertFalse(grammar.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar.is_chomsky_normal_form())
        grammar = grammar.chomsky_normal_form(flexible=True)
        self.assertTrue(grammar.is_flexible_chomsky_normal_form())
        self.assertFalse(grammar.is_chomsky_normal_form())