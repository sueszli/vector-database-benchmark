import pytest
from nltk.data import find
from nltk.parse.bllip import BllipParser
from nltk.tree import Tree

@pytest.fixture(scope='module')
def parser():
    if False:
        i = 10
        return i + 15
    model_dir = find('models/bllip_wsj_no_aux').path
    return BllipParser.from_unified_model_dir(model_dir)

def setup_module():
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('bllipparser')

class TestBllipParser:

    def test_parser_loads_a_valid_tree(self, parser):
        if False:
            while True:
                i = 10
        parsed = parser.parse('I saw the man with the telescope')
        tree = next(parsed)
        assert isinstance(tree, Tree)
        assert tree.pformat() == '\n(S1\n  (S\n    (NP (PRP I))\n    (VP\n      (VBD saw)\n      (NP (DT the) (NN man))\n      (PP (IN with) (NP (DT the) (NN telescope))))))\n'.strip()

    def test_tagged_parse_finds_matching_element(self, parser):
        if False:
            i = 10
            return i + 15
        parsed = parser.parse('I saw the man with the telescope')
        tagged_tree = next(parser.tagged_parse([('telescope', 'NN')]))
        assert isinstance(tagged_tree, Tree)
        assert tagged_tree.pformat() == '(S1 (NP (NN telescope)))'