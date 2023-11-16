from nltk.parse.api import ParserI
from nltk.tree import Tree
'\nInterface for parsing with BLLIP Parser. Requires the Python\nbllipparser module. BllipParser objects can be constructed with the\n``BllipParser.from_unified_model_dir`` class method or manually using the\n``BllipParser`` constructor. The former is generally easier if you have\na BLLIP Parser unified model directory -- a basic model can be obtained\nfrom NLTK\'s downloader. More unified parsing models can be obtained with\nBLLIP Parser\'s ModelFetcher (run ``python -m bllipparser.ModelFetcher``\nor see docs for ``bllipparser.ModelFetcher.download_and_install_model``).\n\nBasic usage::\n\n    # download and install a basic unified parsing model (Wall Street Journal)\n    # sudo python -m nltk.downloader bllip_wsj_no_aux\n\n    >>> from nltk.data import find\n    >>> model_dir = find(\'models/bllip_wsj_no_aux\').path\n    >>> bllip = BllipParser.from_unified_model_dir(model_dir)\n\n    # 1-best parsing\n    >>> sentence1 = \'British left waffles on Falklands .\'.split()\n    >>> top_parse = bllip.parse_one(sentence1)\n    >>> print(top_parse)\n    (S1\n      (S\n        (NP (JJ British) (NN left))\n        (VP (VBZ waffles) (PP (IN on) (NP (NNP Falklands))))\n        (. .)))\n\n    # n-best parsing\n    >>> sentence2 = \'Time flies\'.split()\n    >>> all_parses = bllip.parse_all(sentence2)\n    >>> print(len(all_parses))\n    50\n    >>> print(all_parses[0])\n    (S1 (S (NP (NNP Time)) (VP (VBZ flies))))\n\n    # incorporating external tagging constraints (None means unconstrained tag)\n    >>> constrained1 = bllip.tagged_parse([(\'Time\', \'VB\'), (\'flies\', \'NNS\')])\n    >>> print(next(constrained1))\n    (S1 (NP (VB Time) (NNS flies)))\n    >>> constrained2 = bllip.tagged_parse([(\'Time\', \'NN\'), (\'flies\', None)])\n    >>> print(next(constrained2))\n    (S1 (NP (NN Time) (VBZ flies)))\n\nReferences\n----------\n\n- Charniak, Eugene. "A maximum-entropy-inspired parser." Proceedings of\n  the 1st North American chapter of the Association for Computational\n  Linguistics conference. Association for Computational Linguistics,\n  2000.\n\n- Charniak, Eugene, and Mark Johnson. "Coarse-to-fine n-best parsing\n  and MaxEnt discriminative reranking." Proceedings of the 43rd Annual\n  Meeting on Association for Computational Linguistics. Association\n  for Computational Linguistics, 2005.\n\nKnown issues\n------------\n\nNote that BLLIP Parser is not currently threadsafe. Since this module\nuses a SWIG interface, it is potentially unsafe to create multiple\n``BllipParser`` objects in the same process. BLLIP Parser currently\nhas issues with non-ASCII text and will raise an error if given any.\n\nSee https://pypi.python.org/pypi/bllipparser/ for more information\non BLLIP Parser\'s Python interface.\n'
__all__ = ['BllipParser']
try:
    from bllipparser import RerankingParser
    from bllipparser.RerankingParser import get_unified_model_parameters

    def _ensure_bllip_import_or_error():
        if False:
            print('Hello World!')
        pass
except ImportError as ie:

    def _ensure_bllip_import_or_error(ie=ie):
        if False:
            while True:
                i = 10
        raise ImportError("Couldn't import bllipparser module: %s" % ie)

def _ensure_ascii(words):
    if False:
        print('Hello World!')
    try:
        for (i, word) in enumerate(words):
            word.encode('ascii')
    except UnicodeEncodeError as e:
        raise ValueError(f"Token {i} ({word!r}) is non-ASCII. BLLIP Parser currently doesn't support non-ASCII inputs.") from e

def _scored_parse_to_nltk_tree(scored_parse):
    if False:
        i = 10
        return i + 15
    return Tree.fromstring(str(scored_parse.ptb_parse))

class BllipParser(ParserI):
    """
    Interface for parsing with BLLIP Parser. BllipParser objects can be
    constructed with the ``BllipParser.from_unified_model_dir`` class
    method or manually using the ``BllipParser`` constructor.
    """

    def __init__(self, parser_model=None, reranker_features=None, reranker_weights=None, parser_options=None, reranker_options=None):
        if False:
            i = 10
            return i + 15
        "\n        Load a BLLIP Parser model from scratch. You'll typically want to\n        use the ``from_unified_model_dir()`` class method to construct\n        this object.\n\n        :param parser_model: Path to parser model directory\n        :type parser_model: str\n\n        :param reranker_features: Path the reranker model's features file\n        :type reranker_features: str\n\n        :param reranker_weights: Path the reranker model's weights file\n        :type reranker_weights: str\n\n        :param parser_options: optional dictionary of parser options, see\n            ``bllipparser.RerankingParser.RerankingParser.load_parser_options()``\n            for more information.\n        :type parser_options: dict(str)\n\n        :param reranker_options: optional\n            dictionary of reranker options, see\n            ``bllipparser.RerankingParser.RerankingParser.load_reranker_model()``\n            for more information.\n        :type reranker_options: dict(str)\n        "
        _ensure_bllip_import_or_error()
        parser_options = parser_options or {}
        reranker_options = reranker_options or {}
        self.rrp = RerankingParser()
        self.rrp.load_parser_model(parser_model, **parser_options)
        if reranker_features and reranker_weights:
            self.rrp.load_reranker_model(features_filename=reranker_features, weights_filename=reranker_weights, **reranker_options)

    def parse(self, sentence):
        if False:
            while True:
                i = 10
        "\n        Use BLLIP Parser to parse a sentence. Takes a sentence as a list\n        of words; it will be automatically tagged with this BLLIP Parser\n        instance's tagger.\n\n        :return: An iterator that generates parse trees for the sentence\n            from most likely to least likely.\n\n        :param sentence: The sentence to be parsed\n        :type sentence: list(str)\n        :rtype: iter(Tree)\n        "
        _ensure_ascii(sentence)
        nbest_list = self.rrp.parse(sentence)
        for scored_parse in nbest_list:
            yield _scored_parse_to_nltk_tree(scored_parse)

    def tagged_parse(self, word_and_tag_pairs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Use BLLIP to parse a sentence. Takes a sentence as a list of\n        (word, tag) tuples; the sentence must have already been tokenized\n        and tagged. BLLIP will attempt to use the tags provided but may\n        use others if it can't come up with a complete parse subject\n        to those constraints. You may also specify a tag as ``None``\n        to leave a token's tag unconstrained.\n\n        :return: An iterator that generates parse trees for the sentence\n            from most likely to least likely.\n\n        :param sentence: Input sentence to parse as (word, tag) pairs\n        :type sentence: list(tuple(str, str))\n        :rtype: iter(Tree)\n        "
        words = []
        tag_map = {}
        for (i, (word, tag)) in enumerate(word_and_tag_pairs):
            words.append(word)
            if tag is not None:
                tag_map[i] = tag
        _ensure_ascii(words)
        nbest_list = self.rrp.parse_tagged(words, tag_map)
        for scored_parse in nbest_list:
            yield _scored_parse_to_nltk_tree(scored_parse)

    @classmethod
    def from_unified_model_dir(cls, model_dir, parser_options=None, reranker_options=None):
        if False:
            print('Hello World!')
        '\n        Create a ``BllipParser`` object from a unified parsing model\n        directory. Unified parsing model directories are a standardized\n        way of storing BLLIP parser and reranker models together on disk.\n        See ``bllipparser.RerankingParser.get_unified_model_parameters()``\n        for more information about unified model directories.\n\n        :return: A ``BllipParser`` object using the parser and reranker\n            models in the model directory.\n\n        :param model_dir: Path to the unified model directory.\n        :type model_dir: str\n        :param parser_options: optional dictionary of parser options, see\n            ``bllipparser.RerankingParser.RerankingParser.load_parser_options()``\n            for more information.\n        :type parser_options: dict(str)\n        :param reranker_options: optional dictionary of reranker options, see\n            ``bllipparser.RerankingParser.RerankingParser.load_reranker_model()``\n            for more information.\n        :type reranker_options: dict(str)\n        :rtype: BllipParser\n        '
        (parser_model_dir, reranker_features_filename, reranker_weights_filename) = get_unified_model_parameters(model_dir)
        return cls(parser_model_dir, reranker_features_filename, reranker_weights_filename, parser_options, reranker_options)

def demo():
    if False:
        while True:
            i = 10
    'This assumes the Python module bllipparser is installed.'
    from nltk.data import find
    model_dir = find('models/bllip_wsj_no_aux').path
    print('Loading BLLIP Parsing models...')
    bllip = BllipParser.from_unified_model_dir(model_dir)
    print('Done.')
    sentence1 = 'British left waffles on Falklands .'.split()
    sentence2 = 'I saw the man with the telescope .'.split()
    fail1 = '# ! ? : -'.split()
    for sentence in (sentence1, sentence2, fail1):
        print('Sentence: %r' % ' '.join(sentence))
        try:
            tree = next(bllip.parse(sentence))
            print(tree)
        except StopIteration:
            print('(parse failed)')
    for (i, parse) in enumerate(bllip.parse(sentence1)):
        print('parse %d:\n%s' % (i, parse))
    print("forcing 'tree' to be 'NN':", next(bllip.tagged_parse([('A', None), ('tree', 'NN')])))
    print("forcing 'A' to be 'DT' and 'tree' to be 'NNP':", next(bllip.tagged_parse([('A', 'DT'), ('tree', 'NNP')])))
    print("forcing 'A' to be 'NNP':", next(bllip.tagged_parse([('A', 'NNP'), ('tree', None)])))