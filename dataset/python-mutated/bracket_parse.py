"""
Corpus reader for corpora that consist of parenthesis-delineated parse trees.
"""
import sys
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
SORTTAGWRD = re.compile('\\((\\d+) ([^\\s()]+) ([^\\s()]+)\\)')
TAGWORD = re.compile('\\(([^\\s()]+) ([^\\s()]+)\\)')
WORD = re.compile('\\([^\\s()]+ ([^\\s()]+)\\)')
EMPTY_BRACKETS = re.compile('\\s*\\(\\s*\\(')

class BracketParseCorpusReader(SyntaxCorpusReader):
    """
    Reader for corpora that consist of parenthesis-delineated parse trees,
    like those found in the "combined" section of the Penn Treebank,
    e.g. "(S (NP (DT the) (JJ little) (NN dog)) (VP (VBD barked)))".

    """

    def __init__(self, root, fileids, comment_char=None, detect_blocks='unindented_paren', encoding='utf8', tagset=None):
        if False:
            while True:
                i = 10
        "\n        :param root: The root directory for this corpus.\n        :param fileids: A list or regexp specifying the fileids in this corpus.\n        :param comment_char: The character which can appear at the start of\n            a line to indicate that the rest of the line is a comment.\n        :param detect_blocks: The method that is used to find blocks\n            in the corpus; can be 'unindented_paren' (every unindented\n            parenthesis starts a new parse) or 'sexpr' (brackets are\n            matched).\n        :param tagset: The name of the tagset used by this corpus, to be used\n            for normalizing or converting the POS tags returned by the\n            ``tagged_...()`` methods.\n        "
        SyntaxCorpusReader.__init__(self, root, fileids, encoding)
        self._comment_char = comment_char
        self._detect_blocks = detect_blocks
        self._tagset = tagset

    def _read_block(self, stream):
        if False:
            print('Hello World!')
        if self._detect_blocks == 'sexpr':
            return read_sexpr_block(stream, comment_char=self._comment_char)
        elif self._detect_blocks == 'blankline':
            return read_blankline_block(stream)
        elif self._detect_blocks == 'unindented_paren':
            toks = read_regexp_block(stream, start_re='^\\(')
            if self._comment_char:
                toks = [re.sub('(?m)^%s.*' % re.escape(self._comment_char), '', tok) for tok in toks]
            return toks
        else:
            assert 0, 'bad block type'

    def _normalize(self, t):
        if False:
            return 10
        t = re.sub('\\((.)\\)', '(\\1 \\1)', t)
        t = re.sub('\\(([^\\s()]+) ([^\\s()]+) [^\\s()]+\\)', '(\\1 \\2)', t)
        return t

    def _parse(self, t):
        if False:
            i = 10
            return i + 15
        try:
            tree = Tree.fromstring(self._normalize(t))
            if tree.label() == '' and len(tree) == 1:
                return tree[0]
            else:
                return tree
        except ValueError as e:
            sys.stderr.write('Bad tree detected; trying to recover...\n')
            if e.args == ('mismatched parens',):
                for n in range(1, 5):
                    try:
                        v = Tree(self._normalize(t + ')' * n))
                        sys.stderr.write('  Recovered by adding %d close paren(s)\n' % n)
                        return v
                    except ValueError:
                        pass
            sys.stderr.write('  Recovered by returning a flat parse.\n')
            return Tree('S', self._tag(t))

    def _tag(self, t, tagset=None):
        if False:
            while True:
                i = 10
        tagged_sent = [(w, p) for (p, w) in TAGWORD.findall(self._normalize(t))]
        if tagset and tagset != self._tagset:
            tagged_sent = [(w, map_tag(self._tagset, tagset, p)) for (w, p) in tagged_sent]
        return tagged_sent

    def _word(self, t):
        if False:
            i = 10
            return i + 15
        return WORD.findall(self._normalize(t))

class CategorizedBracketParseCorpusReader(CategorizedCorpusReader, BracketParseCorpusReader):
    """
    A reader for parsed corpora whose documents are
    divided into categories based on their file identifiers.
    @author: Nathan Schneider <nschneid@cs.cmu.edu>
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        '\n        Initialize the corpus reader.  Categorization arguments\n        (C{cat_pattern}, C{cat_map}, and C{cat_file}) are passed to\n        the L{CategorizedCorpusReader constructor\n        <CategorizedCorpusReader.__init__>}.  The remaining arguments\n        are passed to the L{BracketParseCorpusReader constructor\n        <BracketParseCorpusReader.__init__>}.\n        '
        CategorizedCorpusReader.__init__(self, kwargs)
        BracketParseCorpusReader.__init__(self, *args, **kwargs)

    def tagged_words(self, fileids=None, categories=None, tagset=None):
        if False:
            return 10
        return super().tagged_words(self._resolve(fileids, categories), tagset)

    def tagged_sents(self, fileids=None, categories=None, tagset=None):
        if False:
            while True:
                i = 10
        return super().tagged_sents(self._resolve(fileids, categories), tagset)

    def tagged_paras(self, fileids=None, categories=None, tagset=None):
        if False:
            while True:
                i = 10
        return super().tagged_paras(self._resolve(fileids, categories), tagset)

    def parsed_words(self, fileids=None, categories=None):
        if False:
            print('Hello World!')
        return super().parsed_words(self._resolve(fileids, categories))

    def parsed_sents(self, fileids=None, categories=None):
        if False:
            i = 10
            return i + 15
        return super().parsed_sents(self._resolve(fileids, categories))

    def parsed_paras(self, fileids=None, categories=None):
        if False:
            while True:
                i = 10
        return super().parsed_paras(self._resolve(fileids, categories))

class AlpinoCorpusReader(BracketParseCorpusReader):
    """
    Reader for the Alpino Dutch Treebank.
    This corpus has a lexical breakdown structure embedded, as read by `_parse`
    Unfortunately this puts punctuation and some other words out of the sentence
    order in the xml element tree. This is no good for `tag_` and `word_`
    `_tag` and `_word` will be overridden to use a non-default new parameter 'ordered'
    to the overridden _normalize function. The _parse function can then remain
    untouched.
    """

    def __init__(self, root, encoding='ISO-8859-1', tagset=None):
        if False:
            return 10
        BracketParseCorpusReader.__init__(self, root, 'alpino\\.xml', detect_blocks='blankline', encoding=encoding, tagset=tagset)

    def _normalize(self, t, ordered=False):
        if False:
            for i in range(10):
                print('nop')
        "Normalize the xml sentence element in t.\n        The sentence elements <alpino_ds>, although embedded in a few overall\n        xml elements, are separated by blank lines. That's how the reader can\n        deliver them one at a time.\n        Each sentence has a few category subnodes that are of no use to us.\n        The remaining word nodes may or may not appear in the proper order.\n        Each word node has attributes, among which:\n        - begin : the position of the word in the sentence\n        - pos   : Part of Speech: the Tag\n        - word  : the actual word\n        The return value is a string with all xml elementes replaced by\n        clauses: either a cat clause with nested clauses, or a word clause.\n        The order of the bracket clauses closely follows the xml.\n        If ordered == True, the word clauses include an order sequence number.\n        If ordered == False, the word clauses only have pos and word parts.\n        "
        if t[:10] != '<alpino_ds':
            return ''
        t = re.sub('  <node .*? cat="(\\w+)".*>', '(\\1', t)
        if ordered:
            t = re.sub('  <node. *?begin="(\\d+)".*? pos="(\\w+)".*? word="([^"]+)".*?/>', '(\\1 \\2 \\3)', t)
        else:
            t = re.sub('  <node .*?pos="(\\w+)".*? word="([^"]+)".*?/>', '(\\1 \\2)', t)
        t = re.sub('  </node>', ')', t)
        t = re.sub('<sentence>.*</sentence>', '', t)
        t = re.sub('</?alpino_ds.*>', '', t)
        return t

    def _tag(self, t, tagset=None):
        if False:
            return 10
        tagged_sent = [(int(o), w, p) for (o, p, w) in SORTTAGWRD.findall(self._normalize(t, ordered=True))]
        tagged_sent.sort()
        if tagset and tagset != self._tagset:
            tagged_sent = [(w, map_tag(self._tagset, tagset, p)) for (o, w, p) in tagged_sent]
        else:
            tagged_sent = [(w, p) for (o, w, p) in tagged_sent]
        return tagged_sent

    def _word(self, t):
        if False:
            for i in range(10):
                print('nop')
        'Return a correctly ordered list if words'
        tagged_sent = self._tag(t)
        return [w for (w, p) in tagged_sent]