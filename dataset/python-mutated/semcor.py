"""
Corpus reader for the SemCor Corpus.
"""
__docformat__ = 'epytext en'
from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from nltk.tree import Tree

class SemcorCorpusReader(XMLCorpusReader):
    """
    Corpus reader for the SemCor Corpus.
    For access to the complete XML data structure, use the ``xml()``
    method.  For access to simple word lists and tagged word lists, use
    ``words()``, ``sents()``, ``tagged_words()``, and ``tagged_sents()``.
    """

    def __init__(self, root, fileids, wordnet, lazy=True):
        if False:
            for i in range(10):
                print('nop')
        XMLCorpusReader.__init__(self, root, fileids)
        self._lazy = lazy
        self._wordnet = wordnet

    def words(self, fileids=None):
        if False:
            print('Hello World!')
        '\n        :return: the given file(s) as a list of words and punctuation symbols.\n        :rtype: list(str)\n        '
        return self._items(fileids, 'word', False, False, False)

    def chunks(self, fileids=None):
        if False:
            while True:
                i = 10
        '\n        :return: the given file(s) as a list of chunks,\n            each of which is a list of words and punctuation symbols\n            that form a unit.\n        :rtype: list(list(str))\n        '
        return self._items(fileids, 'chunk', False, False, False)

    def tagged_chunks(self, fileids=None, tag='pos' or 'sem' or 'both'):
        if False:
            print('Hello World!')
        "\n        :return: the given file(s) as a list of tagged chunks, represented\n            in tree form.\n        :rtype: list(Tree)\n\n        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`\n            to indicate the kind of tags to include.  Semantic tags consist of\n            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity\n            without a specific entry in WordNet.  (Named entities of type 'other'\n            have no lemma.  Other chunks not in WordNet have no semantic tag.\n            Punctuation tokens have `None` for their part of speech tag.)\n        "
        return self._items(fileids, 'chunk', False, tag != 'sem', tag != 'pos')

    def sents(self, fileids=None):
        if False:
            return 10
        '\n        :return: the given file(s) as a list of sentences, each encoded\n            as a list of word strings.\n        :rtype: list(list(str))\n        '
        return self._items(fileids, 'word', True, False, False)

    def chunk_sents(self, fileids=None):
        if False:
            i = 10
            return i + 15
        '\n        :return: the given file(s) as a list of sentences, each encoded\n            as a list of chunks.\n        :rtype: list(list(list(str)))\n        '
        return self._items(fileids, 'chunk', True, False, False)

    def tagged_sents(self, fileids=None, tag='pos' or 'sem' or 'both'):
        if False:
            print('Hello World!')
        "\n        :return: the given file(s) as a list of sentences. Each sentence\n            is represented as a list of tagged chunks (in tree form).\n        :rtype: list(list(Tree))\n\n        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`\n            to indicate the kind of tags to include.  Semantic tags consist of\n            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity\n            without a specific entry in WordNet.  (Named entities of type 'other'\n            have no lemma.  Other chunks not in WordNet have no semantic tag.\n            Punctuation tokens have `None` for their part of speech tag.)\n        "
        return self._items(fileids, 'chunk', True, tag != 'sem', tag != 'pos')

    def _items(self, fileids, unit, bracket_sent, pos_tag, sem_tag):
        if False:
            i = 10
            return i + 15
        if unit == 'word' and (not bracket_sent):
            _ = lambda *args: LazyConcatenation((SemcorWordView if self._lazy else self._words)(*args))
        else:
            _ = SemcorWordView if self._lazy else self._words
        return concat([_(fileid, unit, bracket_sent, pos_tag, sem_tag, self._wordnet) for fileid in self.abspaths(fileids)])

    def _words(self, fileid, unit, bracket_sent, pos_tag, sem_tag):
        if False:
            i = 10
            return i + 15
        "\n        Helper used to implement the view methods -- returns a list of\n        tokens, (segmented) words, chunks, or sentences. The tokens\n        and chunks may optionally be tagged (with POS and sense\n        information).\n\n        :param fileid: The name of the underlying file.\n        :param unit: One of `'token'`, `'word'`, or `'chunk'`.\n        :param bracket_sent: If true, include sentence bracketing.\n        :param pos_tag: Whether to include part-of-speech tags.\n        :param sem_tag: Whether to include semantic tags, namely WordNet lemma\n            and OOV named entity status.\n        "
        assert unit in ('token', 'word', 'chunk')
        result = []
        xmldoc = ElementTree.parse(fileid).getroot()
        for xmlsent in xmldoc.findall('.//s'):
            sent = []
            for xmlword in _all_xmlwords_in(xmlsent):
                itm = SemcorCorpusReader._word(xmlword, unit, pos_tag, sem_tag, self._wordnet)
                if unit == 'word':
                    sent.extend(itm)
                else:
                    sent.append(itm)
            if bracket_sent:
                result.append(SemcorSentence(xmlsent.attrib['snum'], sent))
            else:
                result.extend(sent)
        assert None not in result
        return result

    @staticmethod
    def _word(xmlword, unit, pos_tag, sem_tag, wordnet):
        if False:
            for i in range(10):
                print('nop')
        tkn = xmlword.text
        if not tkn:
            tkn = ''
        lemma = xmlword.get('lemma', tkn)
        lexsn = xmlword.get('lexsn')
        if lexsn is not None:
            sense_key = lemma + '%' + lexsn
            wnpos = ('n', 'v', 'a', 'r', 's')[int(lexsn.split(':')[0]) - 1]
        else:
            sense_key = wnpos = None
        redef = xmlword.get('rdf', tkn)
        sensenum = xmlword.get('wnsn')
        isOOVEntity = 'pn' in xmlword.keys()
        pos = xmlword.get('pos')
        if unit == 'token':
            if not pos_tag and (not sem_tag):
                itm = tkn
            else:
                itm = (tkn,) + ((pos,) if pos_tag else ()) + ((lemma, wnpos, sensenum, isOOVEntity) if sem_tag else ())
            return itm
        else:
            ww = tkn.split('_')
            if unit == 'word':
                return ww
            else:
                if sensenum is not None:
                    try:
                        sense = wordnet.lemma_from_key(sense_key)
                    except Exception:
                        try:
                            sense = '%s.%s.%02d' % (lemma, wnpos, int(sensenum))
                        except ValueError:
                            sense = lemma + '.' + wnpos + '.' + sensenum
                bottom = [Tree(pos, ww)] if pos_tag else ww
                if sem_tag and isOOVEntity:
                    if sensenum is not None:
                        return Tree(sense, [Tree('NE', bottom)])
                    else:
                        return Tree('NE', bottom)
                elif sem_tag and sensenum is not None:
                    return Tree(sense, bottom)
                elif pos_tag:
                    return bottom[0]
                else:
                    return bottom

def _all_xmlwords_in(elt, result=None):
    if False:
        return 10
    if result is None:
        result = []
    for child in elt:
        if child.tag in ('wf', 'punc'):
            result.append(child)
        else:
            _all_xmlwords_in(child, result)
    return result

class SemcorSentence(list):
    """
    A list of words, augmented by an attribute ``num`` used to record
    the sentence identifier (the ``n`` attribute from the XML).
    """

    def __init__(self, num, items):
        if False:
            print('Hello World!')
        self.num = num
        list.__init__(self, items)

class SemcorWordView(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with the BNC corpus.
    """

    def __init__(self, fileid, unit, bracket_sent, pos_tag, sem_tag, wordnet):
        if False:
            for i in range(10):
                print('nop')
        "\n        :param fileid: The name of the underlying file.\n        :param unit: One of `'token'`, `'word'`, or `'chunk'`.\n        :param bracket_sent: If true, include sentence bracketing.\n        :param pos_tag: Whether to include part-of-speech tags.\n        :param sem_tag: Whether to include semantic tags, namely WordNet lemma\n            and OOV named entity status.\n        "
        if bracket_sent:
            tagspec = '.*/s'
        else:
            tagspec = '.*/s/(punc|wf)'
        self._unit = unit
        self._sent = bracket_sent
        self._pos_tag = pos_tag
        self._sem_tag = sem_tag
        self._wordnet = wordnet
        XMLCorpusView.__init__(self, fileid, tagspec)

    def handle_elt(self, elt, context):
        if False:
            i = 10
            return i + 15
        if self._sent:
            return self.handle_sent(elt)
        else:
            return self.handle_word(elt)

    def handle_word(self, elt):
        if False:
            for i in range(10):
                print('nop')
        return SemcorCorpusReader._word(elt, self._unit, self._pos_tag, self._sem_tag, self._wordnet)

    def handle_sent(self, elt):
        if False:
            while True:
                i = 10
        sent = []
        for child in elt:
            if child.tag in ('wf', 'punc'):
                itm = self.handle_word(child)
                if self._unit == 'word':
                    sent.extend(itm)
                else:
                    sent.append(itm)
            else:
                raise ValueError('Unexpected element %s' % child.tag)
        return SemcorSentence(elt.attrib['snum'], sent)