"""Various noun phrase extractors."""
from __future__ import unicode_literals, absolute_import
import nltk
from textblob.taggers import PatternTagger
from textblob.decorators import requires_nltk_corpus
from textblob.utils import tree2str, filter_insignificant
from textblob.base import BaseNPExtractor

class ChunkParser(nltk.ChunkParserI):

    def __init__(self):
        if False:
            return 10
        self._trained = False

    @requires_nltk_corpus
    def train(self):
        if False:
            i = 10
            return i + 15
        'Train the Chunker on the ConLL-2000 corpus.'
        train_data = [[(t, c) for (_, t, c) in nltk.chunk.tree2conlltags(sent)] for sent in nltk.corpus.conll2000.chunked_sents('train.txt', chunk_types=['NP'])]
        unigram_tagger = nltk.UnigramTagger(train_data)
        self.tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True

    def parse(self, sentence):
        if False:
            return 10
        'Return the parse tree for the sentence.'
        if not self._trained:
            self.train()
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.util.conlltags2tree(conlltags)

class ConllExtractor(BaseNPExtractor):
    """A noun phrase extractor that uses chunk parsing trained with the
    ConLL-2000 training corpus.
    """
    POS_TAGGER = PatternTagger()
    CFG = {('NNP', 'NNP'): 'NNP', ('NN', 'NN'): 'NNI', ('NNI', 'NN'): 'NNI', ('JJ', 'JJ'): 'JJ', ('JJ', 'NN'): 'NNI'}
    INSIGNIFICANT_SUFFIXES = ['DT', 'CC', 'PRP$', 'PRP']

    def __init__(self, parser=None):
        if False:
            i = 10
            return i + 15
        self.parser = ChunkParser() if not parser else parser

    def extract(self, text):
        if False:
            i = 10
            return i + 15
        'Return a list of noun phrases (strings) for body of text.'
        sentences = nltk.tokenize.sent_tokenize(text)
        noun_phrases = []
        for sentence in sentences:
            parsed = self._parse_sentence(sentence)
            phrases = [_normalize_tags(filter_insignificant(each, self.INSIGNIFICANT_SUFFIXES)) for each in parsed if isinstance(each, nltk.tree.Tree) and each.label() == 'NP' and (len(filter_insignificant(each)) >= 1) and _is_match(each, cfg=self.CFG)]
            nps = [tree2str(phrase) for phrase in phrases]
            noun_phrases.extend(nps)
        return noun_phrases

    def _parse_sentence(self, sentence):
        if False:
            for i in range(10):
                print('nop')
        'Tag and parse a sentence (a plain, untagged string).'
        tagged = self.POS_TAGGER.tag(sentence)
        return self.parser.parse(tagged)

class FastNPExtractor(BaseNPExtractor):
    """A fast and simple noun phrase extractor.

    Credit to Shlomi Babluk. Link to original blog post:

        http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/
    """
    CFG = {('NNP', 'NNP'): 'NNP', ('NN', 'NN'): 'NNI', ('NNI', 'NN'): 'NNI', ('JJ', 'JJ'): 'JJ', ('JJ', 'NN'): 'NNI'}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._trained = False

    @requires_nltk_corpus
    def train(self):
        if False:
            print('Hello World!')
        train_data = nltk.corpus.brown.tagged_sents(categories='news')
        regexp_tagger = nltk.RegexpTagger([('^-?[0-9]+(.[0-9]+)?$', 'CD'), ('(-|:|;)$', ':'), ("\\'*$", 'MD'), ('(The|the|A|a|An|an)$', 'AT'), ('.*able$', 'JJ'), ('^[A-Z].*$', 'NNP'), ('.*ness$', 'NN'), ('.*ly$', 'RB'), ('.*s$', 'NNS'), ('.*ing$', 'VBG'), ('.*ed$', 'VBD'), ('.*', 'NN')])
        unigram_tagger = nltk.UnigramTagger(train_data, backoff=regexp_tagger)
        self.tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True
        return None

    def _tokenize_sentence(self, sentence):
        if False:
            return 10
        'Split the sentence into single words/tokens'
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def extract(self, sentence):
        if False:
            i = 10
            return i + 15
        'Return a list of noun phrases (strings) for body of text.'
        if not self._trained:
            self.train()
        tokens = self._tokenize_sentence(sentence)
        tagged = self.tagger.tag(tokens)
        tags = _normalize_tags(tagged)
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = (t1[1], t2[1])
                value = self.CFG.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = '%s %s' % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break
        matches = [t[0] for t in tags if t[1] in ['NNP', 'NNI']]
        return matches

def _normalize_tags(chunk):
    if False:
        print('Hello World!')
    'Normalize the corpus tags.\n    ("NN", "NN-PL", "NNS") -> "NN"\n    '
    ret = []
    for (word, tag) in chunk:
        if tag == 'NP-TL' or tag == 'NP':
            ret.append((word, 'NNP'))
            continue
        if tag.endswith('-TL'):
            ret.append((word, tag[:-3]))
            continue
        if tag.endswith('S'):
            ret.append((word, tag[:-1]))
            continue
        ret.append((word, tag))
    return ret

def _is_match(tagged_phrase, cfg):
    if False:
        for i in range(10):
            print('nop')
    'Return whether or not a tagged phrases matches a context-free grammar.\n    '
    copy = list(tagged_phrase)
    merge = True
    while merge:
        merge = False
        for i in range(len(copy) - 1):
            (first, second) = (copy[i], copy[i + 1])
            key = (first[1], second[1])
            value = cfg.get(key, None)
            if value:
                merge = True
                copy.pop(i)
                copy.pop(i)
                match = '{0} {1}'.format(first[0], second[0])
                pos = value
                copy.insert(i, (match, pos))
                break
    match = any([t[1] in ('NNP', 'NNI') for t in copy])
    return match