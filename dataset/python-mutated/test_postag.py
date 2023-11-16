"""
Tests for the part-of-speech tagging visualization
"""
import pytest
import matplotlib.pyplot as plt
from tests.base import VisualTestCase
from tests.base import IS_WINDOWS_OR_CONDA
from yellowbrick.text.postag import *
from yellowbrick.exceptions import YellowbrickValueError
try:
    import nltk
    from nltk import pos_tag, sent_tokenize
    from nltk import word_tokenize, wordpunct_tokenize
except ImportError:
    nltk = None
try:
    import spacy
except ImportError:
    spacy = None
sonnets = ["\n    FROM fairest creatures we desire increase,\n    That thereby beauty's rose might never die,\n    But as the riper should by time decease,\n    His tender heir might bear his memory:\n    But thou, contracted to thine own bright eyes,\n    Feed'st thy light'st flame with self-substantial fuel,\n    Making a famine where abundance lies,\n    Thyself thy foe, to thy sweet self too cruel.\n    Thou that art now the world's fresh ornament\n    And only herald to the gaudy spring,\n    Within thine own bud buriest thy content\n    And, tender churl, makest waste in niggarding.\n    Pity the world, or else this glutton be,\n    To eat the world's due, by the grave and thee.\n    ", "\n    When forty winters shall beseige thy brow,\n    And dig deep trenches in thy beauty's field,\n    Thy youth's proud livery, so gazed on now,\n    Will be a tatter'd weed, of small worth held:\n    Then being ask'd where all thy beauty lies,\n    Where all the treasure of thy lusty days,\n    To say, within thine own deep-sunken eyes,\n    Were an all-eating shame and thriftless praise.\n    How much more praise deserved thy beauty's use,\n    If thou couldst answer 'This fair child of mine\n    Shall sum my count and make my old excuse,'\n    Proving his beauty by succession thine!\n    This were to be new made when thou art old,\n    And see thy blood warm when thou feel'st it cold.\n    ", "\n    Look in thy glass, and tell the face thou viewest\n    Now is the time that face should form another;\n    Whose fresh repair if now thou not renewest,\n    Thou dost beguile the world, unbless some mother.\n    For where is she so fair whose unear'd womb\n    Disdains the tillage of thy husbandry?\n    Or who is he so fond will be the tomb\n    Of his self-love, to stop posterity?\n    Thou art thy mother's glass, and she in thee\n    Calls back the lovely April of her prime:\n    So thou through windows of thine age shall see\n    Despite of wrinkles this thy golden time.\n    But if thou live, remember'd not to be,\n    Die single, and thine image dies with thee.\n    "]

def check_nltk_data():
    if False:
        print('Hello World!')
    '\n    Returns True if NLTK data has been downloaded, False otherwise\n    '
    try:
        nltk.data.find('corpora/treebank')
        return True
    except LookupError:
        pytest.xfail('error occured because nltk postag data is not available')

def check_spacy_data():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if SpaCy data has been downloaded, False otherwise\n    '
    try:
        spacy.load('en_core_web_sm')
        return True
    except OSError:
        pytest.xfail('error occured because spacy data model is not available')

def get_tagged_docs(X, model='nltk', tagger='word'):
    if False:
        return 10
    '\n    X is a list of strings; each string is a single document.\n    For each document, perform part-of-speech tagging, and\n    yield a list of sentences, where each sentence is a list\n    of (token, tag) tuples\n\n    If model=="nltk", `NLTK` will be used to sentence and word\n    tokenize the incoming documents. User may select the `NLTK`\n    tagger to be used; (for now) either the word tokenizer or the\n    workpunct tokenizer.\n\n    If model=="spacy", `SpaCy` will be used to sentence and word\n    tokenize the incoming documents.\n    '
    if model == 'spacy':
        nlp = spacy.load('en_core_web_sm')
        for doc in X:
            tagged = nlp(doc)
            yield [list(((token.text, token.pos_) for token in sent)) for sent in tagged.sents]
    elif model == 'nltk':
        if tagger == 'wordpunct':
            for doc in X:
                yield [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(doc)]
        else:
            for doc in X:
                yield [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(doc)]

class TestPosTag(VisualTestCase):
    """
    PosTag (Part of Speech Tagging Visualizer) Tests
    """

    def test_quick_method(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert no errors occur when using the quick method\n        '
        check_nltk_data()
        (_, ax) = plt.subplots()
        tagged_docs = list(get_tagged_docs(sonnets))
        viz = postag(tagged_docs, ax=ax, show=False)
        viz.ax.grid(False)
        tol = 5.5 if IS_WINDOWS_OR_CONDA else 0.25
        self.assert_images_similar(viz, tol=tol)

    def test_unknown_tagset(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure an exception is raised if the specified tagset is unknown\n        '
        with pytest.raises(YellowbrickValueError):
            PosTagVisualizer(tagset='brill')

    def test_frequency_mode(self):
        if False:
            print('Hello World!')
        '\n        Assert no errors occur when the visualizer is run on frequency mode\n        '
        check_nltk_data()
        (_, ax) = plt.subplots()
        tagged_docs = list(get_tagged_docs(sonnets))
        viz = PosTagVisualizer(ax=ax, frequency=True)
        viz.fit(tagged_docs)
        viz.finalize()
        ax.grid(False)
        sorted_tags = ['noun', 'adjective', 'punctuation', 'verb', 'preposition', 'determiner', 'adverb', 'conjunction', 'pronoun', 'wh- word', 'modal', 'infinitive', 'possessive', 'other', 'symbol', 'existential', 'digit', 'non-English', 'interjection', 'list']
        ticks_ax = [tick.get_text() for tick in ax.xaxis.get_ticklabels()]
        assert ticks_ax == sorted_tags
        tol = 5.5 if IS_WINDOWS_OR_CONDA else 0.5
        self.assert_images_similar(ax=ax, tol=tol)

    @pytest.mark.skipif(nltk is None, reason='test requires nltk')
    def test_word_tagged(self):
        if False:
            print('Hello World!')
        '\n        Assert no errors occur during PosTagVisualizer integration\n        with word tokenized corpus\n        '
        check_nltk_data()
        tagged_docs = list(get_tagged_docs(sonnets, model='nltk', tagger='word'))
        visualizer = PosTagVisualizer(tagset='penn_treebank')
        visualizer.fit(tagged_docs)
        visualizer.ax.grid(False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(nltk is None, reason='test requires nltk')
    def test_wordpunct_tagged(self):
        if False:
            print('Hello World!')
        '\n        Assert no errors occur during PosTagVisualizer integration\n        with wordpunct tokenized corpus\n        '
        check_nltk_data()
        wordpunct_tagged_docs = list(get_tagged_docs(sonnets, model='nltk', tagger='wordpunct'))
        visualizer = PosTagVisualizer(tagset='penn_treebank')
        visualizer.fit(wordpunct_tagged_docs)
        visualizer.ax.grid(False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(spacy is None, reason='test requires spacy')
    def test_spacy_tagged(self):
        if False:
            print('Hello World!')
        '\n        Assert no errors occur during PosTagVisualizer integration\n        with spacy tokenized corpus\n        '
        check_spacy_data()
        spacy_tagged_docs = list(get_tagged_docs(sonnets, model='spacy'))
        visualizer = PosTagVisualizer(tagset='universal')
        visualizer.fit(spacy_tagged_docs)
        visualizer.ax.grid(False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(spacy is None, reason='test requires spacy')
    def test_spacy_raw(self):
        if False:
            print('Hello World!')
        '\n        Assert no errors occur during PosTagVisualizer integration\n        with raw corpus to be parsed using spacy\n        '
        visualizer = PosTagVisualizer(parser='spacy', tagset='universal')
        visualizer.fit(sonnets)
        visualizer.ax.grid(False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(nltk is None, reason='test requires nltk')
    def test_nltk_word_raw(self):
        if False:
            return 10
        '\n        Assert no errors occur during PosTagVisualizer integration\n        with raw corpus to be parsed using nltk\n        '
        visualizer = PosTagVisualizer(parser='nltk', tagset='penn_treebank')
        visualizer.fit(sonnets)
        visualizer.ax.grid(False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(nltk is None, reason='test requires nltk')
    def test_nltk_wordpunct_raw(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert no errors occur during PosTagVisualizer integration\n        with raw corpus to be parsed using nltk\n        '
        visualizer = PosTagVisualizer(parser='nltk_wordpunct', tagset='penn_treebank')
        visualizer.fit(sonnets)
        visualizer.ax.grid(False)
        self.assert_images_similar(visualizer)

    def test_stack_mode(self):
        if False:
            print('Hello World!')
        '\n        Assert no errors occur when the visualizer is run on stack mode\n        '
        check_nltk_data()
        (_, ax) = plt.subplots()
        tagged_docs = list(get_tagged_docs(sonnets))
        visualizer = PosTagVisualizer(stack=True, ax=ax)
        visualizer.fit(tagged_docs, y=['a', 'b', 'c'])
        visualizer.ax.grid(False)
        self.assert_images_similar(ax=ax)

    def test_stack_frequency_mode(self):
        if False:
            i = 10
            return i + 15
        '\n        Assert no errors occur when the visualizer is run on both stack and\n        frequency mode\n        '
        check_nltk_data()
        (_, ax) = plt.subplots()
        tagged_docs = list(get_tagged_docs(sonnets))
        visualizer = PosTagVisualizer(stack=True, frequency=True, ax=ax)
        visualizer.fit(tagged_docs, y=['a', 'b', 'c'])
        visualizer.ax.grid(False)
        sorted_tags = ['noun', 'adjective', 'punctuation', 'verb', 'preposition', 'determiner', 'adverb', 'conjunction', 'pronoun', 'wh- word', 'modal', 'infinitive', 'possessive', 'other', 'symbol', 'existential', 'digit', 'non-English', 'interjection', 'list']
        ticks_ax = [tick.get_text() for tick in ax.xaxis.get_ticklabels()]
        assert ticks_ax == sorted_tags
        self.assert_images_similar(ax=ax)