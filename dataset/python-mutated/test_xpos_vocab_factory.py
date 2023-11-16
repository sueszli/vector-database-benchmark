"""
Test some pieces of the depparse dataloader
"""
import pytest
import logging
import os
import tempfile
from stanza.models import tagger
from stanza.models.common import pretrain
from stanza.models.pos.data import DataLoader
from stanza.models.pos.trainer import Trainer
from stanza.models.pos.vocab import WordVocab, XPOSVocab
from stanza.models.pos.xpos_vocab_factory import xpos_vocab_factory
from stanza.models.pos.xpos_vocab_utils import XPOSDescription, XPOSType, build_xpos_vocab, choose_simplest_factory
from stanza.utils.conll import CoNLL
from stanza.tests import TEST_WORKING_DIR
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]
logger = logging.getLogger('stanza.models.pos.xpos_vocab_factory')
EN_EXAMPLE = "\n1\tSh'reyan\tSh'reyan\tPROPN\tNNP%(tag)s\tNumber=Sing\t3\tnmod:poss\t3:nmod:poss\t_\n2\t's\t's\tPART\tPOS%(tag)s\t_\t1\tcase\t1:case\t_\n3\tantennae\tantenna\tNOUN%(tag)s\tNNS\tNumber=Plur\t6\tnsubj\t6:nsubj\t_\n4\tare\tbe\tVERB\tVBP%(tag)s\tMood=Ind|Tense=Pres|VerbForm=Fin\t6\tcop\t6:cop\t_\n5\thella\thella\tADV\tRB%(tag)s\t_\t6\tadvmod\t6:advmod\t_\n6\tthicc\tthicc\tADJ\tJJ%(tag)s\tDegree=Pos\t0\troot\t0:root\t_\n"
EMPTY_TAG = lambda x: ''
DASH_TAGS = lambda x: '-%d' % x

def build_doc(iterations, suffix):
    if False:
        return 10
    '\n    build N copies of the english text above, with a lambda function applied for the tag suffices\n\n    for example:\n      lambda x: "" means the suffices are all blank (NNP, POS, NNS, etc) for each iteration\n      lambda x: "-%d" % x means they go (NNP-0, NNP-1, NNP-2, etc) for the first word\'s tag\n    '
    texts = [EN_EXAMPLE % {'tag': suffix(i)} for i in range(iterations)]
    text = '\n\n'.join(texts)
    doc = CoNLL.conll2doc(input_str=text)
    return doc

def build_data(iterations, suffix):
    if False:
        print('Hello World!')
    '\n    Same thing, but passes the Doc through a POS Tagger DataLoader\n    '
    doc = build_doc(iterations, suffix)
    data = DataLoader.load_doc(doc)
    return data

class ErrorFatalHandler(logging.Handler):
    """
    This handler turns any error logs into a fatal error

    Theoretically you could change the level to make other things fatal as well
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.setLevel(logging.ERROR)

    def emit(self, record):
        if False:
            while True:
                i = 10
        raise AssertionError('Oh no, we printed an error')

class TestXPOSVocabFactory:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        "\n        Add a logger to the xpos factory logger so that it will throw an assertion instead of logging an error\n\n        We don't actually want assertions, since that would be a huge\n        pain in the event one of the models actually changes, so\n        instead we just logger.error in the factory.  Using this\n        handler is a simple way to check that the error is correctly\n        logged when something changes\n        "
        logger.info('About to start xpos_vocab_factory tests - logger.error in that module will now cause AssertionError')
        handler = ErrorFatalHandler()
        logger.addHandler(handler)

    @classmethod
    def teardown_class(cls):
        if False:
            i = 10
            return i + 15
        '\n        Remove the handler we installed earlier\n        '
        handlers = [x for x in logger.handlers if isinstance(x, ErrorFatalHandler)]
        for handler in handlers:
            logger.removeHandler(handler)
        logger.error('Done with xpos_vocab_factory tests - this should not throw an error')

    def test_basic_en_ewt(self):
        if False:
            i = 10
            return i + 15
        '\n        en_ewt is currently the basic vocab\n\n        note that this may change if the dataset is drastically relabeled in the future\n        '
        data = build_data(1, EMPTY_TAG)
        vocab = xpos_vocab_factory(data, 'en_ewt')
        assert isinstance(vocab, WordVocab)

    def test_basic_en_unknown(self):
        if False:
            print('Hello World!')
        '\n        With only 6 tags, it should use a basic vocab for an unknown dataset\n        '
        data = build_data(10, EMPTY_TAG)
        vocab = xpos_vocab_factory(data, 'en_unknown')
        assert isinstance(vocab, WordVocab)

    def test_dash_en_unknown(self):
        if False:
            i = 10
            return i + 15
        '\n        With this many different tags, it should choose to reduce it to the base xpos removing the -\n        '
        data = build_data(10, DASH_TAGS)
        vocab = xpos_vocab_factory(data, 'en_unknown')
        assert isinstance(vocab, XPOSVocab)
        assert vocab.sep == '-'

    def test_dash_en_ewt_wrong(self):
        if False:
            while True:
                i = 10
        '\n        The dataset looks like XPOS(-), which is wrong for en_ewt\n        '
        with pytest.raises(AssertionError):
            data = build_data(10, DASH_TAGS)
            vocab = xpos_vocab_factory(data, 'en_ewt')
            assert isinstance(vocab, XPOSVocab)
            assert vocab.sep == '-'

    def check_reload(self, pt, shorthand, iterations, suffix, expected_vocab):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a Trainer (no actual training), save it, and load it back in to check the type of Vocab restored\n\n        TODO: This test may be a bit "eager" in that there are no other\n        tests which check building, saving, & loading a pos trainer.\n        Could add tests to test_trainer.py, for example\n        '
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as tmpdirname:
            args = tagger.parse_args(['--batch_size', '1', '--shorthand', shorthand])
            train_doc = build_doc(iterations, suffix)
            train_batch = DataLoader(train_doc, args['batch_size'], args, pt, evaluation=False)
            vocab = train_batch.vocab
            assert isinstance(vocab['xpos'], expected_vocab)
            trainer = Trainer(args=args, vocab=vocab, pretrain=pt, device='cpu')
            model_file = os.path.join(tmpdirname, 'foo.pt')
            trainer.save(model_file)
            new_trainer = Trainer(model_file=model_file, pretrain=pt)
            assert isinstance(new_trainer.vocab['xpos'], expected_vocab)

    @pytest.fixture(scope='class')
    def pt(self):
        if False:
            while True:
                i = 10
        pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz', save_to_file=False)
        return pt

    def test_reload_word_vocab(self, pt):
        if False:
            print('Hello World!')
        '\n        Test that building a model with a known word vocab shorthand, saving it, and loading it gets back a word vocab\n        '
        self.check_reload(pt, 'en_ewt', 10, EMPTY_TAG, WordVocab)

    def test_reload_unknown_word_vocab(self, pt):
        if False:
            while True:
                i = 10
        '\n        Test that building a model with an unknown word vocab, saving it, and loading it gets back a word vocab\n        '
        self.check_reload(pt, 'en_unknown', 10, EMPTY_TAG, WordVocab)

    def test_reload_unknown_xpos_vocab(self, pt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that building a model with an unknown xpos vocab, saving it, and loading it gets back an xpos vocab\n        '
        self.check_reload(pt, 'en_unknown', 10, DASH_TAGS, XPOSVocab)