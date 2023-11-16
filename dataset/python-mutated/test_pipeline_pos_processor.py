"""
Basic testing of part of speech tagging
"""
import pytest
import stanza
from stanza.tests import *
pytestmark = pytest.mark.pipeline
EN_DOC = 'Joe Smith was born in California.'
EN_DOC_GOLD = '\n<Token id=1;words=[<Word id=1;text=Joe;upos=PROPN;xpos=NNP;feats=Number=Sing>]>\n<Token id=2;words=[<Word id=2;text=Smith;upos=PROPN;xpos=NNP;feats=Number=Sing>]>\n<Token id=3;words=[<Word id=3;text=was;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin>]>\n<Token id=4;words=[<Word id=4;text=born;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass>]>\n<Token id=5;words=[<Word id=5;text=in;upos=ADP;xpos=IN>]>\n<Token id=6;words=[<Word id=6;text=California;upos=PROPN;xpos=NNP;feats=Number=Sing>]>\n<Token id=7;words=[<Word id=7;text=.;upos=PUNCT;xpos=.>]>\n'.strip()

@pytest.fixture(scope='module')
def pos_pipeline():
    if False:
        while True:
            i = 10
    return stanza.Pipeline(**{'processors': 'tokenize,pos', 'dir': TEST_MODELS_DIR, 'download_method': None, 'lang': 'en'})

def test_part_of_speech(pos_pipeline):
    if False:
        i = 10
        return i + 15
    doc = pos_pipeline(EN_DOC)
    assert EN_DOC_GOLD == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])

def test_get_known_xpos(pos_pipeline):
    if False:
        i = 10
        return i + 15
    tags = pos_pipeline.processors['pos'].get_known_xpos()
    assert 'DT' in tags
    assert 'DET' not in tags

def test_get_known_upos(pos_pipeline):
    if False:
        print('Hello World!')
    tags = pos_pipeline.processors['pos'].get_known_upos()
    assert 'DET' in tags
    assert 'DT' not in tags

def test_get_known_feats(pos_pipeline):
    if False:
        for i in range(10):
            print('nop')
    feats = pos_pipeline.processors['pos'].get_known_feats()
    assert 'Abbr' in feats
    assert 'Yes' in feats['Abbr']