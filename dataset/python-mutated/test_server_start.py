"""
Tests for starting a server in Python code
"""
import pytest
import stanza.server as corenlp
from stanza.server.client import AnnotationException
import time
from stanza.tests import *
pytestmark = pytest.mark.client
EN_DOC = 'Joe Smith lives in California.'
EN_PRELOAD_GOLD = '\nSentence #1 (6 tokens):\nJoe Smith lives in California.\n\nTokens:\n[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP Lemma=Joe NamedEntityTag=PERSON]\n[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP Lemma=Smith NamedEntityTag=PERSON]\n[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ Lemma=live NamedEntityTag=O]\n[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN Lemma=in NamedEntityTag=O]\n[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP Lemma=California NamedEntityTag=STATE_OR_PROVINCE]\n[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=. Lemma=. NamedEntityTag=O]\n\nDependency Parse (enhanced plus plus dependencies):\nroot(ROOT-0, lives-3)\ncompound(Smith-2, Joe-1)\nnsubj(lives-3, Smith-2)\ncase(California-5, in-4)\nobl:in(lives-3, California-5)\npunct(lives-3, .-6)\n\nExtracted the following NER entity mentions:\nJoe Smith       PERSON              PERSON:0.9972202681743931\nCalifornia      STATE_OR_PROVINCE   LOCATION:0.9990868267559281\n\nExtracted the following KBP triples:\n1.0 Joe Smith per:statesorprovinces_of_residence California\n'
EN_PROPS_FILE_GOLD = '\nSentence #1 (6 tokens):\nJoe Smith lives in California.\n\nTokens:\n[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP]\n[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP]\n[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ]\n[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN]\n[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP]\n[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=.]\n'
GERMAN_DOC = 'Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.'
GERMAN_FULL_PROPS_GOLD = '\nSentence #1 (10 tokens):\nAngela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.\n\nTokens:\n[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=PROPN Lemma=angela NamedEntityTag=PERSON]\n[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=PROPN Lemma=merkel NamedEntityTag=PERSON]\n[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17 PartOfSpeech=AUX Lemma=ist NamedEntityTag=O]\n[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22 PartOfSpeech=ADP Lemma=seit NamedEntityTag=O]\n[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27 PartOfSpeech=NUM Lemma=2005 NamedEntityTag=O]\n[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43 PartOfSpeech=NOUN Lemma=bundeskanzlerin NamedEntityTag=O]\n[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47 PartOfSpeech=DET Lemma=der NamedEntityTag=O]\n[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62 PartOfSpeech=PROPN Lemma=bundesrepublik NamedEntityTag=LOCATION]\n[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74 PartOfSpeech=PROPN Lemma=deutschland NamedEntityTag=LOCATION]\n[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75 PartOfSpeech=PUNCT Lemma=. NamedEntityTag=O]\n\nDependency Parse (enhanced plus plus dependencies):\nroot(ROOT-0, Bundeskanzlerin-6)\nnsubj(Bundeskanzlerin-6, Angela-1)\nflat(Angela-1, Merkel-2)\ncop(Bundeskanzlerin-6, ist-3)\ncase(2005-5, seit-4)\nnmod:seit(Bundeskanzlerin-6, 2005-5)\ndet(Bundesrepublik-8, der-7)\nnmod(Bundeskanzlerin-6, Bundesrepublik-8)\nappos(Bundesrepublik-8, Deutschland-9)\npunct(Bundeskanzlerin-6, .-10)\n\nExtracted the following NER entity mentions:\nAngela Merkel              PERSON   PERSON:0.9999981583351504\nBundesrepublik Deutschland LOCATION LOCATION:0.9682902289749544\n'
GERMAN_SMALL_PROPS = {'annotators': 'tokenize,ssplit,pos', 'tokenize.language': 'de', 'pos.model': 'edu/stanford/nlp/models/pos-tagger/german-ud.tagger'}
GERMAN_SMALL_PROPS_GOLD = '\nSentence #1 (10 tokens):\nAngela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.\n\nTokens:\n[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=PROPN]\n[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=PROPN]\n[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17 PartOfSpeech=AUX]\n[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22 PartOfSpeech=ADP]\n[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27 PartOfSpeech=NUM]\n[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43 PartOfSpeech=NOUN]\n[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47 PartOfSpeech=DET]\n[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62 PartOfSpeech=PROPN]\n[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74 PartOfSpeech=PROPN]\n[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75 PartOfSpeech=PUNCT]\n'
GERMAN_SMALL_PROPS_W_ANNOTATORS_GOLD = '\nSentence #1 (10 tokens):\nAngela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.\n\nTokens:\n[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6]\n[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13]\n[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17]\n[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22]\n[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27]\n[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43]\n[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47]\n[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62]\n[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74]\n[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75]\n'
USERNAME_PASS_PROPS = {'annotators': 'tokenize,ssplit,pos'}
USERNAME_PASS_GOLD = '\nSentence #1 (6 tokens):\nJoe Smith lives in California.\n\nTokens:\n[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP]\n[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP]\n[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ]\n[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN]\n[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP]\n[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=.]\n'

def annotate_and_time(client, text, properties={}):
    if False:
        return 10
    ' Submit an annotation request and return how long it took '
    start = time.time()
    ann = client.annotate(text, properties=properties, output_format='text')
    end = time.time()
    return {'annotation': ann, 'start_time': start, 'end_time': end}

def test_preload():
    if False:
        while True:
            i = 10
    ' Test that the default annotators load fully immediately upon server start '
    with corenlp.CoreNLPClient(server_id='test_server_start_preload') as client:
        time.sleep(140)
        results = annotate_and_time(client, EN_DOC)
        compare_ignoring_whitespace(results['annotation'], EN_PRELOAD_GOLD)
        assert results['end_time'] - results['start_time'] < 3

def test_props_file():
    if False:
        i = 10
        return i + 15
    ' Test starting the server with a props file '
    with corenlp.CoreNLPClient(properties=SERVER_TEST_PROPS, server_id='test_server_start_props_file') as client:
        ann = client.annotate(EN_DOC, output_format='text')
        assert ann.strip() == EN_PROPS_FILE_GOLD.strip()

def test_lang_start():
    if False:
        print('Hello World!')
    ' Test starting the server with a Stanford CoreNLP language name '
    with corenlp.CoreNLPClient(properties='german', server_id='test_server_start_lang_name') as client:
        ann = client.annotate(GERMAN_DOC, output_format='text')
        compare_ignoring_whitespace(ann, GERMAN_FULL_PROPS_GOLD)

def test_python_dict():
    if False:
        for i in range(10):
            print('nop')
    ' Test starting the server with a Python dictionary as default properties '
    with corenlp.CoreNLPClient(properties=GERMAN_SMALL_PROPS, server_id='test_server_start_python_dict') as client:
        ann = client.annotate(GERMAN_DOC, output_format='text')
        assert ann.strip() == GERMAN_SMALL_PROPS_GOLD.strip()

def test_python_dict_w_annotators():
    if False:
        for i in range(10):
            print('nop')
    ' Test starting the server with a Python dictionary as default properties, override annotators '
    with corenlp.CoreNLPClient(properties=GERMAN_SMALL_PROPS, annotators='tokenize,ssplit', server_id='test_server_start_python_dict_w_annotators') as client:
        ann = client.annotate(GERMAN_DOC, output_format='text')
        assert ann.strip() == GERMAN_SMALL_PROPS_W_ANNOTATORS_GOLD.strip()

def test_username_password():
    if False:
        i = 10
        return i + 15
    ' Test starting a server with a username and password '
    with corenlp.CoreNLPClient(properties=USERNAME_PASS_PROPS, username='user-1234', password='1234', server_id='test_server_username_pass') as client:
        ann = client.annotate(EN_DOC, output_format='text', username='user-1234', password='1234')
        assert ann.strip() == USERNAME_PASS_GOLD.strip()
        try:
            ann = client.annotate(EN_DOC, output_format='text', username='user-1234', password='12345')
            assert False
        except AnnotationException as ae:
            pass
        except Exception as e:
            assert False