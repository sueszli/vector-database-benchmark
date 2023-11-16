"""
Tests for setting request properties of servers
"""
import json
import pytest
import stanza.server as corenlp
from stanza.protobuf import Document
from stanza.tests import TEST_WORKING_DIR, compare_ignoring_whitespace
pytestmark = pytest.mark.client
EN_DOC = 'Joe Smith lives in California.'
EN_DOC_GOLD = '\nSentence #1 (6 tokens):\nJoe Smith lives in California.\n\nTokens:\n[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP]\n[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP]\n[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ]\n[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN]\n[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP]\n[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=.]\n'
GERMAN_DOC = 'Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.'
GERMAN_DOC_GOLD = '\nSentence #1 (10 tokens):\nAngela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.\n\nTokens:\n[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=PROPN]\n[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=PROPN]\n[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17 PartOfSpeech=AUX]\n[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22 PartOfSpeech=ADP]\n[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27 PartOfSpeech=NUM]\n[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43 PartOfSpeech=NOUN]\n[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47 PartOfSpeech=DET]\n[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62 PartOfSpeech=PROPN]\n[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74 PartOfSpeech=PROPN]\n[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75 PartOfSpeech=PUNCT]\n'
FRENCH_CUSTOM_PROPS = {'annotators': 'tokenize,ssplit,mwt,pos,parse', 'tokenize.language': 'fr', 'pos.model': 'edu/stanford/nlp/models/pos-tagger/french-ud.tagger', 'parse.model': 'edu/stanford/nlp/models/srparser/frenchSR.ser.gz', 'mwt.mappingFile': 'edu/stanford/nlp/models/mwt/french/french-mwt.tsv', 'mwt.pos.model': 'edu/stanford/nlp/models/mwt/french/french-mwt.tagger', 'mwt.statisticalMappingFile': 'edu/stanford/nlp/models/mwt/french/french-mwt-statistical.tsv', 'mwt.preserveCasing': 'false', 'outputFormat': 'text'}
FRENCH_EXTRA_PROPS = {'annotators': 'tokenize,ssplit,mwt,pos,depparse', 'tokenize.language': 'fr', 'pos.model': 'edu/stanford/nlp/models/pos-tagger/french-ud.tagger', 'mwt.mappingFile': 'edu/stanford/nlp/models/mwt/french/french-mwt.tsv', 'mwt.pos.model': 'edu/stanford/nlp/models/mwt/french/french-mwt.tagger', 'mwt.statisticalMappingFile': 'edu/stanford/nlp/models/mwt/french/french-mwt-statistical.tsv', 'mwt.preserveCasing': 'false', 'depparse.model': 'edu/stanford/nlp/models/parser/nndep/UD_French.gz'}
FRENCH_DOC = 'Cette enquête préliminaire fait suite aux révélations de l’hebdomadaire quelques jours plus tôt.'
FRENCH_CUSTOM_GOLD = '\nSentence #1 (16 tokens):\nCette enquête préliminaire fait suite aux révélations de l’hebdomadaire quelques jours plus tôt.\n\nTokens:\n[Text=Cette CharacterOffsetBegin=0 CharacterOffsetEnd=5 PartOfSpeech=DET]\n[Text=enquête CharacterOffsetBegin=6 CharacterOffsetEnd=13 PartOfSpeech=NOUN]\n[Text=préliminaire CharacterOffsetBegin=14 CharacterOffsetEnd=26 PartOfSpeech=ADJ]\n[Text=fait CharacterOffsetBegin=27 CharacterOffsetEnd=31 PartOfSpeech=VERB]\n[Text=suite CharacterOffsetBegin=32 CharacterOffsetEnd=37 PartOfSpeech=NOUN]\n[Text=à CharacterOffsetBegin=38 CharacterOffsetEnd=41 PartOfSpeech=ADP]\n[Text=les CharacterOffsetBegin=38 CharacterOffsetEnd=41 PartOfSpeech=DET]\n[Text=révélations CharacterOffsetBegin=42 CharacterOffsetEnd=53 PartOfSpeech=NOUN]\n[Text=de CharacterOffsetBegin=54 CharacterOffsetEnd=56 PartOfSpeech=ADP]\n[Text=l’ CharacterOffsetBegin=57 CharacterOffsetEnd=59 PartOfSpeech=NOUN]\n[Text=hebdomadaire CharacterOffsetBegin=59 CharacterOffsetEnd=71 PartOfSpeech=ADJ]\n[Text=quelques CharacterOffsetBegin=72 CharacterOffsetEnd=80 PartOfSpeech=DET]\n[Text=jours CharacterOffsetBegin=81 CharacterOffsetEnd=86 PartOfSpeech=NOUN]\n[Text=plus CharacterOffsetBegin=87 CharacterOffsetEnd=91 PartOfSpeech=ADV]\n[Text=tôt CharacterOffsetBegin=92 CharacterOffsetEnd=95 PartOfSpeech=ADV]\n[Text=. CharacterOffsetBegin=95 CharacterOffsetEnd=96 PartOfSpeech=PUNCT]\n\nConstituency parse: \n(ROOT\n  (SENT\n    (NP (DET Cette)\n      (MWN (NOUN enquête) (ADJ préliminaire)))\n    (VN\n      (MWV (VERB fait) (NOUN suite)))\n    (PP (ADP à)\n      (NP (DET les) (NOUN révélations)\n        (PP (ADP de)\n          (NP (NOUN l’)\n            (AP (ADJ hebdomadaire))))))\n    (NP (DET quelques) (NOUN jours))\n    (AdP (ADV plus) (ADV tôt))\n    (PUNCT .)))\n'
FRENCH_EXTRA_GOLD = '\nSentence #1 (16 tokens):\nCette enquête préliminaire fait suite aux révélations de l’hebdomadaire quelques jours plus tôt.\n\nTokens:\n[Text=Cette CharacterOffsetBegin=0 CharacterOffsetEnd=5 PartOfSpeech=DET]\n[Text=enquête CharacterOffsetBegin=6 CharacterOffsetEnd=13 PartOfSpeech=NOUN]\n[Text=préliminaire CharacterOffsetBegin=14 CharacterOffsetEnd=26 PartOfSpeech=ADJ]\n[Text=fait CharacterOffsetBegin=27 CharacterOffsetEnd=31 PartOfSpeech=VERB]\n[Text=suite CharacterOffsetBegin=32 CharacterOffsetEnd=37 PartOfSpeech=NOUN]\n[Text=à CharacterOffsetBegin=38 CharacterOffsetEnd=41 PartOfSpeech=ADP]\n[Text=les CharacterOffsetBegin=38 CharacterOffsetEnd=41 PartOfSpeech=DET]\n[Text=révélations CharacterOffsetBegin=42 CharacterOffsetEnd=53 PartOfSpeech=NOUN]\n[Text=de CharacterOffsetBegin=54 CharacterOffsetEnd=56 PartOfSpeech=ADP]\n[Text=l’ CharacterOffsetBegin=57 CharacterOffsetEnd=59 PartOfSpeech=NOUN]\n[Text=hebdomadaire CharacterOffsetBegin=59 CharacterOffsetEnd=71 PartOfSpeech=ADJ]\n[Text=quelques CharacterOffsetBegin=72 CharacterOffsetEnd=80 PartOfSpeech=DET]\n[Text=jours CharacterOffsetBegin=81 CharacterOffsetEnd=86 PartOfSpeech=NOUN]\n[Text=plus CharacterOffsetBegin=87 CharacterOffsetEnd=91 PartOfSpeech=ADV]\n[Text=tôt CharacterOffsetBegin=92 CharacterOffsetEnd=95 PartOfSpeech=ADV]\n[Text=. CharacterOffsetBegin=95 CharacterOffsetEnd=96 PartOfSpeech=PUNCT]\n\nDependency Parse (enhanced plus plus dependencies):\nroot(ROOT-0, fait-4)\ndet(enquête-2, Cette-1)\nnsubj(fait-4, enquête-2)\namod(enquête-2, préliminaire-3)\nobj(fait-4, suite-5)\ncase(révélations-8, à-6)\ndet(révélations-8, les-7)\nobl:à(fait-4, révélations-8)\ncase(l’-10, de-9)\nnmod:de(révélations-8, l’-10)\namod(révélations-8, hebdomadaire-11)\ndet(jours-13, quelques-12)\nobl(fait-4, jours-13)\nadvmod(tôt-15, plus-14)\nadvmod(jours-13, tôt-15)\npunct(fait-4, .-16)\n'
FRENCH_JSON_GOLD = json.loads(open(f'{TEST_WORKING_DIR}/out/example_french.json', encoding='utf-8').read())
ES_DOC = 'Andrés Manuel López Obrador es el presidente de México.'
ES_PROPS = {'annotators': 'tokenize,ssplit,mwt,pos,depparse', 'tokenize.language': 'es', 'pos.model': 'edu/stanford/nlp/models/pos-tagger/spanish-ud.tagger', 'mwt.mappingFile': 'edu/stanford/nlp/models/mwt/spanish/spanish-mwt.tsv', 'depparse.model': 'edu/stanford/nlp/models/parser/nndep/UD_Spanish.gz'}
ES_PROPS_GOLD = '\nSentence #1 (10 tokens):\nAndrés Manuel López Obrador es el presidente de México.\n\nTokens:\n[Text=Andrés CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=PROPN]\n[Text=Manuel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=PROPN]\n[Text=López CharacterOffsetBegin=14 CharacterOffsetEnd=19 PartOfSpeech=PROPN]\n[Text=Obrador CharacterOffsetBegin=20 CharacterOffsetEnd=27 PartOfSpeech=PROPN]\n[Text=es CharacterOffsetBegin=28 CharacterOffsetEnd=30 PartOfSpeech=AUX]\n[Text=el CharacterOffsetBegin=31 CharacterOffsetEnd=33 PartOfSpeech=DET]\n[Text=presidente CharacterOffsetBegin=34 CharacterOffsetEnd=44 PartOfSpeech=NOUN]\n[Text=de CharacterOffsetBegin=45 CharacterOffsetEnd=47 PartOfSpeech=ADP]\n[Text=México CharacterOffsetBegin=48 CharacterOffsetEnd=54 PartOfSpeech=PROPN]\n[Text=. CharacterOffsetBegin=54 CharacterOffsetEnd=55 PartOfSpeech=PUNCT]\n\nDependency Parse (enhanced plus plus dependencies):\nroot(ROOT-0, presidente-7)\nnsubj(presidente-7, Andrés-1)\nflat(Andrés-1, Manuel-2)\nflat(Andrés-1, López-3)\nflat(Andrés-1, Obrador-4)\ncop(presidente-7, es-5)\ndet(presidente-7, el-6)\ncase(México-9, de-8)\nnmod:de(presidente-7, México-9)\npunct(presidente-7, .-10)\n'

class TestServerRequest:

    @pytest.fixture(scope='class')
    def corenlp_client(self):
        if False:
            return 10
        ' Client to run tests on '
        client = corenlp.CoreNLPClient(annotators='tokenize,ssplit,pos', server_id='stanza_request_tests_server')
        yield client
        client.stop()

    def test_basic(self, corenlp_client):
        if False:
            for i in range(10):
                print('nop')
        ' Basic test of making a request, test default output format is a Document '
        ann = corenlp_client.annotate(EN_DOC, output_format='text')
        compare_ignoring_whitespace(ann, EN_DOC_GOLD)
        ann = corenlp_client.annotate(EN_DOC)
        assert isinstance(ann, Document)

    def test_python_dict(self, corenlp_client):
        if False:
            for i in range(10):
                print('nop')
        ' Test using a Python dictionary to specify all request properties '
        ann = corenlp_client.annotate(ES_DOC, properties=ES_PROPS, output_format='text')
        compare_ignoring_whitespace(ann, ES_PROPS_GOLD)
        ann = corenlp_client.annotate(FRENCH_DOC, properties=FRENCH_CUSTOM_PROPS)
        compare_ignoring_whitespace(ann, FRENCH_CUSTOM_GOLD)

    def test_lang_setting(self, corenlp_client):
        if False:
            return 10
        ' Test using a Stanford CoreNLP supported languages as a properties key '
        ann = corenlp_client.annotate(GERMAN_DOC, properties='german', output_format='text')
        compare_ignoring_whitespace(ann, GERMAN_DOC_GOLD)

    def test_annotators_and_output_format(self, corenlp_client):
        if False:
            for i in range(10):
                print('nop')
        ' Test setting the annotators and output_format '
        ann = corenlp_client.annotate(FRENCH_DOC, properties=FRENCH_EXTRA_PROPS, annotators='tokenize,ssplit,mwt,pos', output_format='json')
        assert ann == FRENCH_JSON_GOLD