"""
Basic testing of French pipeline

The benefit of this test is to verify that the bulk processing works
for languages with MWT in them
"""
import pytest
import stanza
from stanza.models.common.doc import Document
from stanza.tests import *
from stanza.tests.pipeline.pipeline_device_tests import check_on_gpu, check_on_cpu
pytestmark = pytest.mark.pipeline
FR_MWT_SENTENCE = "Alors encore inconnu du grand public, Emmanuel Macron devient en 2014 ministre de l'Économie, de l'Industrie et du Numérique."
EXPECTED_RESULT = '\n[\n  [\n    {\n      "id": 1,\n      "text": "Alors",\n      "lemma": "alors",\n      "upos": "ADV",\n      "head": 3,\n      "deprel": "advmod",\n      "start_char": 0,\n      "end_char": 5\n    },\n    {\n      "id": 2,\n      "text": "encore",\n      "lemma": "encore",\n      "upos": "ADV",\n      "head": 3,\n      "deprel": "advmod",\n      "start_char": 6,\n      "end_char": 12\n    },\n    {\n      "id": 3,\n      "text": "inconnu",\n      "lemma": "inconnu",\n      "upos": "ADJ",\n      "feats": "Gender=Masc|Number=Sing",\n      "head": 11,\n      "deprel": "advcl",\n      "start_char": 13,\n      "end_char": 20\n    },\n    {\n      "id": [\n        4,\n        5\n      ],\n      "text": "du",\n      "start_char": 21,\n      "end_char": 23\n    },\n    {\n      "id": 4,\n      "text": "de",\n      "lemma": "de",\n      "upos": "ADP",\n      "head": 7,\n      "deprel": "case"\n    },\n    {\n      "id": 5,\n      "text": "le",\n      "lemma": "le",\n      "upos": "DET",\n      "feats": "Definite=Def|Gender=Masc|Number=Sing|PronType=Art",\n      "head": 7,\n      "deprel": "det"\n    },\n    {\n      "id": 6,\n      "text": "grand",\n      "lemma": "grand",\n      "upos": "ADJ",\n      "feats": "Gender=Masc|Number=Sing",\n      "head": 7,\n      "deprel": "amod",\n      "start_char": 24,\n      "end_char": 29\n    },\n    {\n      "id": 7,\n      "text": "public",\n      "lemma": "public",\n      "upos": "NOUN",\n      "feats": "Gender=Masc|Number=Sing",\n      "head": 3,\n      "deprel": "obl:mod",\n      "start_char": 30,\n      "end_char": 36\n    },\n    {\n      "id": 8,\n      "text": ",",\n      "lemma": ",",\n      "upos": "PUNCT",\n      "head": 3,\n      "deprel": "punct",\n      "start_char": 36,\n      "end_char": 37\n    },\n    {\n      "id": 9,\n      "text": "Emmanuel",\n      "lemma": "Emmanuel",\n      "upos": "PROPN",\n      "head": 11,\n      "deprel": "nsubj",\n      "start_char": 38,\n      "end_char": 46\n    },\n    {\n      "id": 10,\n      "text": "Macron",\n      "lemma": "Macron",\n      "upos": "PROPN",\n      "head": 9,\n      "deprel": "flat:name",\n      "start_char": 47,\n      "end_char": 53\n    },\n    {\n      "id": 11,\n      "text": "devient",\n      "lemma": "devenir",\n      "upos": "VERB",\n      "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",\n      "head": 0,\n      "deprel": "root",\n      "start_char": 54,\n      "end_char": 61\n    },\n    {\n      "id": 12,\n      "text": "en",\n      "lemma": "en",\n      "upos": "ADP",\n      "head": 13,\n      "deprel": "case",\n      "start_char": 62,\n      "end_char": 64\n    },\n    {\n      "id": 13,\n      "text": "2014",\n      "lemma": "2014",\n      "upos": "NUM",\n      "feats": "Number=Plur",\n      "head": 11,\n      "deprel": "obl:mod",\n      "start_char": 65,\n      "end_char": 69\n    },\n    {\n      "id": 14,\n      "text": "ministre",\n      "lemma": "ministre",\n      "upos": "NOUN",\n      "feats": "Gender=Masc|Number=Sing",\n      "head": 11,\n      "deprel": "xcomp",\n      "start_char": 70,\n      "end_char": 78\n    },\n    {\n      "id": 15,\n      "text": "de",\n      "lemma": "de",\n      "upos": "ADP",\n      "head": 17,\n      "deprel": "case",\n      "start_char": 79,\n      "end_char": 81\n    },\n    {\n      "id": 16,\n      "text": "l\'",\n      "lemma": "le",\n      "upos": "DET",\n      "feats": "Definite=Def|Number=Sing|PronType=Art",\n      "head": 17,\n      "deprel": "det",\n      "start_char": 82,\n      "end_char": 84\n    },\n    {\n      "id": 17,\n      "text": "Économie",\n      "lemma": "économie",\n      "upos": "NOUN",\n      "feats": "Gender=Fem|Number=Sing",\n      "head": 14,\n      "deprel": "nmod",\n      "start_char": 84,\n      "end_char": 92\n    },\n    {\n      "id": 18,\n      "text": ",",\n      "lemma": ",",\n      "upos": "PUNCT",\n      "head": 21,\n      "deprel": "punct",\n      "start_char": 92,\n      "end_char": 93\n    },\n    {\n      "id": 19,\n      "text": "de",\n      "lemma": "de",\n      "upos": "ADP",\n      "head": 21,\n      "deprel": "case",\n      "start_char": 94,\n      "end_char": 96\n    },\n    {\n      "id": 20,\n      "text": "l\'",\n      "lemma": "le",\n      "upos": "DET",\n      "feats": "Definite=Def|Number=Sing|PronType=Art",\n      "head": 21,\n      "deprel": "det",\n      "start_char": 97,\n      "end_char": 99\n    },\n    {\n      "id": 21,\n      "text": "Industrie",\n      "lemma": "industrie",\n      "upos": "NOUN",\n      "feats": "Gender=Fem|Number=Sing",\n      "head": 17,\n      "deprel": "conj",\n      "start_char": 99,\n      "end_char": 108\n    },\n    {\n      "id": 22,\n      "text": "et",\n      "lemma": "et",\n      "upos": "CCONJ",\n      "head": 25,\n      "deprel": "cc",\n      "start_char": 109,\n      "end_char": 111\n    },\n    {\n      "id": [\n        23,\n        24\n      ],\n      "text": "du",\n      "start_char": 112,\n      "end_char": 114\n    },\n    {\n      "id": 23,\n      "text": "de",\n      "lemma": "de",\n      "upos": "ADP",\n      "head": 25,\n      "deprel": "case"\n    },\n    {\n      "id": 24,\n      "text": "le",\n      "lemma": "le",\n      "upos": "DET",\n      "feats": "Definite=Def|Gender=Masc|Number=Sing|PronType=Art",\n      "head": 25,\n      "deprel": "det"\n    },\n    {\n      "id": 25,\n      "text": "Numérique",\n      "lemma": "numérique",\n      "upos": "PROPN",\n      "feats": "Gender=Masc|Number=Sing",\n      "head": 17,\n      "deprel": "conj",\n      "start_char": 115,\n      "end_char": 124\n    },\n    {\n      "id": 26,\n      "text": ".",\n      "lemma": ".",\n      "upos": "PUNCT",\n      "head": 11,\n      "deprel": "punct",\n      "start_char": 124,\n      "end_char": 125\n    }\n  ]\n]\n'

class TestFrenchPipeline:

    @pytest.fixture(scope='class')
    def pipeline(self):
        if False:
            print('Hello World!')
        ' Create a pipeline with French models '
        pipeline = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', dir=TEST_MODELS_DIR, lang='fr')
        return pipeline

    def test_single(self, pipeline):
        if False:
            print('Hello World!')
        doc = pipeline(FR_MWT_SENTENCE)
        compare_ignoring_whitespace(str(doc), EXPECTED_RESULT)

    def test_bulk(self, pipeline):
        if False:
            return 10
        NUM_DOCS = 10
        raw_text = [FR_MWT_SENTENCE] * NUM_DOCS
        raw_doc = [Document([], text=doccontent) for doccontent in raw_text]
        result = pipeline(raw_doc)
        assert len(result) == NUM_DOCS
        for doc in result:
            compare_ignoring_whitespace(str(doc), EXPECTED_RESULT)
            assert len(doc.sentences) == 1
            assert doc.num_words == 26
            assert doc.num_tokens == 24

    def test_on_gpu(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        '\n        The default pipeline should have all the models on the GPU\n        '
        check_on_gpu(pipeline)

    def test_on_cpu(self):
        if False:
            return 10
        '\n        Create a pipeline on the CPU, check that all the models on CPU\n        '
        pipeline = stanza.Pipeline('fr', dir=TEST_MODELS_DIR, use_gpu=False)
        check_on_cpu(pipeline)