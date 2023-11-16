"""
Misc tests for the server
"""
import pytest
import re
import stanza.server as corenlp
from stanza.tests import compare_ignoring_whitespace
pytestmark = pytest.mark.client
EN_DOC = 'Joe Smith lives in California.'
EN_DOC_GOLD = '\nSentence #1 (6 tokens):\nJoe Smith lives in California.\n\nTokens:\n[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP Lemma=Joe NamedEntityTag=PERSON]\n[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP Lemma=Smith NamedEntityTag=PERSON]\n[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ Lemma=live NamedEntityTag=O]\n[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN Lemma=in NamedEntityTag=O]\n[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP Lemma=California NamedEntityTag=STATE_OR_PROVINCE]\n[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=. Lemma=. NamedEntityTag=O]\n\nDependency Parse (enhanced plus plus dependencies):\nroot(ROOT-0, lives-3)\ncompound(Smith-2, Joe-1)\nnsubj(lives-3, Smith-2)\ncase(California-5, in-4)\nobl:in(lives-3, California-5)\npunct(lives-3, .-6)\n\nExtracted the following NER entity mentions:\nJoe Smith       PERSON  PERSON:0.9972202681743931\nCalifornia      STATE_OR_PROVINCE       LOCATION:0.9990868267559281\n\nExtracted the following KBP triples:\n1.0     Joe Smith       per:statesorprovinces_of_residence      California\n'
EN_DOC_POS_ONLY_GOLD = '\nSentence #1 (6 tokens):\nJoe Smith lives in California.\n\nTokens:\n[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP]\n[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP]\n[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ]\n[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN]\n[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP]\n[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=.]\n'

def test_english_request():
    if False:
        while True:
            i = 10
    ' Test case of starting server with Spanish defaults, and then requesting default English properties '
    with corenlp.CoreNLPClient(properties='spanish', server_id='test_spanish_english_request') as client:
        ann = client.annotate(EN_DOC, properties='english', output_format='text')
        compare_ignoring_whitespace(ann, EN_DOC_GOLD)
    with corenlp.CoreNLPClient(properties='english', server_id='test_english_request') as client:
        ann = client.annotate(EN_DOC, output_format='text')
        compare_ignoring_whitespace(ann, EN_DOC_GOLD)

def test_default_annotators():
    if False:
        while True:
            i = 10
    "\n    Test case of creating a client with start_server=False and a set of annotators\n    The annotators should be used instead of the server's default annotators\n    "
    with corenlp.CoreNLPClient(server_id='test_default_annotators', output_format='text', annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'depparse']) as client:
        with corenlp.CoreNLPClient(start_server=False, output_format='text', annotators=['tokenize', 'ssplit', 'pos']) as client2:
            ann = client2.annotate(EN_DOC)
expected_codepoints = ((0, 1), (2, 4), (5, 8), (9, 15), (16, 20))
expected_characters = ((0, 1), (2, 4), (5, 10), (11, 17), (18, 22))
codepoint_doc = 'I am 𝒚̂𝒊 random text'

def test_codepoints():
    if False:
        for i in range(10):
            print('nop')
    ' Test case of asking for codepoints from the English tokenizer '
    with corenlp.CoreNLPClient(annotators=['tokenize', 'ssplit'], properties={'tokenize.codepoint': 'true'}) as client:
        ann = client.annotate(codepoint_doc)
        for (i, (codepoints, characters)) in enumerate(zip(expected_codepoints, expected_characters)):
            token = ann.sentence[0].token[i]
            assert token.codepointOffsetBegin == codepoints[0]
            assert token.codepointOffsetEnd == codepoints[1]
            assert token.beginChar == characters[0]
            assert token.endChar == characters[1]

def test_codepoint_text():
    if False:
        while True:
            i = 10
    ' Test case of extracting the correct sentence text using codepoints '
    text = 'Unban mox opal 🐱.  This is a second sentence.'
    with corenlp.CoreNLPClient(annotators=['tokenize', 'ssplit'], properties={'tokenize.codepoint': 'true'}) as client:
        ann = client.annotate(text)
        text_start = ann.sentence[0].token[0].codepointOffsetBegin
        text_end = ann.sentence[0].token[-1].codepointOffsetEnd
        sentence_text = text[text_start:text_end]
        assert sentence_text == 'Unban mox opal 🐱.'
        text_start = ann.sentence[1].token[0].codepointOffsetBegin
        text_end = ann.sentence[1].token[-1].codepointOffsetEnd
        sentence_text = text[text_start:text_end]
        assert sentence_text == 'This is a second sentence.'