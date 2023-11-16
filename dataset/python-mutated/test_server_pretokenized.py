"""
Misc tests for the server
"""
import pytest
import re
from stanza.server import CoreNLPClient
pytestmark = pytest.mark.client
tokens = {}
tags = {}
tokens['italian'] = ['È vero , tutti possiamo essere sostituiti .\n Alcune chiamate partirono da il Quirinale .']
tags['italian'] = [[['AUX', 'ADJ', 'PUNCT', 'PRON', 'AUX', 'AUX', 'VERB', 'PUNCT'], ['DET', 'NOUN', 'VERB', 'ADP', 'DET', 'PROPN', 'PUNCT']]]
tokens['french'] = ['Les études durent six ans mais leur contenu diffère donc selon les Facultés .\nIl est fêté le 22 mai .']
tags['french'] = [[['DET', 'NOUN', 'VERB', 'NUM', 'NOUN', 'CCONJ', 'DET', 'NOUN', 'VERB', 'ADV', 'ADP', 'DET', 'PROPN', 'PUNCT'], ['PRON', 'AUX', 'VERB', 'DET', 'NUM', 'NOUN', 'PUNCT']]]
tokens['english'] = ["This shouldn't be split .\n I hope it's not ."]
tags['english'] = [[['DT', 'NN', 'VB', 'VBN', '.'], ['PRP', 'VBP', 'PRP$', 'RB', '.']]]

def pretokenized_test(lang):
    if False:
        while True:
            i = 10
    'Test submitting pretokenized French text.'
    with CoreNLPClient(properties=lang, annotators='pos', pretokenized=True, be_quiet=True) as client:
        for (input_text, gold_tags) in zip(tokens[lang], tags[lang]):
            ann = client.annotate(input_text)
            for (sentence_tags, sentence) in zip(gold_tags, ann.sentence):
                result_tags = [tok.pos for tok in sentence.token]
                assert sentence_tags == result_tags

def test_english_pretokenized():
    if False:
        while True:
            i = 10
    pretokenized_test('english')

def test_italian_pretokenized():
    if False:
        while True:
            i = 10
    pretokenized_test('italian')

def test_french_pretokenized():
    if False:
        for i in range(10):
            print('nop')
    pretokenized_test('french')