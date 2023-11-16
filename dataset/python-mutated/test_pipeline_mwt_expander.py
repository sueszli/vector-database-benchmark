"""
Basic testing of multi-word-token expansion
"""
import pytest
import stanza
from stanza.tests import *
pytestmark = pytest.mark.pipeline
FR_MWT_SENTENCE = "Alors encore inconnu du grand public, Emmanuel Macron devient en 2014 ministre de l'Économie, de l'Industrie et du Numérique."
FR_MWT_TOKEN_TO_WORDS_GOLD = "\ntoken: Alors    \t\twords: [<Word id=1;text=Alors>]\ntoken: encore   \t\twords: [<Word id=2;text=encore>]\ntoken: inconnu  \t\twords: [<Word id=3;text=inconnu>]\ntoken: du       \t\twords: [<Word id=4;text=de>, <Word id=5;text=le>]\ntoken: grand    \t\twords: [<Word id=6;text=grand>]\ntoken: public   \t\twords: [<Word id=7;text=public>]\ntoken: ,        \t\twords: [<Word id=8;text=,>]\ntoken: Emmanuel \t\twords: [<Word id=9;text=Emmanuel>]\ntoken: Macron   \t\twords: [<Word id=10;text=Macron>]\ntoken: devient  \t\twords: [<Word id=11;text=devient>]\ntoken: en       \t\twords: [<Word id=12;text=en>]\ntoken: 2014     \t\twords: [<Word id=13;text=2014>]\ntoken: ministre \t\twords: [<Word id=14;text=ministre>]\ntoken: de       \t\twords: [<Word id=15;text=de>]\ntoken: l'       \t\twords: [<Word id=16;text=l'>]\ntoken: Économie \t\twords: [<Word id=17;text=Économie>]\ntoken: ,        \t\twords: [<Word id=18;text=,>]\ntoken: de       \t\twords: [<Word id=19;text=de>]\ntoken: l'       \t\twords: [<Word id=20;text=l'>]\ntoken: Industrie\t\twords: [<Word id=21;text=Industrie>]\ntoken: et       \t\twords: [<Word id=22;text=et>]\ntoken: du       \t\twords: [<Word id=23;text=de>, <Word id=24;text=le>]\ntoken: Numérique\t\twords: [<Word id=25;text=Numérique>]\ntoken: .        \t\twords: [<Word id=26;text=.>]\n".strip()
FR_MWT_WORD_TO_TOKEN_GOLD = "\nword: Alors    \t\ttoken parent:1-Alors\nword: encore   \t\ttoken parent:2-encore\nword: inconnu  \t\ttoken parent:3-inconnu\nword: de       \t\ttoken parent:4-5-du\nword: le       \t\ttoken parent:4-5-du\nword: grand    \t\ttoken parent:6-grand\nword: public   \t\ttoken parent:7-public\nword: ,        \t\ttoken parent:8-,\nword: Emmanuel \t\ttoken parent:9-Emmanuel\nword: Macron   \t\ttoken parent:10-Macron\nword: devient  \t\ttoken parent:11-devient\nword: en       \t\ttoken parent:12-en\nword: 2014     \t\ttoken parent:13-2014\nword: ministre \t\ttoken parent:14-ministre\nword: de       \t\ttoken parent:15-de\nword: l'       \t\ttoken parent:16-l'\nword: Économie \t\ttoken parent:17-Économie\nword: ,        \t\ttoken parent:18-,\nword: de       \t\ttoken parent:19-de\nword: l'       \t\ttoken parent:20-l'\nword: Industrie\t\ttoken parent:21-Industrie\nword: et       \t\ttoken parent:22-et\nword: de       \t\ttoken parent:23-24-du\nword: le       \t\ttoken parent:23-24-du\nword: Numérique\t\ttoken parent:25-Numérique\nword: .        \t\ttoken parent:26-.\n".strip()

def test_mwt():
    if False:
        for i in range(10):
            print('nop')
    pipeline = stanza.Pipeline(processors='tokenize,mwt', dir=TEST_MODELS_DIR, lang='fr')
    doc = pipeline(FR_MWT_SENTENCE)
    token_to_words = '\n'.join([f"token: {token.text.ljust(9)}\t\twords: [{', '.join([word.pretty_print() for word in token.words])}]" for sent in doc.sentences for token in sent.tokens]).strip()
    word_to_token = '\n'.join([f"word: {word.text.ljust(9)}\t\ttoken parent:{'-'.join([str(x) for x in word.parent.id])}-{word.parent.text}" for sent in doc.sentences for word in sent.words]).strip()
    assert token_to_words == FR_MWT_TOKEN_TO_WORDS_GOLD
    assert word_to_token == FR_MWT_WORD_TO_TOKEN_GOLD