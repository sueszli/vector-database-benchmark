from __future__ import absolute_import, division, print_function, unicode_literals
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

def test_parse_plaintext():
    if False:
        i = 10
        return i + 15
    parser = PlaintextParser.from_string('\n        Ako sa máš? Ja dobre! A ty? No\n        mohlo to byť aj lepšie!!! Ale pohodička.\n\n\n        TOTO JE AKOŽE NADPIS\n        A toto je text pod ním, ktorý je textový.\n        A tak ďalej...\n    ', Tokenizer('czech'))
    document = parser.document
    assert len(document.paragraphs) == 2
    assert len(document.paragraphs[0].headings) == 0
    assert len(document.paragraphs[0].sentences) == 5
    assert len(document.paragraphs[1].headings) == 1
    assert len(document.paragraphs[1].sentences) == 2

def test_parse_plaintext_long():
    if False:
        print('Hello World!')
    parser = PlaintextParser.from_string('\n        Ako sa máš? Ja dobre! A ty? No\n        mohlo to byť aj lepšie!!! Ale pohodička.\n\n        TOTO JE AKOŽE NADPIS\n        A toto je text pod ním, ktorý je textový.\n        A tak ďalej...\n\n        VEĽKOLEPÉ PREKVAPENIE\n        Tretí odstavec v tomto texte je úplne o ničom. Ale má\n        vety a to je hlavné. Takže sa majte na pozore ;-)\n\n        A tak ďalej...\n\n\n        A tak este dalej!\n    ', Tokenizer('czech'))
    document = parser.document
    assert len(document.paragraphs) == 5
    assert len(document.paragraphs[0].headings) == 0
    assert len(document.paragraphs[0].sentences) == 5
    assert len(document.paragraphs[1].headings) == 1
    assert len(document.paragraphs[1].sentences) == 2
    assert len(document.paragraphs[2].headings) == 1
    assert len(document.paragraphs[2].sentences) == 3
    assert len(document.paragraphs[3].headings) == 0
    assert len(document.paragraphs[3].sentences) == 1
    assert len(document.paragraphs[4].headings) == 0
    assert len(document.paragraphs[4].sentences) == 1