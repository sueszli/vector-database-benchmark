"""
Basic tests of the data conversion
"""
import io
import pytest
import tempfile
from zipfile import ZipFile
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.tests import *
pytestmark = pytest.mark.pipeline
CONLL = [[['1', 'Nous', 'il', 'PRON', '_', 'Number=Plur|Person=1|PronType=Prs', '3', 'nsubj', '_', 'start_char=0|end_char=4'], ['2', 'avons', 'avoir', 'AUX', '_', 'Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin', '3', 'aux:tense', '_', 'start_char=5|end_char=10'], ['3', 'atteint', 'atteindre', 'VERB', '_', 'Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part', '0', 'root', '_', 'start_char=11|end_char=18'], ['4', 'la', 'le', 'DET', '_', 'Definite=Def|Gender=Fem|Number=Sing|PronType=Art', '5', 'det', '_', 'start_char=19|end_char=21'], ['5', 'fin', 'fin', 'NOUN', '_', 'Gender=Fem|Number=Sing', '3', 'obj', '_', 'start_char=22|end_char=25'], ['6-7', 'du', '_', '_', '_', '_', '_', '_', '_', 'start_char=26|end_char=28'], ['6', 'de', 'de', 'ADP', '_', '_', '8', 'case', '_', '_'], ['7', 'le', 'le', 'DET', '_', 'Definite=Def|Gender=Masc|Number=Sing|PronType=Art', '8', 'det', '_', '_'], ['8', 'sentier', 'sentier', 'NOUN', '_', 'Gender=Masc|Number=Sing', '5', 'nmod', '_', 'start_char=29|end_char=36'], ['9', '.', '.', 'PUNCT', '_', '_', '3', 'punct', '_', 'start_char=36|end_char=37']]]
DICT = [[{'id': (1,), 'text': 'Nous', 'lemma': 'il', 'upos': 'PRON', 'feats': 'Number=Plur|Person=1|PronType=Prs', 'head': 3, 'deprel': 'nsubj', 'misc': 'start_char=0|end_char=4'}, {'id': (2,), 'text': 'avons', 'lemma': 'avoir', 'upos': 'AUX', 'feats': 'Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin', 'head': 3, 'deprel': 'aux:tense', 'misc': 'start_char=5|end_char=10'}, {'id': (3,), 'text': 'atteint', 'lemma': 'atteindre', 'upos': 'VERB', 'feats': 'Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part', 'head': 0, 'deprel': 'root', 'misc': 'start_char=11|end_char=18'}, {'id': (4,), 'text': 'la', 'lemma': 'le', 'upos': 'DET', 'feats': 'Definite=Def|Gender=Fem|Number=Sing|PronType=Art', 'head': 5, 'deprel': 'det', 'misc': 'start_char=19|end_char=21'}, {'id': (5,), 'text': 'fin', 'lemma': 'fin', 'upos': 'NOUN', 'feats': 'Gender=Fem|Number=Sing', 'head': 3, 'deprel': 'obj', 'misc': 'start_char=22|end_char=25'}, {'id': (6, 7), 'text': 'du', 'misc': 'start_char=26|end_char=28'}, {'id': (6,), 'text': 'de', 'lemma': 'de', 'upos': 'ADP', 'head': 8, 'deprel': 'case'}, {'id': (7,), 'text': 'le', 'lemma': 'le', 'upos': 'DET', 'feats': 'Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'head': 8, 'deprel': 'det'}, {'id': (8,), 'text': 'sentier', 'lemma': 'sentier', 'upos': 'NOUN', 'feats': 'Gender=Masc|Number=Sing', 'head': 5, 'deprel': 'nmod', 'misc': 'start_char=29|end_char=36'}, {'id': (9,), 'text': '.', 'lemma': '.', 'upos': 'PUNCT', 'head': 3, 'deprel': 'punct', 'misc': 'start_char=36|end_char=37'}]]

def test_conll_to_dict():
    if False:
        i = 10
        return i + 15
    dicts = CoNLL.convert_conll(CONLL)
    assert dicts == DICT

def test_dict_to_conll():
    if False:
        i = 10
        return i + 15
    document = Document(DICT)
    conll = [[sentence.split('\t') for sentence in doc.split('\n')] for doc in '{:c}'.format(document).split('\n\n')]
    assert conll == CONLL

def test_dict_to_doc_and_doc_to_dict():
    if False:
        print('Hello World!')
    '\n    Test the conversion from raw dict to Document and back\n\n    This code path will first turn start_char|end_char into start_char & end_char fields in the Document\n    That version to a dict will have separate fields for each of those\n    Finally, the conversion from that dict to a list of conll entries should convert that back to misc\n    '
    document = Document(DICT)
    dicts = document.to_dict()
    document = Document(dicts)
    conll = [[sentence.split('\t') for sentence in doc.split('\n')] for doc in '{:c}'.format(document).split('\n\n')]
    assert conll == CONLL
RUSSIAN_SAMPLE = '\n# sent_id = yandex.reviews-f-8xh5zqnmwak3t6p68y4rhwd4e0-1969-9253\n# genre = review\n# text = Как- то слишком мало цветов получают актёры после спектакля.\n1\tКак\tкак-то\tADV\t_\tDegree=Pos|PronType=Ind\t7\tadvmod\t_\tSpaceAfter=No\n2\t-\t-\tPUNCT\t_\t_\t3\tpunct\t_\t_\n3\tто\tто\tPART\t_\t_\t1\tlist\t_\tdeprel=list:goeswith\n4\tслишком\tслишком\tADV\t_\tDegree=Pos\t5\tadvmod\t_\t_\n5\tмало\tмало\tADV\t_\tDegree=Pos\t6\tadvmod\t_\t_\n6\tцветов\tцветок\tNOUN\t_\tAnimacy=Inan|Case=Gen|Gender=Masc|Number=Plur\t7\tobj\t_\t_\n7\tполучают\tполучать\tVERB\t_\tAspect=Imp|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act\t0\troot\t_\t_\n8\tактёры\tактер\tNOUN\t_\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Plur\t7\tnsubj\t_\t_\n9\tпосле\tпосле\tADP\t_\t_\t10\tcase\t_\t_\n10\tспектакля\tспектакль\tNOUN\t_\tAnimacy=Inan|Case=Gen|Gender=Masc|Number=Sing\t7\tobl\t_\tSpaceAfter=No\n11\t.\t.\tPUNCT\t_\t_\t7\tpunct\t_\t_\n\n# sent_id = 4\n# genre = social\n# text = В женщине важна верность, а не красота.\n1\tВ\tв\tADP\t_\t_\t2\tcase\t_\t_\n2\tженщине\tженщина\tNOUN\t_\tAnimacy=Anim|Case=Loc|Gender=Fem|Number=Sing\t3\tobl\t_\t_\n3\tважна\tважный\tADJ\t_\tDegree=Pos|Gender=Fem|Number=Sing|Variant=Short\t0\troot\t_\t_\n4\tверность\tверность\tNOUN\t_\tAnimacy=Inan|Case=Nom|Gender=Fem|Number=Sing\t3\tnsubj\t_\tSpaceAfter=No\n5\t,\t,\tPUNCT\t_\t_\t8\tpunct\t_\t_\n6\tа\tа\tCCONJ\t_\t_\t8\tcc\t_\t_\n7\tне\tне\tPART\t_\tPolarity=Neg\t8\tadvmod\t_\t_\n8\tкрасота\tкрасота\tNOUN\t_\tAnimacy=Inan|Case=Nom|Gender=Fem|Number=Sing\t4\tconj\t_\tSpaceAfter=No\n9\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_\n'.strip()
RUSSIAN_TEXT = ['Как- то слишком мало цветов получают актёры после спектакля.', 'В женщине важна верность, а не красота.']
RUSSIAN_IDS = ['yandex.reviews-f-8xh5zqnmwak3t6p68y4rhwd4e0-1969-9253', '4']

def check_russian_doc(doc):
    if False:
        print('Hello World!')
    '\n    Refactored the test for the Russian doc so we can use it to test various file methods\n    '
    lines = RUSSIAN_SAMPLE.split('\n')
    assert len(doc.sentences) == 2
    assert lines[0] == doc.sentences[0].comments[0]
    assert lines[1] == doc.sentences[0].comments[1]
    assert lines[2] == doc.sentences[0].comments[2]
    for (sent_idx, (expected_text, expected_id, sentence)) in enumerate(zip(RUSSIAN_TEXT, RUSSIAN_IDS, doc.sentences)):
        assert expected_text == sentence.text
        assert expected_id == sentence.sent_id
        assert sent_idx == sentence.index
        assert len(sentence.comments) == 3
    sentences = '{:C}'.format(doc)
    sentences = sentences.split('\n\n')
    assert len(sentences) == 2
    sentence = sentences[0].split('\n')
    assert len(sentence) == 14
    assert lines[0] == sentence[0]
    assert lines[1] == sentence[1]
    assert lines[2] == sentence[2]
    assert doc.sentences[0].words[2].head == 1
    assert doc.sentences[0].words[2].deprel == 'list:goeswith'

def test_write_russian_doc(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Specifically test the write_doc2conll method\n    '
    filename = tmp_path / 'russian.conll'
    doc = CoNLL.conll2doc(input_str=RUSSIAN_SAMPLE)
    check_russian_doc(doc)
    CoNLL.write_doc2conll(doc, filename)
    with open(filename) as fin:
        text = fin.read()
    assert text.endswith('\n\n')
    text = text.strip()
    text = text[text.find('# sent_id = 4'):]
    sample = RUSSIAN_SAMPLE[RUSSIAN_SAMPLE.find('# sent_id = 4'):]
    assert text == sample
    doc2 = CoNLL.conll2doc(filename)
    check_russian_doc(doc2)
ENGLISH_SAMPLE = '\n# newdoc\n# sent_id = 1\n# text = It is hers.\n# previous = Which person owns this?\n# comment = copular subject\n1\tIt\tit\tPRON\tPRP\tNumber=Sing|Person=3|PronType=Prs\t3\tnsubj\t_\t_\n2\tis\tbe\tAUX\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t3\tcop\t_\t_\n3\thers\thers\tPRON\tPRP\tGender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs\t0\troot\t_\tSpaceAfter=No\n4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_\n'.strip()

def test_write_to_io():
    if False:
        print('Hello World!')
    doc = CoNLL.conll2doc(input_str=ENGLISH_SAMPLE)
    output = io.StringIO()
    CoNLL.write_doc2conll(doc, output)
    output_value = output.getvalue()
    assert output_value.endswith('\n\n')
    assert output_value.strip() == ENGLISH_SAMPLE

def test_write_doc2conll_append(tmp_path):
    if False:
        return 10
    doc = CoNLL.conll2doc(input_str=ENGLISH_SAMPLE)
    filename = tmp_path / 'english.conll'
    CoNLL.write_doc2conll(doc, filename)
    CoNLL.write_doc2conll(doc, filename, mode='a')
    with open(filename) as fin:
        text = fin.read()
    expected = ENGLISH_SAMPLE + '\n\n' + ENGLISH_SAMPLE + '\n\n'
    assert text == expected

def test_doc_with_comments():
    if False:
        i = 10
        return i + 15
    '\n    Test that a doc with comments gets converted back with comments\n    '
    doc = CoNLL.conll2doc(input_str=RUSSIAN_SAMPLE)
    check_russian_doc(doc)

def test_unusual_misc():
    if False:
        for i in range(10):
            print('nop')
    '\n    The above RUSSIAN_SAMPLE resulted in a blank misc field in one particular implementation of the conll code\n    (the below test would fail)\n    '
    doc = CoNLL.conll2doc(input_str=RUSSIAN_SAMPLE)
    sentences = '{:C}'.format(doc).split('\n\n')
    assert len(sentences) == 2
    sentence = sentences[0].split('\n')
    assert len(sentence) == 14
    for word in sentence:
        pieces = word.split('\t')
        assert len(pieces) == 1 or len(pieces) == 10
        if len(pieces) == 10:
            assert all((piece for piece in pieces))

def test_file():
    if False:
        while True:
            i = 10
    '\n    Test loading a doc from a file\n    '
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'russian.conll')
        with open(filename, 'w', encoding='utf-8') as fout:
            fout.write(RUSSIAN_SAMPLE)
        doc = CoNLL.conll2doc(input_file=filename)
        check_russian_doc(doc)

def test_zip_file():
    if False:
        while True:
            i = 10
    '\n    Test loading a doc from a zip file\n    '
    with tempfile.TemporaryDirectory() as tempdir:
        zip_file = os.path.join(tempdir, 'russian.zip')
        filename = 'russian.conll'
        with ZipFile(zip_file, 'w') as zout:
            with zout.open(filename, 'w') as fout:
                fout.write(RUSSIAN_SAMPLE.encode())
        doc = CoNLL.conll2doc(input_file=filename, zip_file=zip_file)
        check_russian_doc(doc)
SIMPLE_NER = "\n# text = Teferi's best friend is Karn\n# sent_id = 0\n1\tTeferi\t_\t_\t_\t_\t0\t_\t_\tstart_char=0|end_char=6|ner=S-PERSON\n2\t's\t_\t_\t_\t_\t1\t_\t_\tstart_char=6|end_char=8|ner=O\n3\tbest\t_\t_\t_\t_\t2\t_\t_\tstart_char=9|end_char=13|ner=O\n4\tfriend\t_\t_\t_\t_\t3\t_\t_\tstart_char=14|end_char=20|ner=O\n5\tis\t_\t_\t_\t_\t4\t_\t_\tstart_char=21|end_char=23|ner=O\n6\tKarn\t_\t_\t_\t_\t5\t_\t_\tstart_char=24|end_char=28|ner=S-PERSON\n".strip()

def test_simple_ner_conversion():
    if False:
        return 10
    '\n    Test that tokens get properly created with NER tags\n    '
    doc = CoNLL.conll2doc(input_str=SIMPLE_NER)
    assert len(doc.sentences) == 1
    sentence = doc.sentences[0]
    assert len(sentence.tokens) == 6
    EXPECTED_NER = ['S-PERSON', 'O', 'O', 'O', 'O', 'S-PERSON']
    for (token, ner) in zip(sentence.tokens, EXPECTED_NER):
        assert token.ner == ner
        assert not token.misc
        assert len(token.words) == 1
        assert not token.words[0].misc
    conll = '{:C}'.format(doc)
    assert conll == SIMPLE_NER
MWT_NER = "\n# text = This makes John's headache worse\n# sent_id = 0\n1\tThis\t_\t_\t_\t_\t0\t_\t_\tstart_char=0|end_char=4|ner=O\n2\tmakes\t_\t_\t_\t_\t1\t_\t_\tstart_char=5|end_char=10|ner=O\n3-4\tJohn's\t_\t_\t_\t_\t_\t_\t_\tstart_char=11|end_char=17|ner=S-PERSON\n3\tJohn\t_\t_\t_\t_\t2\t_\t_\t_\n4\t's\t_\t_\t_\t_\t3\t_\t_\t_\n5\theadache\t_\t_\t_\t_\t4\t_\t_\tstart_char=18|end_char=26|ner=O\n6\tworse\t_\t_\t_\t_\t5\t_\t_\tstart_char=27|end_char=32|ner=O\n".strip()

def test_mwt_ner_conversion():
    if False:
        print('Hello World!')
    '\n    Test that tokens including MWT get properly created with NER tags\n\n    Note that this kind of thing happens with the EWT tokenizer for English, for example\n    '
    doc = CoNLL.conll2doc(input_str=MWT_NER)
    assert len(doc.sentences) == 1
    sentence = doc.sentences[0]
    assert len(sentence.tokens) == 5
    EXPECTED_NER = ['O', 'O', 'S-PERSON', 'O', 'O']
    EXPECTED_WORDS = [1, 1, 2, 1, 1]
    for (token, ner, expected_words) in zip(sentence.tokens, EXPECTED_NER, EXPECTED_WORDS):
        assert token.ner == ner
        assert not token.misc
        assert len(token.words) == expected_words
        assert not token.words[0].misc
    conll = '{:C}'.format(doc)
    assert conll == MWT_NER
ESTONIAN_DEPS = '\n# newpar\n# sent_id = aia_foorum_37\n# text = Sestpeale ei mõistagi neid, kes koduaias sortidega tegelevad.\n1\tSestpeale\tsest_peale\tADV\tD\t_\t3\tadvmod\t3:advmod\t_\n2\tei\tei\tAUX\tV\tPolarity=Neg\t3\taux\t3:aux\t_\n3\tmõistagi\tmõistma\tVERB\tV\tConnegative=Yes|Mood=Ind|Tense=Pres|VerbForm=Fin|Voice=Act\t0\troot\t0:root\t_\n4\tneid\ttema\tPRON\tP\tCase=Par|Number=Plur|Person=3|PronType=Prs\t3\tobj\t3:obj|9:nsubj\tSpaceAfter=No\n5\t,\t,\tPUNCT\tZ\t_\t9\tpunct\t9:punct\t_\n6\tkes\tkes\tPRON\tP\tCase=Nom|Number=Plur|PronType=Int,Rel\t9\tnsubj\t4:ref\t_\n7\tkoduaias\tkodu_aed\tNOUN\tS\tCase=Ine|Number=Sing\t9\tobl\t9:obl\t_\n8\tsortidega\tsort\tNOUN\tS\tCase=Com|Number=Plur\t9\tobl\t9:obl\t_\n9\ttegelevad\ttegelema\tVERB\tV\tMood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act\t4\tacl:relcl\t4:acl\tSpaceAfter=No\n10\t.\t.\tPUNCT\tZ\t_\t3\tpunct\t3:punct\t_\n'.strip()

def test_deps_conversion():
    if False:
        while True:
            i = 10
    doc = CoNLL.conll2doc(input_str=ESTONIAN_DEPS)
    assert len(doc.sentences) == 1
    sentence = doc.sentences[0]
    assert len(sentence.tokens) == 10
    word = doc.sentences[0].words[3]
    assert word.deps == '3:obj|9:nsubj'
    conll = '{:C}'.format(doc)
    assert conll == ESTONIAN_DEPS
ESTONIAN_EMPTY_DEPS = '\n# sent_id = ewtb2_000035_15\n# text = Ja paari aasta pärast rôômalt maasikatele ...\n1\tJa\tja\tCCONJ\tJ\t_\t3\tcc\t5.1:cc\t_\n2\tpaari\tpaar\tNUM\tN\tCase=Gen|Number=Sing|NumForm=Word|NumType=Card\t3\tnummod\t3:nummod\t_\n3\taasta\taasta\tNOUN\tS\tCase=Gen|Number=Sing\t0\troot\t5.1:obl\t_\n4\tpärast\tpärast\tADP\tK\tAdpType=Post\t3\tcase\t3:case\t_\n5\trôômalt\trõõmsalt\tADV\tD\tTypo=Yes\t3\tadvmod\t5.1:advmod\tOrphan=Yes|CorrectForm=rõõmsalt\n5.1\tpanna\tpanema\tVERB\tV\tVerbForm=Inf\t_\t_\t0:root\tEmpty=5.1\n6\tmaasikatele\tmaasikas\tNOUN\tS\tCase=All|Number=Plur\t3\tobl\t5.1:obl\tOrphan=Yes\n7\t...\t...\tPUNCT\tZ\t_\t3\tpunct\t5.1:punct\t_\n'.strip()

def test_empty_deps_conversion():
    if False:
        while True:
            i = 10
    "\n    Ideally we would be able to read & recreate the dependencies\n\n    Currently that is not possible.  Perhaps it should be fixed.\n    At the very least, we shouldn't fail horribly when reading this\n    "
    doc = CoNLL.conll2doc(input_str=ESTONIAN_EMPTY_DEPS)
    assert len(doc.sentences) == 1
    sentence = doc.sentences[0]
    conll = '{:C}'.format(doc)