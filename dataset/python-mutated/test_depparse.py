"""
Basic tests of the depparse processor boolean flags
"""
import pytest
import stanza
from stanza.pipeline.core import PipelineRequirementsException
from stanza.utils.conll import CoNLL
from stanza.tests import *
pytestmark = pytest.mark.pipeline
EN_DOC = 'Barack Obama was born in Hawaii.  He was elected president in 2008.  Obama attended Harvard.'
EN_DOC_CONLLU_PRETAGGED = '\n1\tBarack\tBarack\tPROPN\tNNP\tNumber=Sing\t0\t_\t_\t_\n2\tObama\tObama\tPROPN\tNNP\tNumber=Sing\t1\t_\t_\t_\n3\twas\tbe\tAUX\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t2\t_\t_\t_\n4\tborn\tbear\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t3\t_\t_\t_\n5\tin\tin\tADP\tIN\t_\t4\t_\t_\t_\n6\tHawaii\tHawaii\tPROPN\tNNP\tNumber=Sing\t5\t_\t_\t_\n7\t.\t.\tPUNCT\t.\t_\t6\t_\t_\t_\n\n1\tHe\the\tPRON\tPRP\tCase=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs\t0\t_\t_\t_\n2\twas\tbe\tAUX\tVBD\tMood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin\t1\t_\t_\t_\n3\telected\telect\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t2\t_\t_\t_\n4\tpresident\tpresident\tPROPN\tNNP\tNumber=Sing\t3\t_\t_\t_\n5\tin\tin\tADP\tIN\t_\t4\t_\t_\t_\n6\t2008\t2008\tNUM\tCD\tNumType=Card\t5\t_\t_\t_\n7\t.\t.\tPUNCT\t.\t_\t6\t_\t_\t_\n\n1\tObama\tObama\tPROPN\tNNP\tNumber=Sing\t0\t_\t_\t_\n2\tattended\tattend\tVERB\tVBD\tMood=Ind|Tense=Past|VerbForm=Fin\t1\t_\t_\t_\n3\tHarvard\tHarvard\tPROPN\tNNP\tNumber=Sing\t2\t_\t_\t_\n4\t.\t.\tPUNCT\t.\t_\t3\t_\t_\t_\n\n\n'.lstrip()
EN_DOC_DEPENDENCY_PARSES_GOLD = "\n('Barack', 4, 'nsubj:pass')\n('Obama', 1, 'flat')\n('was', 4, 'aux:pass')\n('born', 0, 'root')\n('in', 6, 'case')\n('Hawaii', 4, 'obl')\n('.', 4, 'punct')\n\n('He', 3, 'nsubj:pass')\n('was', 3, 'aux:pass')\n('elected', 0, 'root')\n('president', 3, 'xcomp')\n('in', 6, 'case')\n('2008', 3, 'obl')\n('.', 3, 'punct')\n\n('Obama', 2, 'nsubj')\n('attended', 0, 'root')\n('Harvard', 2, 'obj')\n('.', 2, 'punct')\n".strip()

@pytest.fixture(scope='module')
def en_depparse_pipeline():
    if False:
        i = 10
        return i + 15
    nlp = stanza.Pipeline(dir=TEST_MODELS_DIR, lang='en', processors='tokenize,pos,lemma,depparse')
    return nlp

def test_depparse(en_depparse_pipeline):
    if False:
        while True:
            i = 10
    doc = en_depparse_pipeline(EN_DOC)
    assert EN_DOC_DEPENDENCY_PARSES_GOLD == '\n\n'.join([sent.dependencies_string() for sent in doc.sentences])

def test_depparse_with_pretagged_doc():
    if False:
        return 10
    nlp = stanza.Pipeline(**{'processors': 'depparse', 'dir': TEST_MODELS_DIR, 'lang': 'en', 'depparse_pretagged': True})
    doc = CoNLL.conll2doc(input_str=EN_DOC_CONLLU_PRETAGGED)
    processed_doc = nlp(doc)
    assert EN_DOC_DEPENDENCY_PARSES_GOLD == '\n\n'.join([sent.dependencies_string() for sent in processed_doc.sentences])

def test_raises_requirements_exception_if_pretagged_not_passed():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(PipelineRequirementsException):
        stanza.Pipeline(**{'processors': 'depparse', 'dir': TEST_MODELS_DIR, 'lang': 'en'})