import pytest
from stanza.tests import compare_ignoring_whitespace
pytestmark = [pytest.mark.travis, pytest.mark.client]
from stanza.utils.conll import CoNLL
import stanza.server.ssurgeon as ssurgeon
SAMPLE_DOC_INPUT = '\n# sent_id = 271\n# text = Hers is easy to clean.\n# previous = What did the dealer like about Alex\'s car?\n# comment = extraction/raising via "tough extraction" and clausal subject\n1\tHers\thers\tPRON\tPRP\tGender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs\t3\tnsubj\t_\t_\n2\tis\tbe\tAUX\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t3\tcop\t_\t_\n3\teasy\teasy\tADJ\tJJ\tDegree=Pos\t0\troot\t_\t_\n4\tto\tto\tPART\tTO\t_\t5\tmark\t_\t_\n5\tclean\tclean\tVERB\tVB\tVerbForm=Inf\t3\tcsubj\t_\tSpaceAfter=No\n6\t.\t.\tPUNCT\t.\t_\t5\tpunct\t_\t_\n'
SAMPLE_DOC_EXPECTED = '\n# sent_id = 271\n# text = Hers is easy to clean.\n# previous = What did the dealer like about Alex\'s car?\n# comment = extraction/raising via "tough extraction" and clausal subject\n1\tHers\thers\tPRON\tPRP\tGender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs\t3\tnsubj\t_\t_\n2\tis\tbe\tAUX\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t3\tcop\t_\t_\n3\teasy\teasy\tADJ\tJJ\tDegree=Pos\t0\troot\t_\t_\n4\tto\tto\tPART\tTO\t_\t5\tmark\t_\t_\n5\tclean\tclean\tVERB\tVB\tVerbForm=Inf\t3\tadvcl\t_\tSpaceAfter=No\n6\t.\t.\tPUNCT\t.\t_\t5\tpunct\t_\t_\n'

def test_ssurgeon_same_length():
    if False:
        i = 10
        return i + 15
    semgrex_pattern = '{}=source >nsubj {} >csubj=bad {}'
    ssurgeon_edits = ['relabelNamedEdge -edge bad -reln advcl']
    doc = CoNLL.conll2doc(input_str=SAMPLE_DOC_INPUT)
    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)
    result = '{:C}'.format(updated_doc)
    compare_ignoring_whitespace(result, SAMPLE_DOC_EXPECTED)
ADD_WORD_DOC_INPUT = "\n# text = Jennifer has lovely antennae.\n# sent_id = 12\n# comment = if you're in to that kind of thing\n1\tJennifer\tJennifer\tPROPN\tNNP\tNumber=Sing\t2\tnsubj\t_\tstart_char=0|end_char=8|ner=S-PERSON\n2\thas\thave\tVERB\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t_\tstart_char=9|end_char=12|ner=O\n3\tlovely\tlovely\tADJ\tJJ\tDegree=Pos\t4\tamod\t_\tstart_char=13|end_char=19|ner=O\n4\tantennae\tantenna\tNOUN\tNNS\tNumber=Plur\t2\tobj\t_\tstart_char=20|end_char=28|ner=O|SpaceAfter=No\n5\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tstart_char=28|end_char=29|ner=O\n"
ADD_WORD_DOC_EXPECTED = "\n# text = Jennifer has lovely blue antennae.\n# sent_id = 12\n# comment = if you're in to that kind of thing\n1\tJennifer\tJennifer\tPROPN\tNNP\tNumber=Sing\t2\tnsubj\t_\tner=S-PERSON\n2\thas\thave\tVERB\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t_\tner=O\n3\tlovely\tlovely\tADJ\tJJ\tDegree=Pos\t5\tamod\t_\tner=O\n4\tblue\tblue\tADJ\tJJ\t_\t5\tamod\t_\tner=O\n5\tantennae\tantenna\tNOUN\tNNS\tNumber=Plur\t2\tobj\t_\tSpaceAfter=No|ner=O\n6\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tner=O\n"

def test_ssurgeon_different_length():
    if False:
        while True:
            i = 10
    semgrex_pattern = '{word:antennae}=antennae !> {word:blue}'
    ssurgeon_edits = ['addDep -gov antennae -reln amod -word blue -lemma blue -cpos ADJ -pos JJ -ner O -position -antennae -after " "']
    doc = CoNLL.conll2doc(input_str=ADD_WORD_DOC_INPUT)
    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)
    result = '{:C}'.format(updated_doc)
    compare_ignoring_whitespace(result, ADD_WORD_DOC_EXPECTED)
BECOME_MWT_DOC_INPUT = "\n# sent_id = 25\n# text = It's not yours!\n# comment = negation \n1\tIt\tit\tPRON\tPRP\tNumber=Sing|Person=2|PronType=Prs\t4\tnsubj\t_\tSpaceAfter=No\n2\t's\tbe\tAUX\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t4\tcop\t_\t_\n3\tnot\tnot\tPART\tRB\tPolarity=Neg\t4\tadvmod\t_\t_\n4\tyours\tyours\tPRON\tPRP\tGender=Neut|Number=Sing|Person=2|Poss=Yes|PronType=Prs\t0\troot\t_\tSpaceAfter=No\n5\t!\t!\tPUNCT\t.\t_\t4\tpunct\t_\t_\n"
BECOME_MWT_DOC_EXPECTED = "\n# sent_id = 25\n# text = It's not yours!\n# comment = negation\n1-2\tIt's\t_\t_\t_\t_\t_\t_\t_\t_\n1\tIt\tit\tPRON\tPRP\tNumber=Sing|Person=2|PronType=Prs\t4\tnsubj\t_\t_\n2\t's\tbe\tAUX\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t4\tcop\t_\t_\n3\tnot\tnot\tPART\tRB\tPolarity=Neg\t4\tadvmod\t_\t_\n4\tyours\tyours\tPRON\tPRP\tGender=Neut|Number=Sing|Person=2|Poss=Yes|PronType=Prs\t0\troot\t_\tSpaceAfter=No\n5\t!\t!\tPUNCT\t.\t_\t4\tpunct\t_\t_\n"

def test_ssurgeon_become_mwt():
    if False:
        return 10
    '\n    Test that converting a document, adding a new MWT, works as expected\n    '
    semgrex_pattern = "{word:It}=it . {word:/'s/}=s"
    ssurgeon_edits = ["EditNode -node it -is_mwt true  -is_first_mwt true  -mwt_text It's", "EditNode -node s  -is_mwt true  -is_first_mwt false -mwt_text It's"]
    doc = CoNLL.conll2doc(input_str=BECOME_MWT_DOC_INPUT)
    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)
    result = '{:C}'.format(updated_doc)
    compare_ignoring_whitespace(result, BECOME_MWT_DOC_EXPECTED)
EXISTING_MWT_DOC_INPUT = '\n# sent_id = newsgroup-groups.google.com_GayMarriage_0ccbb50b41a5830b_ENG_20050321_181500-0005\n# text = One of “NCRC4ME’s”\n1\tOne\tone\tNUM\tCD\tNumType=Card\t0\troot\t0:root\t_\n2\tof\tof\tADP\tIN\t_\t4\tcase\t4:case\t_\n3\t“\t"\tPUNCT\t``\t_\t4\tpunct\t4:punct\tSpaceAfter=No\n4-5\tNCRC4ME’s\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No\n4\tNCRC4ME\tNCRC4ME\tPROPN\tNNP\tNumber=Sing\t1\tcompound\t1:compound\t_\n5\t’s\t\'s\tPART\tPOS\t_\t4\tcase\t4:case\t_\n6\t”\t"\tPUNCT\t\'\'\t_\t4\tpunct\t4:punct\t_\n'
EXISTING_MWT_DOC_EXPECTED = '\n# sent_id = newsgroup-groups.google.com_GayMarriage_0ccbb50b41a5830b_ENG_20050321_181500-0005\n# text = One of “NCRC4ME’s”\n1\tOne\tone\tNUM\tCD\tNumType=Card\t0\troot\t_\t_\n2\tof\tof\tADP\tIN\t_\t4\tcase\t_\t_\n3\t“\t"\tPUNCT\t``\t_\t4\tpunct\t_\tSpaceAfter=No\n4-5\tNCRC4ME’s\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No\n4\tNCRC4ME\tNCRC4ME\tPROPN\tNNP\tNumber=Sing\t1\tcompound\t_\t_\n5\t’s\t\'s\tPART\tPOS\t_\t4\tcase\t_\t_\n6\t”\t"\tPUNCT\t\'\'\t_\t4\tpunct\t_\t_\n'

def test_ssurgeon_existing_mwt_no_change():
    if False:
        while True:
            i = 10
    '\n    Test that converting a document with an MWT works as expected\n\n    Note regarding this test:\n    Currently it works because ssurgeon.py doesn\'t look at the\n      "changed" flag because of a bug in EditNode in CoreNLP 4.5.3\n    If that is fixed, but the enhanced dependencies aren\'t fixed,\n      this test will fail because the enhanced dependencies *aren\'t*\n      removed.  Fixing the enhanced dependencies as well will fix\n      that, though.\n    '
    semgrex_pattern = "{word:It}=it . {word:/'s/}=s"
    ssurgeon_edits = ["EditNode -node it -is_mwt true  -is_first_mwt true  -mwt_text It's", "EditNode -node s  -is_mwt true  -is_first_mwt false -mwt_text It's"]
    doc = CoNLL.conll2doc(input_str=EXISTING_MWT_DOC_INPUT)
    ssurgeon_response = ssurgeon.process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits)
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)
    result = '{:C}'.format(updated_doc)
    compare_ignoring_whitespace(result, EXISTING_MWT_DOC_EXPECTED)

def check_empty_test(input_text, expected=None, echo=False):
    if False:
        while True:
            i = 10
    if expected is None:
        expected = input_text
    doc = CoNLL.conll2doc(input_str=input_text)
    ssurgeon_response = ssurgeon.process_doc(doc, [])
    updated_doc = ssurgeon.convert_response_to_doc(doc, ssurgeon_response)
    result = '{:C}'.format(updated_doc)
    if echo:
        print('INPUT')
        print(input_text)
        print('EXPECTED')
        print(expected)
        print('RESULT')
        print(result)
    compare_ignoring_whitespace(result, expected)
ITALIAN_MWT_INPUT = '\n# sent_id = train_78\n# text = @user dovrebbe fare pace col cervello\n# twittiro = IMPLICIT\tANALOGY\n1\t@user\t@user\tSYM\tSYM\t_\t3\tnsubj\t_\t_\n2\tdovrebbe\tdovere\tAUX\tVM\tMood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t3\taux\t_\t_\n3\tfare\tfare\tVERB\tV\tVerbForm=Inf\t0\troot\t_\t_\n4\tpace\tpace\tNOUN\tS\tGender=Fem|Number=Sing\t3\tobj\t_\t_\n5-6\tcol\t_\t_\t_\t_\t_\t_\t_\t_\n5\tcon\tcon\tADP\tE\t_\t7\tcase\t_\t_\n6\til\til\tDET\tRD\tDefinite=Def|Gender=Masc|Number=Sing|PronType=Art\t7\tdet\t_\t_\n7\tcervello\tcervello\tNOUN\tS\tGender=Masc|Number=Sing\t3\tobl\t_\t_\n'

def test_ssurgeon_mwt_text():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that an MWT which is split into pieces which don\'t make up\n    the original token results in a correct #text annotation\n\n    For example, in Italian, "col" splits into "con il", and we want\n    the #text to contain "col"\n    '
    check_empty_test(ITALIAN_MWT_INPUT)
ITALIAN_SPACES_AFTER_INPUT = '\n# sent_id = train_1114\n# text = ““““ buona scuola ““““\n# twittiro = EXPLICIT\tOTHER\n1\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n2\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n3\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n4\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\t_\n5\tbuona\tbuono\tADJ\tA\tGender=Fem|Number=Sing\t6\tamod\t_\t_\n6\tscuola\tscuola\tNOUN\tS\tGender=Fem|Number=Sing\t0\troot\t_\t_\n7\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n8\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n9\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n10\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpacesAfter=\\n\n'
ITALIAN_SPACES_AFTER_YES_INPUT = '\n# sent_id = train_1114\n# text = ““““ buona scuola ““““\n# twittiro = EXPLICIT\tOTHER\n1\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n2\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n3\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n4\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=Yes\n5\tbuona\tbuono\tADJ\tA\tGender=Fem|Number=Sing\t6\tamod\t_\t_\n6\tscuola\tscuola\tNOUN\tS\tGender=Fem|Number=Sing\t0\troot\t_\t_\n7\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n8\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n9\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpaceAfter=No\n10\t“\t“\tPUNCT\tFB\t_\t6\tpunct\t_\tSpacesAfter=\\n\n'

def test_ssurgeon_spaces_after_text():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that SpacesAfter goes and comes back the same way\n\n    Tested using some random example from the UD_Italian-TWITTIRO dataset\n    '
    check_empty_test(ITALIAN_SPACES_AFTER_INPUT)

def test_ssurgeon_spaces_after_yes():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that an unnecessary SpaceAfter=Yes is eliminated\n    '
    check_empty_test(ITALIAN_SPACES_AFTER_YES_INPUT, ITALIAN_SPACES_AFTER_INPUT)
EMPTY_VALUES_INPUT = "\n# text = Jennifer has lovely antennae.\n# sent_id = 12\n# comment = if you're in to that kind of thing\n1\tJennifer\t_\t_\t_\tNumber=Sing\t2\tnsubj\t_\tner=S-PERSON\n2\thas\t_\t_\t_\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t_\tner=O\n3\tlovely\t_\t_\t_\tDegree=Pos\t4\tamod\t_\tner=O\n4\tantennae\t_\t_\t_\tNumber=Plur\t2\tobj\t_\tSpaceAfter=No|ner=O\n5\t.\t_\t_\t_\t_\t2\tpunct\t_\tner=O\n"

def test_ssurgeon_blank_values():
    if False:
        return 10
    '\n    Check that various None fields such as lemma & xpos are not turned into blanks\n\n    Tests, like regulations, are often written in blood\n    '
    check_empty_test(EMPTY_VALUES_INPUT)
CANTONESE_MISC_WORDS_INPUT = '\n# sent_id = 1\n# text = 你喺度搵乜嘢呀？\n1\t你\t你\tPRON\t_\t_\t3\tnsubj\t_\tSpaceAfter=No|Translit=nei5|Gloss=2SG\n2\t喺度\t喺度\tADV\t_\t_\t3\tadvmod\t_\tSpaceAfter=No|Translit=hai2dou6|Gloss=PROG\n3\t搵\t搵\tVERB\t_\t_\t0\troot\t_\tTranslit=wan2|Gloss=find|SpaceAfter=No\n4\t乜嘢\t乜嘢\tPRON\t_\t_\t3\tobj\t_\tSpaceAfter=No|Translit=mat1je5|Gloss=what\n5\t呀\t呀\tPART\t_\t_\t3\tdiscourse:sp\t_\tSpaceAfter=No|Translit=aa3|Gloss=SFP\n6\t？\t？\tPUNCT\t_\t_\t3\tpunct\t_\tSpaceAfter=No\n\n# sent_id = 2\n# text = 咪執返啲嘢去阿哥個新屋度囖。\n1\t咪\t咪\tADV\t_\t_\t2\tadvmod\t_\tSpaceAfter=No\n2\t執\t執\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No\n3\t返\t返\tVERB\t_\t_\t2\tcompound:dir\t_\tSpaceAfter=No\n4\t啲\t啲\tNOUN\t_\tNounType=Clf\t5\tclf:det\t_\tSpaceAfter=No\n5\t嘢\t嘢\tNOUN\t_\t_\t3\tobj\t_\tSpaceAfter=No\n6\t去\t去\tVERB\t_\t_\t2\tconj\t_\tSpaceAfter=No\n7\t阿哥\t阿哥\tNOUN\t_\t_\t10\tnmod\t_\tSpaceAfter=No\n8\t個\t個\tNOUN\t_\tNounType=Clf\t10\tclf:det\t_\tSpaceAfter=No\n9\t新\t新\tADJ\t_\t_\t10\tamod\t_\tSpaceAfter=No\n10\t屋\t屋\tNOUN\t_\t_\t6\tobj\t_\tSpaceAfter=No\n11\t度\t度\tADP\t_\t_\t10\tcase:loc\t_\tSpaceAfter=No\n12\t囖\t囖\tPART\t_\t_\t2\tdiscourse:sp\t_\tSpaceAfter=No\n13\t。\t。\tPUNCT\t_\t_\t2\tpunct\t_\tSpaceAfter=No\n'

def test_ssurgeon_misc_words():
    if False:
        i = 10
        return i + 15
    '\n    Check that various None fields such as lemma & xpos are not turned into blanks\n\n    Tests, like regulations, are often written in blood\n    '
    check_empty_test(CANTONESE_MISC_WORDS_INPUT)
ITALIAN_MWT_SPACE_AFTER_INPUT = '\n# sent_id = train_78\n# text = @user dovrebbe fare pace colcervello\n# twittiro = IMPLICIT\tANALOGY\n1\t@user\t@user\tSYM\tSYM\t_\t3\tnsubj\t_\t_\n2\tdovrebbe\tdovere\tAUX\tVM\tMood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t3\taux\t_\t_\n3\tfare\tfare\tVERB\tV\tVerbForm=Inf\t0\troot\t_\t_\n4\tpace\tpace\tNOUN\tS\tGender=Fem|Number=Sing\t3\tobj\t_\t_\n5-6\tcol\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No\n5\tcon\tcon\tADP\tE\t_\t7\tcase\t_\t_\n6\til\til\tDET\tRD\tDefinite=Def|Gender=Masc|Number=Sing|PronType=Art\t7\tdet\t_\t_\n7\tcervello\tcervello\tNOUN\tS\tGender=Masc|Number=Sing\t3\tobl\t_\tRandomFeature=foo\n'

def test_ssurgeon_mwt_space_after():
    if False:
        return 10
    '\n    Check the SpaceAfter=No on an MWT (rather than a word)\n\n    the RandomFeature=foo is on account of a silly bug in the initial\n    version of passing in MWT misc features\n    '
    check_empty_test(ITALIAN_MWT_SPACE_AFTER_INPUT)
ITALIAN_MWT_MISC_INPUT = '\n# sent_id = train_78\n# text = @user dovrebbe farepacecolcervello\n# twittiro = IMPLICIT\tANALOGY\n1\t@user\t@user\tSYM\tSYM\t_\t3\tnsubj\t_\t_\n2\tdovrebbe\tdovere\tAUX\tVM\tMood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t3\taux\t_\t_\n3-4\tfarepace\t_\t_\t_\t_\t_\t_\t_\tPlayers=GonnaPlay|SpaceAfter=No\n3\tfare\tfare\tVERB\tV\tVerbForm=Inf\t0\troot\t_\t_\n4\tpace\tpace\tNOUN\tS\tGender=Fem|Number=Sing\t3\tobj\t_\t_\n5-6\tcol\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No|Haters=GonnaHate\n5\tcon\tcon\tADP\tE\t_\t7\tcase\t_\t_\n6\til\til\tDET\tRD\tDefinite=Def|Gender=Masc|Number=Sing|PronType=Art\t7\tdet\t_\t_\n7\tcervello\tcervello\tNOUN\tS\tGender=Masc|Number=Sing\t3\tobl\t_\tRandomFeature=foo\n'

def test_ssurgeon_mwt_misc():
    if False:
        i = 10
        return i + 15
    '\n    Check the SpaceAfter=No on an MWT (rather than a word)\n\n    the RandomFeature=foo is on account of a silly bug in the initial\n    version of passing in MWT misc features\n    '