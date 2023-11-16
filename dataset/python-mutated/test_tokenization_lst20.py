import os
import tempfile
import pytest
import stanza
from stanza.tests import *
from stanza.utils.datasets.common import convert_conllu_to_txt
from stanza.utils.datasets.tokenization.convert_th_lst20 import read_document
from stanza.utils.datasets.tokenization.process_thai_tokenization import write_section
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]
SMALL_LST_SAMPLE = '\nสุรยุทธ์\tNN\tB_PER\tB_CLS\nยัน\tVV\tO\tI_CLS\nปฏิเสธ\tVV\tO\tI_CLS\nลงนาม\tVV\tO\tI_CLS\n_\tPU\tO\tI_CLS\nMOU\tNN\tO\tI_CLS\n_\tPU\tO\tI_CLS\nกับ\tPS\tO\tI_CLS\nอียู\tNN\tB_ORG\tI_CLS\nไม่\tNG\tO\tI_CLS\nกระทบ\tVV\tO\tI_CLS\nสัมพันธ์\tNN\tO\tE_CLS\n\n1\tNU\tB_DTM\tB_CLS\n_\tPU\tI_DTM\tI_CLS\nกันยายน\tNN\tI_DTM\tI_CLS\n_\tPU\tI_DTM\tI_CLS\n2550\tNU\tE_DTM\tI_CLS\n_\tPU\tO\tI_CLS\n12:21\tNU\tB_DTM\tI_CLS\n_\tPU\tI_DTM\tI_CLS\nน.\tCL\tE_DTM\tE_CLS\n\nผู้สื่อข่าว\tNN\tO\tB_CLS\nรายงาน\tVV\tO\tI_CLS\nเพิ่มเติม\tVV\tO\tI_CLS\nว่า\tCC\tO\tE_CLS\n_\tPU\tO\tO\nจาก\tPS\tO\tB_CLS\nการ\tFX\tO\tI_CLS\nลง\tVV\tO\tI_CLS\nพื้นที่\tNN\tO\tI_CLS\nพบ\tVV\tO\tI_CLS\nว่า\tCC\tO\tE_CLS\n'.strip()
EXPECTED_CONLLU = '\n1\tสุรยุทธ์\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No|NewPar=Yes\n2\tยัน\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tปฏิเสธ\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tลงนาม\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tMOU\t_\t_\t_\t_\t4\tdep\t4:dep\t_\n6\tกับ\t_\t_\t_\t_\t5\tdep\t5:dep\tSpaceAfter=No\n7\tอียู\t_\t_\t_\t_\t6\tdep\t6:dep\tSpaceAfter=No\n8\tไม่\t_\t_\t_\t_\t7\tdep\t7:dep\tSpaceAfter=No\n9\tกระทบ\t_\t_\t_\t_\t8\tdep\t8:dep\tSpaceAfter=No\n10\tสัมพันธ์\t_\t_\t_\t_\t9\tdep\t9:dep\tSpaceAfter=No\n\n1\t1\t_\t_\t_\t_\t0\troot\t0:root\t_\n2\tกันยายน\t_\t_\t_\t_\t1\tdep\t1:dep\t_\n3\t2550\t_\t_\t_\t_\t2\tdep\t2:dep\t_\n4\t12:21\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tน.\t_\t_\t_\t_\t4\tdep\t4:dep\tSpaceAfter=No\n\n1\tผู้สื่อข่าว\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No\n2\tรายงาน\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tเพิ่มเติม\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tว่า\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tจาก\t_\t_\t_\t_\t4\tdep\t4:dep\tSpaceAfter=No\n6\tการ\t_\t_\t_\t_\t5\tdep\t5:dep\tSpaceAfter=No\n7\tลง\t_\t_\t_\t_\t6\tdep\t6:dep\tSpaceAfter=No\n8\tพื้นที่\t_\t_\t_\t_\t7\tdep\t7:dep\tSpaceAfter=No\n9\tพบ\t_\t_\t_\t_\t8\tdep\t8:dep\tSpaceAfter=No\n10\tว่า\t_\t_\t_\t_\t9\tdep\t9:dep\tSpaceAfter=No\n'.strip()
EXPECTED_TXT = 'สุรยุทธ์ยันปฏิเสธลงนาม MOU กับอียูไม่กระทบสัมพันธ์1 กันยายน 2550 12:21 น.ผู้สื่อข่าวรายงานเพิ่มเติมว่า จากการลงพื้นที่พบว่า\n\n'
EXPECTED_LABELS = '000000010010000010000100010001000100100001000000021000000010000100000100200000000001000001000000001001000100101000000101002\n\n'

def check_results(documents, expected_conllu, expected_txt, expected_labels):
    if False:
        return 10
    with tempfile.TemporaryDirectory() as output_dir:
        write_section(output_dir, 'lst20', 'train', documents)
        with open(os.path.join(output_dir, 'th_lst20.train.gold.conllu')) as fin:
            conllu = fin.read().strip()
        with open(os.path.join(output_dir, 'th_lst20.train.txt')) as fin:
            txt = fin.read()
        with open(os.path.join(output_dir, 'th_lst20-ud-train.toklabels')) as fin:
            labels = fin.read()
        assert conllu == expected_conllu
        assert txt == expected_txt
        assert labels == expected_labels
        assert len(txt) == len(labels)

def test_small():
    if False:
        while True:
            i = 10
    '\n    A small test just to verify that the output is being produced as we want\n\n    Note that there currently are no spaces after the first sentence.\n    Apparently this is wrong, but weirdly, doing that makes the model even worse.\n    '
    lines = SMALL_LST_SAMPLE.strip().split('\n')
    documents = read_document(lines, spaces_after=False, split_clauses=False)
    check_results(documents, EXPECTED_CONLLU, EXPECTED_TXT, EXPECTED_LABELS)
EXPECTED_SPACE_CONLLU = '\n1\tสุรยุทธ์\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No|NewPar=Yes\n2\tยัน\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tปฏิเสธ\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tลงนาม\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tMOU\t_\t_\t_\t_\t4\tdep\t4:dep\t_\n6\tกับ\t_\t_\t_\t_\t5\tdep\t5:dep\tSpaceAfter=No\n7\tอียู\t_\t_\t_\t_\t6\tdep\t6:dep\tSpaceAfter=No\n8\tไม่\t_\t_\t_\t_\t7\tdep\t7:dep\tSpaceAfter=No\n9\tกระทบ\t_\t_\t_\t_\t8\tdep\t8:dep\tSpaceAfter=No\n10\tสัมพันธ์\t_\t_\t_\t_\t9\tdep\t9:dep\t_\n\n1\t1\t_\t_\t_\t_\t0\troot\t0:root\t_\n2\tกันยายน\t_\t_\t_\t_\t1\tdep\t1:dep\t_\n3\t2550\t_\t_\t_\t_\t2\tdep\t2:dep\t_\n4\t12:21\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tน.\t_\t_\t_\t_\t4\tdep\t4:dep\t_\n\n1\tผู้สื่อข่าว\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No\n2\tรายงาน\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tเพิ่มเติม\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tว่า\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tจาก\t_\t_\t_\t_\t4\tdep\t4:dep\tSpaceAfter=No\n6\tการ\t_\t_\t_\t_\t5\tdep\t5:dep\tSpaceAfter=No\n7\tลง\t_\t_\t_\t_\t6\tdep\t6:dep\tSpaceAfter=No\n8\tพื้นที่\t_\t_\t_\t_\t7\tdep\t7:dep\tSpaceAfter=No\n9\tพบ\t_\t_\t_\t_\t8\tdep\t8:dep\tSpaceAfter=No\n10\tว่า\t_\t_\t_\t_\t9\tdep\t9:dep\t_\n'.strip()
EXPECTED_SPACE_TXT = 'สุรยุทธ์ยันปฏิเสธลงนาม MOU กับอียูไม่กระทบสัมพันธ์ 1 กันยายน 2550 12:21 น. ผู้สื่อข่าวรายงานเพิ่มเติมว่า จากการลงพื้นที่พบว่า\n\n'
EXPECTED_SPACE_LABELS = '00000001001000001000010001000100010010000100000002010000000100001000001002000000000001000001000000001001000100101000000101002\n\n'

def test_space_after():
    if False:
        return 10
    '\n    This version of the test adds the space after attribute\n    '
    lines = SMALL_LST_SAMPLE.strip().split('\n')
    documents = read_document(lines, spaces_after=True, split_clauses=False)
    check_results(documents, EXPECTED_SPACE_CONLLU, EXPECTED_SPACE_TXT, EXPECTED_SPACE_LABELS)
EXPECTED_CLAUSE_CONLLU = '\n1\tสุรยุทธ์\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No|NewPar=Yes\n2\tยัน\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tปฏิเสธ\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tลงนาม\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tMOU\t_\t_\t_\t_\t4\tdep\t4:dep\t_\n6\tกับ\t_\t_\t_\t_\t5\tdep\t5:dep\tSpaceAfter=No\n7\tอียู\t_\t_\t_\t_\t6\tdep\t6:dep\tSpaceAfter=No\n8\tไม่\t_\t_\t_\t_\t7\tdep\t7:dep\tSpaceAfter=No\n9\tกระทบ\t_\t_\t_\t_\t8\tdep\t8:dep\tSpaceAfter=No\n10\tสัมพันธ์\t_\t_\t_\t_\t9\tdep\t9:dep\t_\n\n1\t1\t_\t_\t_\t_\t0\troot\t0:root\t_\n2\tกันยายน\t_\t_\t_\t_\t1\tdep\t1:dep\t_\n3\t2550\t_\t_\t_\t_\t2\tdep\t2:dep\t_\n4\t12:21\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tน.\t_\t_\t_\t_\t4\tdep\t4:dep\t_\n\n1\tผู้สื่อข่าว\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No\n2\tรายงาน\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tเพิ่มเติม\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tว่า\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n\n1\tจาก\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No\n2\tการ\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tลง\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tพื้นที่\t_\t_\t_\t_\t3\tdep\t3:dep\tSpaceAfter=No\n5\tพบ\t_\t_\t_\t_\t4\tdep\t4:dep\tSpaceAfter=No\n6\tว่า\t_\t_\t_\t_\t5\tdep\t5:dep\t_\n'.strip()
EXPECTED_CLAUSE_TXT = 'สุรยุทธ์ยันปฏิเสธลงนาม MOU กับอียูไม่กระทบสัมพันธ์ 1 กันยายน 2550 12:21 น. ผู้สื่อข่าวรายงานเพิ่มเติมว่า จากการลงพื้นที่พบว่า\n\n'
EXPECTED_CLAUSE_LABELS = '00000001001000001000010001000100010010000100000002010000000100001000001002000000000001000001000000001002000100101000000101002\n\n'

def test_split_clause():
    if False:
        while True:
            i = 10
    '\n    This version of the test also resplits on spaces between clauses\n    '
    lines = SMALL_LST_SAMPLE.strip().split('\n')
    documents = read_document(lines, spaces_after=True, split_clauses=True)
    check_results(documents, EXPECTED_CLAUSE_CONLLU, EXPECTED_CLAUSE_TXT, EXPECTED_CLAUSE_LABELS)
if __name__ == '__main__':
    lines = SMALL_LST_SAMPLE.strip().split('\n')
    documents = read_document(lines, spaces_after=False, split_clauses=False)
    write_section('foo', 'lst20', 'train', documents)