import os
import tempfile
import pytest
import xml.etree.ElementTree as ET
import stanza
from stanza.tests import *
from stanza.utils.datasets.common import convert_conllu_to_txt
from stanza.utils.datasets.tokenization.convert_th_orchid import parse_xml
from stanza.utils.datasets.tokenization.process_thai_tokenization import write_section
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]
SMALL_DOC = '\n<corpus>\n<document TPublisher="ศูนย์เทคโนโลยีอิเล็กทรอนิกส์และคอมพิวเตอร์แห่งชาติ, กระทรวงวิทยาศาสตร์ เทคโนโลยีและการพลังงาน" EPublisher="National Electronics and Computer Technology Center, Ministry of Science, Technology and Energy" TInbook="การประชุมทางวิชาการ ครั้งที่ 1, โครงการวิจัยและพัฒนาอิเล็กทรอนิกส์และคอมพิวเตอร์, ปีงบประมาณ 2531, เล่ม 1" TTitle="การประชุมทางวิชาการ ครั้งที่ 1" Year="1989" EInbook="The 1st Annual Conference, Electronics and Computer Research and Development Project, Fiscal Year 1988, Book 1" ETitle="[1st Annual Conference]">\n<paragraph id="1" line_num="12">\n<sentence id="1" line_num = "13" raw_txt = "การประชุมทางวิชาการ ครั้งที่ 1">\n<word surface="การ" pos="FIXN"/>\n<word surface="ประชุม" pos="VACT"/>\n<word surface="ทาง" pos="NCMN"/>\n<word surface="วิชาการ" pos="NCMN"/>\n<word surface="&lt;space&gt;" pos="PUNC"/>\n<word surface="ครั้ง" pos="CFQC"/>\n<word surface="ที่ 1" pos="DONM"/>\n</sentence>\n<sentence id="2" line_num = "23" raw_txt = "โครงการวิจัยและพัฒนาอิเล็กทรอนิกส์และคอมพิวเตอร์">\n<word surface="โครงการวิจัยและพัฒนา" pos="NCMN"/>\n<word surface="อิเล็กทรอนิกส์" pos="NCMN"/>\n<word surface="และ" pos="JCRG"/>\n<word surface="คอมพิวเตอร์" pos="NCMN"/>\n</sentence>\n</paragraph>\n<paragraph id="3" line_num="51">\n<sentence id="1" line_num = "52" raw_txt = "วันที่ 15-16 สิงหาคม 2532">\n<word surface="วัน" pos="NCMN"/>\n<word surface="ที่ 15" pos="DONM"/>\n<word surface="&lt;minus&gt;" pos="PUNC"/>\n<word surface="16" pos="DONM"/>\n<word surface="&lt;space&gt;" pos="PUNC"/>\n<word surface="สิงหาคม" pos="NCMN"/>\n<word surface="&lt;space&gt;" pos="PUNC"/>\n<word surface="2532" pos="NCNM"/>\n</sentence>\n</paragraph>\n</document>\n</corpus>\n'
EXPECTED_RESULTS = '\n1\tการ\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No|NewPar=Yes\n2\tประชุม\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tทาง\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tวิชาการ\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tครั้ง\t_\t_\t_\t_\t4\tdep\t4:dep\tSpaceAfter=No\n6\tที่ 1\t_\t_\t_\t_\t5\tdep\t5:dep\t_\n\n1\tโครงการวิจัยและพัฒนา\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No\n2\tอิเล็กทรอนิกส์\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\tและ\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\tคอมพิวเตอร์\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n\n1\tวัน\t_\t_\t_\t_\t0\troot\t0:root\tSpaceAfter=No|NewPar=Yes\n2\tที่ 15\t_\t_\t_\t_\t1\tdep\t1:dep\tSpaceAfter=No\n3\t-\t_\t_\t_\t_\t2\tdep\t2:dep\tSpaceAfter=No\n4\t16\t_\t_\t_\t_\t3\tdep\t3:dep\t_\n5\tสิงหาคม\t_\t_\t_\t_\t4\tdep\t4:dep\t_\n6\t2532\t_\t_\t_\t_\t5\tdep\t5:dep\t_\n'.strip()
EXPECTED_TEXT = 'การประชุมทางวิชาการ ครั้งที่ 1 โครงการวิจัยและพัฒนาอิเล็กทรอนิกส์และคอมพิวเตอร์\n\nวันที่ 15-16 สิงหาคม 2532\n\n'
EXPECTED_LABELS = '0010000010010000001000001000020000000000000000000010000000000000100100000000002\n\n0010000011010000000100002\n\n'

def check_results(documents, expected_conllu, expected_txt, expected_labels):
    if False:
        print('Hello World!')
    with tempfile.TemporaryDirectory() as output_dir:
        write_section(output_dir, 'orchid', 'train', documents)
        with open(os.path.join(output_dir, 'th_orchid.train.gold.conllu')) as fin:
            conllu = fin.read().strip()
        with open(os.path.join(output_dir, 'th_orchid.train.txt')) as fin:
            txt = fin.read()
        with open(os.path.join(output_dir, 'th_orchid-ud-train.toklabels')) as fin:
            labels = fin.read()
        assert conllu == expected_conllu
        assert txt == expected_txt
        assert labels == expected_labels
        assert len(txt) == len(labels)

def test_orchid():
    if False:
        i = 10
        return i + 15
    tree = ET.ElementTree(ET.fromstring(SMALL_DOC))
    documents = parse_xml(tree)
    check_results(documents, EXPECTED_RESULTS, EXPECTED_TEXT, EXPECTED_LABELS)