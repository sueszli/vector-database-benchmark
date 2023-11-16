"""
Tests the conversion code for the SUC3 NER dataset
"""
import os
import tempfile
from zipfile import ZipFile
import pytest
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]
import stanza.utils.datasets.ner.suc_conll_to_iob as suc_conll_to_iob
TEST_CONLL = '\n1\tDen\tden\tPN\tPN\tUTR|SIN|DEF|SUB/OBJ\t_\t_\t_\t_\tO\t_\tac01b-030:2328\n2\tGud\tGud\tPM\tPM\tNOM\t_\t_\t_\t_\tB\tmyth\tac01b-030:2329\n3\tgiver\tgiva\tVB\tVB\tPRS|AKT\t_\t_\t_\t_\tO\t_\tac01b-030:2330\n4\tämbetet\tämbete\tNN\tNN\tNEU|SIN|DEF|NOM\t_\t_\t_\t_\tO\t_\tac01b-030:2331\n5\tfår\tfå\tVB\tVB\tPRS|AKT\t_\t_\t_\t_\tO\t_\tac01b-030:2332\n6\tockså\tockså\tAB\tAB\t\t_\t_\t_\t_\tO\t_\tac01b-030:2333\n7\tförståndet\tförstånd\tNN\tNN\tNEU|SIN|DEF|NOM\t_\t_\t_\t_\tO\t_\tac01b-030:2334\n8\t.\t.\tMAD\tMAD\t\t_\t_\t_\t_\tO\t_\tac01b-030:2335\n\n1\tHan\than\tPN\tPN\tUTR|SIN|DEF|SUB\t_\t_\t_\t_\tO\t_\taa01a-017:227\n2\tberättar\tberätta\tVB\tVB\tPRS|AKT\t_\t_\t_\t_\tO\t_\taa01a-017:228\n3\tanekdoten\tanekdot\tNN\tNN\tUTR|SIN|DEF|NOM\t_\t_\t_\t_\tO\t_\taa01a-017:229\n4\tsom\tsom\tHP\tHP\t-|-|-\t_\t_\t_\t_\tO\t_\taa01a-017:230\n5\tFN-medlaren\tFN-medlare\tNN\tNN\tUTR|SIN|DEF|NOM\t_\t_\t_\t_\tO\t_\taa01a-017:231\n6\tBrian\tBrian\tPM\tPM\tNOM\t_\t_\t_\t_\tB\tperson\taa01a-017:232\n7\tUrquhart\tUrquhart\tPM\tPM\tNOM\t_\t_\t_\t_\tI\tperson\taa01a-017:233\n8\tmyntat\tmynta\tVB\tVB\tSUP|AKT\t_\t_\t_\t_\tO\t_\taa01a-017:234\n9\t:\t:\tMAD\tMAD\t\t_\t_\t_\t_\tO\t_\taa01a-017:235\n'
EXPECTED_IOB = '\nDen\tO\nGud\tB-myth\ngiver\tO\nämbetet\tO\nfår\tO\nockså\tO\nförståndet\tO\n.\tO\n\nHan\tO\nberättar\tO\nanekdoten\tO\nsom\tO\nFN-medlaren\tO\nBrian\tB-person\nUrquhart\tI-person\nmyntat\tO\n:\tO\n'

def test_read_zip():
    if False:
        i = 10
        return i + 15
    '\n    Test creating a fake zip file, then converting it to an .iob file\n    '
    with tempfile.TemporaryDirectory() as tempdir:
        zip_name = os.path.join(tempdir, 'test.zip')
        in_filename = 'conll'
        with ZipFile(zip_name, 'w') as zout:
            with zout.open(in_filename, 'w') as fout:
                fout.write(TEST_CONLL.encode())
        out_filename = 'iob'
        num = suc_conll_to_iob.extract_from_zip(zip_name, in_filename, out_filename)
        assert num == 2
        with open(out_filename) as fin:
            result = fin.read()
        assert EXPECTED_IOB.strip() == result.strip()

def test_read_raw():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test a direct text file conversion w/o the zip file\n    '
    with tempfile.TemporaryDirectory() as tempdir:
        in_filename = os.path.join(tempdir, 'test.txt')
        with open(in_filename, 'w', encoding='utf-8') as fout:
            fout.write(TEST_CONLL)
        out_filename = 'iob'
        with open(in_filename, encoding='utf-8') as fin, open(out_filename, 'w', encoding='utf-8') as fout:
            num = suc_conll_to_iob.extract(fin, fout)
        assert num == 2
        with open(out_filename) as fin:
            result = fin.read()
        assert EXPECTED_IOB.strip() == result.strip()