"""Testing online code for fetching NCBI qblast.

Uses Bio.Blast.NCBIWWW.qblast() to run some online blast queries, get XML
blast results back, and then checks Bio.Blast.NCBIXML.parse() can read them.

Goals:
    - Make sure that all retrieval is working as expected.
    - Make sure we can parse the latest XML format being used by the NCBI.

If an internet connection is available and run_tests.py has not given the argument
'--offline', these tests will run online (and can take a long time). Otherwise, the
module runs offline using a mocked version of urllib.request.urlopen, which returns
file contents instead of internet responses.

IMPORTANT:
If you add new tests (or change existing tests) you must provide new 'mock' xml
files which fulfill the requirements of the respective tests. These need to be
added to the 'response_list' within the 'mock_response' function. Note: The
tests are run in alphabetical order, so you must place your mock file at the
correct position.

"""
import unittest
from unittest import mock
from urllib.error import HTTPError
from io import BytesIO
from Bio import MissingExternalDependencyError
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import requires_internet
NCBIWWW.email = 'biopython@biopython.org'
try:
    requires_internet.check()
except MissingExternalDependencyError:
    pass
if not requires_internet.check.available:

    def mock_response():
        if False:
            return 10
        'Mimic an NCBI qblast response.'
        wait = ['Blast/mock_wait.html']
        response_list = ['Blast/mock_actin.xml', 'Blast/mock_disco.xml', 'Blast/mock_orchid.xml', 'Blast/mock_pcr.xml', 'Blast/mock_short_empty.xml', 'Blast/mock_short_result.xml', 'Blast/mock_short_result.xml']
        responses = (BytesIO(open(a, 'rb').read()) for b in zip(len(response_list) * wait, response_list) for a in b)
        return responses
    NCBIWWW.time.sleep = mock.Mock()
    NCBIWWW.urlopen = mock.Mock(side_effect=mock_response())

class TestQblast(unittest.TestCase):

    def test_blastp_nr_actin(self):
        if False:
            i = 10
            return i + 15
        self.run_qblast('blastp', 'nr', 'NP_075631.2', 0.001, 'rat [ORGN]', {'megablast': 'FALSE'}, ['NP_112408.1', 'AAH59131.1', 'EDM14357.1', 'NP_001008766.1', 'NP_001102411.1', 'EDL80109.1', 'EDL80106.1', 'NP_001100434.1', 'AAI67084.1'])

    def test_pcr_primers(self):
        if False:
            return 10
        self.run_qblast('blastn', 'nr', 'GTACCTTGATTTCGTATTC' + 'N' * 30 + 'GACTCTACTACCTTTACCC', 10, 'pan [ORGN]', {'megablast': 'FALSE'}, ['XM_034941187.1', 'XM_034941186.1', 'XM_034941185.1', 'XM_034941184.1', 'XM_034941183.1', 'XM_034941182.1', 'XM_034941180.1', 'XM_034941179.1', 'XM_034941178.1', 'XM_034941177.1'])

    def test_orchid_est(self):
        if False:
            while True:
                i = 10
        self.run_qblast('blastx', 'nr', ">gi|116660609|gb|EG558220.1|EG558220 CR02019H04 Leaf CR02 cDNA library Catharanthus roseus cDNA clone CR02019H04 5', mRNA sequence\n               CTCCATTCCCTCTCTATTTTCAGTCTAATCAAATTAGAGCTTAAAAGAATGAGATTTTTAACAAATAAAA\n               AAACATAGGGGAGATTTCATAAAAGTTATATTAGTGATTTGAAGAATATTTTAGTCTATTTTTTTTTTTT\n               TCTTTTTTTGATGAAGAAAGGGTATATAAAATCAAGAATCTGGGGTGTTTGTGTTGACTTGGGTCGGGTG\n               TGTATAATTCTTGATTTTTTCAGGTAGTTGAAAAGGTAGGGAGAAAAGTGGAGAAGCCTAAGCTGATATT\n               GAAATTCATATGGATGGAAAAGAACATTGGTTTAGGATTGGATCAAAAAATAGGTGGACATGGAACTGTA\n               CCACTACGTCCTTACTATTTTTGGCCGAGGAAAGATGCTTGGGAAGAACTTAAAACAGTTTTAGAAAGCA\n               AGCCATGGATTTCTCAGAAGAAAATGATTATACTTCTTAATCAGGCAACTGATATTATCAATTTATGGCA\n               GCAGAGTGGTGGCTCCTTGTCCCAGCAGCAGTAATTACTTTTTTTTCTCTTTTTGTTTCCAAATTAAGAA\n               ACATTAGTATCATATGGCTATTTGCTCAATTGCAGATTTCTTTCTTTTGTGAATG", 1e-07, None, {'megablast': 'FALSE'}, ['XP_021665344.1', 'XP_021615158.1', 'XP_017223689.1', 'OMP06800.1', 'XP_021634873.1', 'XP_021299673.1', 'XP_002311451.2', 'XP_021976565.1', 'OMO90244.1'])

    def test_discomegablast(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_qblast('blastn', 'nr', '>some sequence\n               ATGAAGATCTTCCAGATCCAGTGCAGCAGCTTCAAGGAGAGCAGGTGGCAGAAGAGCAAGTGCGACAACT\n               GCCTGAAGTTCCACATCGACATCAACAACAACAGCAAGACCAGCAACACCGACACCGACTTCGACGCCAA\n               CACCAACATCAACAGCAACATCAACAGCAACATCAACAGCAACATCAACATCAACAACAGCGGCAACAAC\n               AACAAGAACAGCAACAACATCGAGATCACCGAGAACATCGACAACAAGGCCAAGATCATCAACAAGCACA\n               TCAAGACCATCACCAACAGCAAGCCCATCCCCATCCCCATCCCCACCCCCACCCCCATCAGCATCAAGGA\n               GAAGGAGAAGGAGAAGGAGAAGGAGAAGGAGAAGGAGAAGGAGAAGGAGAAGGAGAAGGAGAAGGAGATG\n               AAGAGCACCATCAACCTGAGGAGCGAGGACACCACCAGCAACAAGAGCACCATCGTGTTCACCGAGTGCC\n               TGGAGTACAAGGGCCACCAGTGGAGGCCCAACATCTGCGTGACCTGCTTCAGCCCCAAGAACAAGCACAA\n               GAACGTGCTGCCCGAGACCAGCACCCCCCTGATCAGCCAGAGCAGCCAGACCAGCACCATCACCCCCAGC\n               AGCAGCAGCACCAGCACCAGCACCAGCAGCATCAGCACCCACAAGACCGCCAACAACAAGACCGTGATCA\n               CCTACATCAGCAGCACCACCACCACCACCACCACCAGCAGCAGCAGCAGCAGCCCCCCCAGCAGCAGCAT\n               CGCCGGCATCACCAACCCCACCAGCAGGAGCAGCAGCCCCATCCTGAAGAGCGTGCCCCCCAGCGCCTAC\n               AGCAACGTGGTGATCCCCATCAACAACATCAACAACAGCAACAGCAACAGCAGCAGCGGCGGCGGCAACA\n               ACAACAACAAGAGCATCAGCACCCCCAGCAGCCCCATCATCAGCAGGCCCATCACCAACAAGATCAACAA\n               CAACAACAACAACAACCAGCCCCAGCTGCACTACAACCAGCCCCAGAGCAGCAGCGTGAGCACCACCAGC\n               AGCCCCATCATCAGGCCCGTGCTGAGGAGGCAGTTCCAGAGCTTCCCCAGCAACCCCAAGATCAGCAAGG\n               CCATCCTGGAGCAGTGCAACATCATCAACAACAACAGCAACAGCAACAACAGCAACAACAAGGACCCCGT\n               GATCCTGTGCAAGTACACCATCGAGAGCCAGCCCAAGAGCAACATCAGCGTGCTGAAGCCCACCCTGGTG\n               GAGTTCATCAACCAGCCCGACAGCAAGGACGACGAGAGCAGCGTGAAGAGCCCCCCCCTGCCCGTGGAGA\n               GCCAGCCCATCTTCAACAGCAAGCAGAGCGCCACCATGGACGGCATCACCACCCACAAGAGCGTGAGCAT\n               CACCATCAGCACCAGCACCAGCCCCAGCAGCACCACCACCACCACCAGCACCACCACCAGCATCATCGCC\n               GAGGAGCCCAGCAGCCCCATCCTGCCCACCGCCAGCCCCAGCAGCAGCAGCAGCAGCATCATCACCACCG\n               CCACCGCCAGCACCATCCCCATGAGCCCCAGCCTGCCCAGCATCCCCTTCCACGAGTTCGAGACCATGGA\n               GAGCAGCACCACCACCACCCTGCTGAGCGAGAACAACGGCGGCGGCGGCGGCAGCAGCTGCAACGACAAC\n               AGCAGGAGGAACAGCCTGAACATCCTGCCCCTGAGGCTGAAGAGCTTCAGCTTCAGCGCCCCCCAGAGCG\n               ACAGCATGATCGAGCAGCCCGAGGACGACCCCTTCTTCGACTTCGAGGACCTGAGCGACGACGACGACAG\n               CAACGACAACGACGACGAGGAGCTGAAGGAGATCAACGGCGAGAAGATCATCCAGCAGAACGACCTGACC\n               CCCACCACCACCATCACCAGCACCACCACCATCCTGCAGAGCCCCACCCTGGAGAAGACCCTGAGCACCA\n               CCACCACCACCACCATCCCCAGCCCCAGCACCAACAGCAGGAGCATCTGCAACACCCTGATGGACAGCAC\n               CGACAGCATCAACAACACCAACACCAACACCAACACCAACACCAACACCAACACCAACACCAACACCAAC\n               ACCAACACCAACACCAACACCAACGCCAACATCAACAACAAGGTGAGCACCACCACCACCACCACCACCA\n               CCAAGAGGAGGAGCCTGAAGATGGACCAGTTCAAGGAGAAGGAGGACGAGTGGGACCAGGGCGTGGACCT\n               GACCAGCTTCCTGAAGAGGAAGCCCACCCTGCAGAGGGACTTCAGCTACTGCAACAACAAGGTGATGGAG\n               ATCAGCAGCGTGAAGGAGGAGGCCAAGAGGCTGCACGGCGGCACCGGCTACATCCACCAGTTCGCCTTCG\n               AGGCCTTCAAGGACATCCTGGAGGCCAAGCAGACCCAGATCAACAGGGCCTTCTGCAGCCAGAAGATCGA\n               CGCCCCCGACTGCGAGATGCTGATCAACGAGATCAACACCGCCAAGAAGCTGCTGGAGGACCTGCTGGAG\n               CTGAACAGCAACAGCAGCGGCAGCGGCAACAACAGCAACGACAACAGCGGCAGCAGCAGCCCCAGCAGCA\n               GCAAGACCAACACCCTGAACCAGCAGAGCATCTGCATCAAGAGCGAGATCCAACGATACGTTGAAATTCG\n               CTTGTGTGCCACTGGTAAATCCACCCCCCCTAAGCCTCTAATAGGGAGACCTTAG', 1e-07, None, {'template_type': 0, 'template_length': 18, 'megablast': 'on'}, ['XM_635681.1', 'XM_008496783.1'])

    def run_qblast(self, program, database, query, e_value, entrez_filter, additional_args, expected_hits):
        if False:
            while True:
                i = 10
        'Do qblast searches with given parameters and analyze results.'
        try:
            if program == 'blastn':
                handle = NCBIWWW.qblast(program, database, query, alignments=10, descriptions=10, hitlist_size=10, entrez_query=entrez_filter, expect=e_value, **additional_args)
            else:
                handle = NCBIWWW.qblast(program, database, query, alignments=10, descriptions=10, hitlist_size=10, entrez_query=entrez_filter, expect=e_value, **additional_args)
        except HTTPError:
            raise MissingExternalDependencyError('internet connection failed') from None
        record = NCBIXML.read(handle)
        if record.query == 'No definition line':
            self.assertEqual(len(query), record.query_letters)
        elif query.startswith('>'):
            expected = query[1:].split('\n', 1)[0]
            self.assertEqual(expected, record.query)
        elif record.query_id.startswith('Query_') and len(query) == record.query_letters:
            pass
        else:
            self.assertIn(query, record.query_id.split('|'), f'Expected {query!r} within query_id {record.query_id!r}')
        self.assertEqual(float(record.expect), e_value)
        self.assertEqual(record.application.lower(), program)
        self.assertLessEqual(len(record.alignments), 10)
        self.assertLessEqual(len(record.descriptions), 10)
        if expected_hits is None:
            self.assertEqual(len(record.alignments), 0)
        else:
            self.assertGreater(len(record.alignments), 0)
            found_result = False
            for expected_hit in expected_hits:
                for alignment in record.alignments:
                    if expected_hit in alignment.hit_id.split('|'):
                        found_result = True
                        break
            self.assertTrue(found_result, 'Missing all expected hits (%s), instead have: %s' % (', '.join(expected_hits), ', '.join((a.hit_id for a in record.alignments))))
        if expected_hits is None:
            self.assertEqual(len(record.descriptions), 0)
        else:
            self.assertGreater(len(record.descriptions), 0)
            found_result = False
            for expected_hit in expected_hits:
                for descr in record.descriptions:
                    if expected_hit == descr.accession or expected_hit in descr.title.split(None, 1)[0].split('|'):
                        found_result = True
                        break
            msg = f'Missing all of {expected_hit} in descriptions'
            self.assertTrue(found_result, msg=msg)

    def test_parse_qblast_ref_page(self):
        if False:
            while True:
                i = 10
        with open('Blast/html_msgid_29_blastx_001.html', 'rb') as f:
            handle = BytesIO(f.read())
        self.assertRaises(ValueError, NCBIWWW._parse_qblast_ref_page, handle)

    def test_short_query(self):
        if False:
            print('Hello World!')
        'Test SHORT_QUERY_ADJUST parameter.'
        my_search = NCBIWWW.qblast('blastp', 'nr', 'ICWENRMP', hitlist_size=5)
        my_hits = NCBIXML.read(my_search)
        my_search.close()
        self.assertEqual(len(my_hits.alignments), 0)
        my_search = NCBIWWW.qblast('blastp', 'nr', 'ICWENRMP', hitlist_size=5, short_query=True)
        my_hits = NCBIXML.read(my_search)
        my_search.close()
        self.assertNotEqual(len(my_hits.alignments), 0)

    def test_error_conditions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if exceptions were properly handled.'
        self.assertRaises(ValueError, NCBIWWW.qblast, 'megablast', 'nt', 'ATGCGTACGCAGCTAAAGTAAACCTATCGCGTCTCCT')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)