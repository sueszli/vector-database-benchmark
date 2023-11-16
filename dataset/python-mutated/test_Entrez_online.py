"""Tests for online Entrez access.

This file include tests for accessing the online Entrez service and parsing the
returned results. Note that we are merely testing the access and whether the
results are parseable. Detailed tests on each Entrez service are not within the
scope of this file as they are already covered in test_Entrez.py.

"""
from Bio import Entrez
from Bio import Medline
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import doctest
import sys
import unittest
import requires_internet
requires_internet.check()
Entrez.email = 'biopython@biopython.org'
Entrez.api_key = '5cfd4026f9df285d6cfc723c662d74bcbe09'
URL_HEAD = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

class EntrezOnlineCase(unittest.TestCase):

    def test_no_api_key(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Entrez.read without API key.'
        cached = Entrez.api_key
        Entrez.api_key = None
        try:
            stream = Entrez.einfo()
        finally:
            Entrez.api_key = cached
        self.assertNotIn('api_key=', stream.url)
        rec = Entrez.read(stream)
        stream.close()
        self.assertIsInstance(rec, dict)
        self.assertIn('DbList', rec)
        self.assertGreater(len(rec['DbList']), 5)

    def test_read_from_url(self):
        if False:
            print('Hello World!')
        'Test Entrez.read from URL.'
        stream = Entrez.einfo()
        rec = Entrez.read(stream)
        stream.close()
        self.assertIsInstance(rec, dict)
        self.assertIn('DbList', rec)
        self.assertGreater(len(rec['DbList']), 5)

    def test_parse_from_url(self):
        if False:
            return 10
        'Test Entrez.parse from URL.'
        stream = Entrez.efetch(db='protein', id='15718680, 157427902, 119703751', retmode='xml')
        recs = list(Entrez.parse(stream))
        stream.close()
        self.assertEqual(3, len(recs))
        self.assertTrue((all(len(rec).keys > 5) for rec in recs))

    def test_webenv_search(self):
        if False:
            print('Hello World!')
        'Test Entrez.search from link webenv history.'
        stream = Entrez.elink(db='nucleotide', dbfrom='protein', id='22347800,48526535', webenv=None, query_key=None, cmd='neighbor_history')
        recs = Entrez.read(stream)
        stream.close()
        record = recs.pop()
        webenv = record['WebEnv']
        query_key = record['LinkSetDbHistory'][0]['QueryKey']
        stream = Entrez.esearch(db='nucleotide', term=None, retstart=0, retmax=10, webenv=webenv, query_key=query_key, usehistory='y')
        search_record = Entrez.read(stream)
        stream.close()
        self.assertEqual(2, len(search_record['IdList']))

    def test_seqio_from_url(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Entrez into SeqIO.read from URL.'
        stream = Entrez.efetch(db='nucleotide', id='186972394', rettype='gb', retmode='text')
        record = SeqIO.read(stream, 'genbank')
        stream.close()
        self.assertIsInstance(record, SeqRecord)
        self.assertEqual('EU490707.1', record.id)
        self.assertEqual(1302, len(record))

    def test_medline_from_url(self):
        if False:
            while True:
                i = 10
        'Test Entrez into Medline.read from URL.'
        stream = Entrez.efetch(db='pubmed', id='19304878', rettype='medline', retmode='text')
        record = Medline.read(stream)
        stream.close()
        self.assertIsInstance(record, dict)
        self.assertEqual('19304878', record['PMID'])
        self.assertEqual('10.1093/bioinformatics/btp163 [doi]', record['LID'])

    def test_efetch_taxonomy_xml(self):
        if False:
            i = 10
            return i + 15
        'Test Entrez using a integer id - like a taxon id.'
        stream = Entrez.efetch(db='taxonomy', id=3702, retmode='XML')
        taxon_record = Entrez.read(stream)
        self.assertTrue(1, len(taxon_record))
        self.assertIn('TaxId', taxon_record[0])
        self.assertTrue('3702', taxon_record[0]['TaxId'])
        stream.close()

    def test_elink(self):
        if False:
            return 10
        'Test Entrez.elink with multiple ids, both comma separated and as list.\n\n        This is tricky because the ELink tool treats the "id" parameter\n        differently than the others (see docstring of the elink() function).\n        '
        params = {'db': 'gene', 'dbfrom': 'protein'}
        ids = ['15718680', '157427902', '119703751']
        with Entrez.elink(id=ids, **params) as stream:
            result1 = Entrez.read(stream)
        self.assertEqual(len(result1), len(ids))
        id_map = {}
        for linkset in result1:
            (from_id,) = linkset['IdList']
            (linksetdb,) = linkset['LinkSetDb']
            to_ids = [link['Id'] for link in linksetdb['Link']]
            self.assertGreater(len(to_ids), 0)
            id_map[from_id] = to_ids
        self.assertCountEqual(id_map.keys(), ids)
        with Entrez.elink(id=','.join(ids), **params) as stream:
            result2 = Entrez.read(stream)
        (linkset,) = result2
        self.assertCountEqual(linkset['IdList'], ids)
        (linksetdb,) = linkset['LinkSetDb']
        to_ids = [link['Id'] for link in linksetdb['Link']]
        prev_to_ids = set().union(*id_map.values())
        self.assertCountEqual(to_ids, prev_to_ids)

    def test_epost(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Entrez.epost with multiple ids, both comma separated and as list.'
        stream = Entrez.epost('nuccore', id='186972394,160418')
        stream.close()
        stream = Entrez.epost('nuccore', id=['160418', '160351'])
        stream.close()

    def test_egquery(self):
        if False:
            return 10
        'Test Entrez.egquery.\n\n        which searches in all Entrez databases for a single text query.\n        '
        stream = Entrez.egquery(term='biopython')
        record = Entrez.read(stream)
        stream.close()
        done = False
        for row in record['eGQueryResult']:
            if 'pmc' in row['DbName']:
                self.assertGreater(int(row['Count']), 60)
                done = True
        self.assertTrue(done)

    def test_espell(self):
        if False:
            return 10
        'Test misspellings with Entrez.espell.'
        stream = Entrez.espell(term='biopythooon')
        record = Entrez.read(stream)
        stream.close()
        self.assertEqual(record['Query'], 'biopythooon')
        self.assertEqual(record['CorrectedQuery'], 'biopython')

    def test_ecitmatch(self):
        if False:
            while True:
                i = 10
        'Test Entrez.ecitmatch to search for a citation.'
        citation = {'journal_title': 'proc natl acad sci u s a', 'year': '1991', 'volume': '88', 'first_page': '3248', 'author_name': 'mann bj', 'key': 'citation_1'}
        stream = Entrez.ecitmatch(db='pubmed', bdata=[citation])
        result = stream.read()
        expected_result = 'proc natl acad sci u s a|1991|88|3248|mann bj|citation_1|2014248\n'
        self.assertEqual(result, expected_result)
        stream.close()

    def test_efetch_ids(self):
        if False:
            i = 10
            return i + 15
        'Test different options to supply ids.'
        id_sets = [[15718680, 157427902], [15718680]]
        id_vals = [[[15718680, 157427902], (15718680, 157427902), {15718680, 157427902}, ['15718680', '157427902'], ('15718680', '157427902'), {15718680, '157427902'}, '15718680, 157427902'], [[15718680], 15718680, {15718680}, 15718680, '15718680', '15718680,']]
        for (ids, vals) in zip(id_sets, id_vals):
            for _id in vals:
                with Entrez.efetch(db='protein', id=_id, retmode='xml') as stream:
                    recs = list(Entrez.parse(stream))
                rec_ids = [int(seqid[3:]) for rec in recs for seqid in rec['GBSeq_other-seqids'] if seqid.startswith('gi|')]
                self.assertCountEqual(rec_ids, ids)

    def test_efetch_gds_utf8(self):
        if False:
            while True:
                i = 10
        'Test correct handling of encodings in Entrez.efetch.'
        stream = Entrez.efetch(db='gds', id='200079209')
        text = stream.read()
        expected_phrase = '“field of injury”'
        self.assertEqual(text[342:359], expected_phrase)
        stream.close()

    def test_fetch_xml_schemas(self):
        if False:
            print('Hello World!')
        stream = Entrez.efetch('protein', id='783730874', rettype='ipg', retmode='xml')
        record = Entrez.read(stream, validate=False)
        stream.close()
        self.assertEqual(len(record), 1)
        self.assertIn('IPGReport', record)
        self.assertIn('Product', record['IPGReport'])
        self.assertIn('Statistics', record['IPGReport'])
        self.assertIn('ProteinList', record['IPGReport'])
if __name__ == '__main__':
    unittest_suite = unittest.TestLoader().loadTestsFromName('test_Entrez_online')
    doctest_suite = doctest.DocTestSuite(Entrez)
    suite = unittest.TestSuite((unittest_suite, doctest_suite))
    runner = unittest.TextTestRunner(sys.stdout, verbosity=2)
    runner.run(suite)