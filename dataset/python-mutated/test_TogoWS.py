"""Testing Bio.TogoWS online code."""
import unittest
from io import StringIO
from urllib.error import HTTPError
from Bio import TogoWS
from Bio import SeqIO
from Bio.SeqUtils.CheckSum import seguid
from Bio import Medline
import requires_internet
requires_internet.check()

class TogoFields(unittest.TestCase):

    def test_invalid_database(self):
        if False:
            i = 10
            return i + 15
        'Check asking for fields of invalid database fails.'
        self.assertRaises(IOError, TogoWS._get_fields, 'http://togows.dbcls.jp/entry/invalid?fields')

    def test_databases(self):
        if False:
            for i in range(10):
                print('nop')
        'Check supported databases.'
        dbs = set(TogoWS._get_entry_dbs())
        expected = {'nuccore', 'nucest', 'nucgss', 'nucleotide', 'protein', 'gene', 'homologene', 'snp', 'mesh', 'pubmed', 'uniprot', 'uniparc', 'uniref100', 'uniref90', 'uniref50', 'ddbj', 'dad', 'pdb', 'compound', 'drug', 'enzyme', 'genes', 'glycan', 'orthology', 'reaction', 'module', 'pathway'}
        self.assertTrue(dbs.issuperset(expected), f"Missing DB: {', '.join(sorted(expected.difference(dbs)))}")

    def test_pubmed(self):
        if False:
            for i in range(10):
                print('nop')
        'Check supported fields for pubmed database.'
        fields = set(TogoWS._get_entry_fields('pubmed'))
        self.assertTrue(fields.issuperset(['abstract', 'au', 'authors', 'doi', 'mesh', 'so', 'title']), fields)

    def test_ncbi_protein(self):
        if False:
            while True:
                i = 10
        'Check supported fields for NCBI protein database.'
        fields = set(TogoWS._get_entry_fields('ncbi-protein'))
        self.assertTrue(fields.issuperset(['entry_id', 'length', 'strand', 'moltype', 'linearity', 'division', 'date', 'definition', 'accession', 'accessions', 'version', 'versions', 'acc_version', 'gi', 'keywords', 'organism', 'common_name', 'taxonomy', 'comment', 'seq']), fields)

    def test_ddbj(self):
        if False:
            i = 10
            return i + 15
        'Check supported fields for ddbj database.'
        fields = set(TogoWS._get_entry_fields('ddbj'))
        self.assertTrue(fields.issuperset(['entry_id', 'length', 'strand', 'moltype', 'linearity', 'division', 'date', 'definition', 'accession', 'accessions', 'version', 'versions', 'acc_version', 'gi', 'keywords', 'organism', 'common_name', 'taxonomy', 'comment', 'seq']), fields)

    def test_uniprot(self):
        if False:
            i = 10
            return i + 15
        'Check supported fields for uniprot database.'
        fields = set(TogoWS._get_entry_fields('uniprot'))
        self.assertTrue(fields.issuperset(['definition', 'entry_id', 'seq']), fields)

    def test_pdb(self):
        if False:
            while True:
                i = 10
        'Check supported fields for pdb database.'
        fields = set(TogoWS._get_entry_fields('pdb'))
        self.assertTrue(fields.issuperset(['accession', 'chains', 'keywords', 'models']), fields)

class TogoEntry(unittest.TestCase):

    def test_pubmed_16381885(self):
        if False:
            for i in range(10):
                print('nop')
        'Bio.TogoWS.entry("pubmed", "16381885").'
        handle = TogoWS.entry('pubmed', '16381885')
        data = Medline.read(handle)
        handle.close()
        self.assertEqual(data['TI'], 'From genomics to chemical genomics: new developments in KEGG.')
        self.assertEqual(data['AU'], ['Kanehisa M', 'Goto S', 'Hattori M', 'Aoki-Kinoshita KF', 'Itoh M', 'Kawashima S', 'Katayama T', 'Araki M', 'Hirakawa M'])

    def test_pubmed_16381885_ti(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("pubmed", "16381885", field="title").'
        handle = TogoWS.entry('pubmed', '16381885', field='title')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, 'From genomics to chemical genomics: new developments in KEGG.')

    def test_pubmed_16381885_title(self):
        if False:
            return 10
        'Bio.TogoWS.entry("pubmed", "16381885", field="title").'
        handle = TogoWS.entry('pubmed', '16381885', field='title')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, 'From genomics to chemical genomics: new developments in KEGG.')

    def test_pubmed_16381885_au(self):
        if False:
            while True:
                i = 10
        'Bio.TogoWS.entry("pubmed", "16381885", field="au").'
        handle = TogoWS.entry('pubmed', '16381885', field='au')
        data = handle.read().strip().split('\n')
        handle.close()
        self.assertEqual(data, ['Kanehisa M', 'Goto S', 'Hattori M', 'Aoki-Kinoshita KF', 'Itoh M', 'Kawashima S', 'Katayama T', 'Araki M', 'Hirakawa M'])

    def test_pubmed_16381885_authors(self):
        if False:
            for i in range(10):
                print('nop')
        'Bio.TogoWS.entry("pubmed", "16381885", field="authors").'
        handle = TogoWS.entry('pubmed', '16381885', field='authors')
        data = handle.read().strip().split('\t')
        handle.close()
        self.assertEqual(data, ['Kanehisa, M.', 'Goto, S.', 'Hattori, M.', 'Aoki-Kinoshita, K. F.', 'Itoh, M.', 'Kawashima, S.', 'Katayama, T.', 'Araki, M.', 'Hirakawa, M.'])

    def test_pubmed_16381885_invalid_field(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("pubmed", "16381885", field="invalid_for_testing").'
        self.assertRaises(ValueError, TogoWS.entry, 'pubmed', '16381885', field='invalid_for_testing')

    def test_pubmed_16381885_invalid_format(self):
        if False:
            while True:
                i = 10
        'Bio.TogoWS.entry("pubmed", "16381885", format="invalid_for_testing").'
        self.assertRaises(ValueError, TogoWS.entry, 'pubmed', '16381885', format='invalid_for_testing')

    def test_pubmed_invalid_id(self):
        if False:
            for i in range(10):
                print('nop')
        'Bio.TogoWS.entry("pubmed", "invalid_for_testing").'
        self.assertRaises(IOError, TogoWS.entry, 'pubmed', 'invalid_for_testing')

    def test_pubmed_16381885_and_19850725(self):
        if False:
            for i in range(10):
                print('nop')
        'Bio.TogoWS.entry("pubmed", "16381885,19850725").'
        handle = TogoWS.entry('pubmed', '16381885,19850725')
        records = list(Medline.parse(handle))
        handle.close()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['TI'], 'From genomics to chemical genomics: new developments in KEGG.')
        self.assertEqual(records[0]['AU'], ['Kanehisa M', 'Goto S', 'Hattori M', 'Aoki-Kinoshita KF', 'Itoh M', 'Kawashima S', 'Katayama T', 'Araki M', 'Hirakawa M'])
        self.assertEqual(records[1]['TI'], 'DDBJ launches a new archive database with analytical tools for next-generation sequence data.')
        self.assertEqual(records[1]['AU'], ['Kaminuma E', 'Mashima J', 'Kodama Y', 'Gojobori T', 'Ogasawara O', 'Okubo K', 'Takagi T', 'Nakamura Y'])

    def test_pubmed_16381885_and_19850725_authors(self):
        if False:
            return 10
        'Bio.TogoWS.entry("pubmed", "16381885,19850725", field="authors").'
        handle = TogoWS.entry('pubmed', '16381885,19850725', field='authors')
        names = handle.read().strip().split('\n')
        handle.close()
        self.assertEqual(2, len(names))
        (names1, names2) = names
        self.assertEqual(names1.split('\t'), ['Kanehisa, M.', 'Goto, S.', 'Hattori, M.', 'Aoki-Kinoshita, K. F.', 'Itoh, M.', 'Kawashima, S.', 'Katayama, T.', 'Araki, M.', 'Hirakawa, M.'])
        self.assertEqual(names2.split('\t'), ['Kaminuma, E.', 'Mashima, J.', 'Kodama, Y.', 'Gojobori, T.', 'Ogasawara, O.', 'Okubo, K.', 'Takagi, T.', 'Nakamura, Y.'])

    def test_invalid_db(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("invalid_db", "invalid_id").'
        self.assertRaises(ValueError, TogoWS.entry, 'invalid_db', 'invalid_id')

    def test_ddbj_genbank_length(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("ddbj", "X52960", field="length").'
        handle = TogoWS.entry('ddbj', 'X52960', field='length')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, '248')

    def test_ddbj_genbank(self):
        if False:
            return 10
        'Bio.TogoWS.entry("ddbj", "X52960").'
        handle = TogoWS.entry('ddbj', 'X52960')
        record = SeqIO.read(handle, 'gb')
        handle.close()
        self.assertEqual(record.id, 'X52960.1')
        self.assertEqual(record.name, 'X52960')
        self.assertEqual(len(record), 248)
        self.assertEqual(seguid(record.seq), 'Ktxz0HgMlhQmrKTuZpOxPZJ6zGU')

    def test_nucleotide_genbank_length(self):
        if False:
            while True:
                i = 10
        'Bio.TogoWS.entry("nucleotide", "X52960", field="length").'
        handle = TogoWS.entry('nucleotide', 'X52960', field='length')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, '248')

    def test_nucleotide_genbank_seq(self):
        if False:
            while True:
                i = 10
        'Bio.TogoWS.entry("nucleotide", "X52960", field="seq").'
        handle = TogoWS.entry('nucleotide', 'X52960', field='seq')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(seguid(data), 'Ktxz0HgMlhQmrKTuZpOxPZJ6zGU')

    def test_nucleotide_genbank_definition(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("nucleotide", "X52960", field="definition").'
        handle = TogoWS.entry('nucleotide', 'X52960', field='definition')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, 'Coleus blumei viroid 1 (CbVd) RNA.')

    def test_nucleotide_genbank_accession(self):
        if False:
            i = 10
            return i + 15
        'Bio.TogoWS.entry("nucleotide", "X52960", field="accession").'
        handle = TogoWS.entry('nucleotide', 'X52960', field='accession')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, 'X52960')

    def test_nucleotide_genbank_version(self):
        if False:
            for i in range(10):
                print('nop')
        'Bio.TogoWS.entry("nucleotide", "X52960", field="version").'
        handle = TogoWS.entry('nucleotide', 'X52960', field='version')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, '1')

    def test_nucleotide_genbank_acc_version(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("nucleotide", "X52960", field="acc_version").'
        handle = TogoWS.entry('nucleotide', 'X52960', field='acc_version')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, 'X52960.1')

    def test_nucleotide_genbank_organism(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("nucleotide", "X52960", field="organism").'
        handle = TogoWS.entry('nucleotide', 'X52960', field='organism')
        data = handle.read().strip()
        handle.close()
        self.assertEqual(data, 'Coleus blumei viroid 1')

    def test_ddbj_genbank_invalid_field(self):
        if False:
            for i in range(10):
                print('nop')
        'Bio.TogoWS.entry("nucleotide", "X52960", field="invalid_for_testing").'
        self.assertRaises(ValueError, TogoWS.entry, 'nucleotide', 'X52960', field='invalid_for_testing')

    def test_nucleotide_invalid_format(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("nucleotide", "X52960", format="invalid_for_testing").'
        self.assertRaises(ValueError, TogoWS.entry, 'nucleotide', 'X52960', format='invalid_for_testing')

    def test_ddbj_gff3(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.entry("ddbj", "X52960", format="gff").'
        handle = TogoWS.entry('ddbj', 'X52960', format='gff')
        data = handle.read()
        handle.close()
        self.assertTrue(data.startswith('##gff-version 3\nX52960\tDDBJ\t'), data)

    def test_genbank_gff3(self):
        if False:
            for i in range(10):
                print('nop')
        'Bio.TogoWS.entry("nucleotide", "X52960", format="gff").'
        handle = TogoWS.entry('nucleotide', 'X52960', format='gff')
        data = handle.read()
        handle.close()
        self.assertTrue(data.startswith('##gff-version 3\nX52960\tGenbank\t'), data)

    def test_ddbj_fasta(self):
        if False:
            i = 10
            return i + 15
        'Bio.TogoWS.entry("ddbj", "X52960", "fasta").'
        handle = TogoWS.entry('ddbj', 'X52960', 'fasta')
        record = SeqIO.read(handle, 'fasta')
        handle.close()
        self.assertIn('X52960', record.id)
        self.assertIn('X52960', record.name)
        self.assertEqual(len(record), 248)
        self.assertEqual(seguid(record.seq), 'Ktxz0HgMlhQmrKTuZpOxPZJ6zGU')

    def test_nucleotide_fasta(self):
        if False:
            return 10
        'Bio.TogoWS.entry("nucleotide", "6273291", "fasta").'
        handle = TogoWS.entry('nucleotide', '6273291', 'fasta')
        record = SeqIO.read(handle, 'fasta')
        handle.close()
        self.assertIn('AF191665.1', record.id)
        self.assertIn('AF191665.1', record.name)
        self.assertEqual(len(record), 902)
        self.assertEqual(seguid(record.seq), 'bLhlq4mEFJOoS9PieOx4nhGnjAQ')

    def test_protein_fasta(self):
        if False:
            i = 10
            return i + 15
        'Bio.TogoWS.entry("protein", "16130152", "fasta").'
        handle = TogoWS.entry('protein', '16130152', 'fasta')
        record = SeqIO.read(handle, 'fasta')
        handle.close()
        self.assertIn('NP_416719.1', record.id)
        self.assertIn('NP_416719.1', record.name)
        self.assertIn(' porin ', record.description)
        self.assertEqual(len(record), 367)
        self.assertEqual(seguid(record.seq), 'fCjcjMFeGIrilHAn6h+yju267lg')

class TogoSearch(unittest.TestCase):
    """Search tests."""

    def test_bad_args_just_limit(self):
        if False:
            for i in range(10):
                print('nop')
        'Reject Bio.TogoWS.search(...) with just limit.'
        self.assertRaises(ValueError, TogoWS.search, 'pubmed', 'lung+cancer', limit=10)

    def test_bad_args_just_offset(self):
        if False:
            print('Hello World!')
        'Reject Bio.TogoWS.search(...) with just offset.'
        self.assertRaises(ValueError, TogoWS.search, 'pubmed', 'lung+cancer', offset=10)

    def test_bad_args_zero_limit(self):
        if False:
            for i in range(10):
                print('nop')
        'Reject Bio.TogoWS.search(...) with zero limit.'
        self.assertRaises(ValueError, TogoWS.search, 'pubmed', 'lung+cancer', offset=1, limit=0)

    def test_bad_args_zero_offset(self):
        if False:
            while True:
                i = 10
        'Reject Bio.TogoWS.search(...) with zero offset.'
        self.assertRaises(ValueError, TogoWS.search, 'pubmed', 'lung+cancer', offset=0, limit=10)

    def test_bad_args_non_int_offset(self):
        if False:
            print('Hello World!')
        'Reject Bio.TogoWS.search(...) with non-integer offset.'
        self.assertRaises(ValueError, TogoWS.search, 'pubmed', 'lung+cancer', offset='test', limit=10)

    def test_bad_args_non_int_limit(self):
        if False:
            print('Hello World!')
        'Reject Bio.TogoWS.search(...) with non-integer limit.'
        self.assertRaises(ValueError, TogoWS.search, 'pubmed', 'lung+cancer', offset=1, limit='lots')

    def test_pubmed_search_togows(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.search_iter("pubmed", "TogoWS") etc.'
        self.check('pubmed', 'TogoWS', ['20472643'])

    def test_pubmed_search_bioruby(self):
        if False:
            print('Hello World!')
        'Bio.TogoWS.search_iter("pubmed", "BioRuby") etc.'
        self.check('pubmed', 'BioRuby', ['22994508', '22399473', '20739307', '20015970', '14693808'])

    def test_pubmed_search_porin(self):
        if False:
            return 10
        'Bio.TogoWS.search_iter("pubmed", "human porin") etc.\n\n        Count was 357 at time of writing, this was chosen to\n        be larger than the default chunk size for iteration,\n        but still not too big to download the full list.\n        '
        self.check('pubmed', 'human porin', ['21189321', '21835183'])

    def test_uniprot_search_lung_cancer(self):
        if False:
            i = 10
            return i + 15
        'Bio.TogoWS.search_iter("uniprot", "terminal+lung+cancer", limit=150) etc.\n\n        Search count was 211 at time of writing, a bit large to\n        download all the results in a unit test. Want to use a limit\n        larger than the batch size (100) to ensure at least two\n        batches.\n        '
        self.check('uniprot', 'terminal+lung+cancer', limit=150)

    def check(self, database, search_term, expected_matches=(), limit=None):
        if False:
            i = 10
            return i + 15
        if expected_matches and limit:
            raise ValueError('Bad test - TogoWS makes no promises about order')
        try:
            search_count = TogoWS.search_count(database, search_term)
        except HTTPError as err:
            raise ValueError(f'{err} from {err.url}') from None
        if expected_matches:
            self.assertGreaterEqual(search_count, len(expected_matches))
        if search_count > 5000 and (not limit):
            print('%i results, skipping' % search_count)
            return
        if limit:
            count = min(search_count, limit)
        else:
            count = search_count
        search_iter = list(TogoWS.search_iter(database, search_term, limit))
        self.assertEqual(count, len(search_iter))
        for match in expected_matches:
            self.assertIn(match, search_iter, f'Expected {match} in results')

class TogoConvert(unittest.TestCase):
    """Conversion tests."""

    def test_invalid_format(self):
        if False:
            i = 10
            return i + 15
        'Check convert file format checking.'
        self.assertRaises(ValueError, TogoWS.convert, StringIO('PLACEHOLDER'), 'genbank', 'invalid_for_testing')
        self.assertRaises(ValueError, TogoWS.convert, StringIO('PLACEHOLDER'), 'invalid_for_testing', 'fasta')

    def test_genbank_to_fasta(self):
        if False:
            while True:
                i = 10
        'Conversion of GenBank to FASTA.'
        filename = 'GenBank/NC_005816.gb'
        old = SeqIO.read(filename, 'gb')
        with open(filename) as handle:
            new = SeqIO.read(TogoWS.convert(handle, 'genbank', 'fasta'), 'fasta')
        self.assertEqual(old.seq, new.seq)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)