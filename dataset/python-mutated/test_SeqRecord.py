"""SeqFeature related tests for SeqRecord objects from Bio.SeqIO.

Initially this takes matched tests of GenBank and FASTA files from the NCBI
and confirms they are consistent using our different parsers.
"""
import unittest
try:
    import numpy
except ImportError:
    numpy = None
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio.SeqFeature import AfterPosition
from Bio.SeqFeature import BeforePosition
from Bio.SeqFeature import ExactPosition
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import OneOfPosition
from Bio.SeqFeature import SeqFeature
from Bio.SeqFeature import WithinPosition
from Bio.SeqRecord import SeqRecord

class SeqRecordCreation(unittest.TestCase):
    """Test basic creation of SeqRecords."""

    def test_annotations(self):
        if False:
            return 10
        'Pass in annotations to SeqRecords.'
        rec = SeqRecord(Seq('ACGT'), id='Test', name='Test', description='Test')
        self.assertEqual(rec.annotations, {})
        rec = SeqRecord(Seq('ACGT'), id='Test', name='Test', description='Test', annotations={'test': ['a test']})
        self.assertEqual(rec.annotations['test'], ['a test'])

    def test_letter_annotations(self):
        if False:
            print('Hello World!')
        'Pass in letter annotations to SeqRecords.'
        rec = SeqRecord(Seq('ACGT'), id='Test', name='Test', description='Test')
        self.assertEqual(rec.annotations, {})
        rec = SeqRecord(Seq('ACGT'), id='Test', name='Test', description='Test', letter_annotations={'test': [1, 2, 3, 4]})
        self.assertEqual(rec.letter_annotations['test'], [1, 2, 3, 4])
        try:
            rec.letter_annotations['bad'] = 'abc'
            self.fail('Adding a bad letter_annotation should fail!')
        except (TypeError, ValueError) as e:
            pass
        rec = SeqRecord(Seq('ACGT'), id='Test', name='Test', description='Test')
        try:
            rec.letter_annotations = {'test': [1, 2, 3]}
            self.fail('Changing to bad letter_annotations should fail!')
        except (TypeError, ValueError) as e:
            pass
        try:
            rec = SeqRecord(Seq('ACGT'), id='Test', name='Test', description='Test', letter_annotations={'test': [1, 2, 3]})
            self.fail('Wrong length letter_annotations should fail!')
        except (TypeError, ValueError) as e:
            pass

    def test_replacing_seq(self):
        if False:
            while True:
                i = 10
        'Replacing .seq if .letter_annotation present.'
        rec = SeqRecord(Seq('ACGT'), id='Test', name='Test', description='Test', letter_annotations={'example': [1, 2, 3, 4]})
        try:
            rec.seq = Seq('ACGTACGT')
            self.fail('Changing .seq length with letter_annotations present should fail!')
        except ValueError as e:
            self.assertEqual(str(e), 'You must empty the letter annotations first!')
        self.assertEqual(rec.seq, 'ACGT')
        self.assertEqual(rec.letter_annotations, {'example': [1, 2, 3, 4]})
        rec.seq = Seq('NNNN')
        self.assertEqual(rec.seq, 'NNNN')
        self.assertEqual(rec.letter_annotations, {'example': [1, 2, 3, 4]})

    def test_valid_id(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            SeqRecord(Seq('ACGT'), id={})

    def test_valid_name(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            SeqRecord(Seq('ACGT'), name={})

    def test_valid_description(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            SeqRecord(Seq('ACGT'), description={})

    def test_valid_dbxrefs(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            SeqRecord(Seq('ACGT'), dbxrefs={})

    def test_valid_annotations(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            SeqRecord(Seq('ACGT'), annotations=[])

    def test_valid_features(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            SeqRecord(Seq('ACGT'), features={})

class SeqRecordMethods(unittest.TestCase):
    """Test SeqRecord methods."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        f0 = SeqFeature(SimpleLocation(0, 26), type='source', qualifiers={'mol_type': ['fake protein']})
        f1 = SeqFeature(SimpleLocation(0, ExactPosition(10)))
        f2 = SeqFeature(SimpleLocation(WithinPosition(12, left=12, right=15), BeforePosition(22)))
        f3 = SeqFeature(SimpleLocation(AfterPosition(16), OneOfPosition(26, [ExactPosition(25), AfterPosition(26)])))
        self.record = SeqRecord(Seq('ABCDEFGHIJKLMNOPQRSTUVWZYX'), id='TestID', name='TestName', description='TestDescr', dbxrefs=['TestXRef'], annotations={'k': 'v'}, letter_annotations={'fake': 'X' * 26}, features=[f0, f1, f2, f3])

    def test_iter(self):
        if False:
            i = 10
            return i + 15
        for amino in self.record:
            self.assertEqual('A', amino)
            break

    def test_contains(self):
        if False:
            i = 10
            return i + 15
        self.assertIn(Seq('ABC'), self.record)

    def test_bytes(self):
        if False:
            return 10
        self.assertEqual(b'ABCDEFGHIJKLMNOPQRSTUVWZYX', bytes(self.record))

    def test_str(self):
        if False:
            while True:
                i = 10
        expected = "\nID: TestID\nName: TestName\nDescription: TestDescr\nDatabase cross-references: TestXRef\nNumber of features: 4\n/k=v\nPer letter annotation for: fake\nSeq('ABCDEFGHIJKLMNOPQRSTUVWZYX')"
        self.assertEqual(expected.lstrip(), str(self.record))

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        expected = "SeqRecord(seq=Seq('ABCDEFGHIJKLMNOPQRSTUVWZYX'), id='TestID', name='TestName', description='TestDescr', dbxrefs=['TestXRef'])"
        self.assertEqual(expected, repr(self.record))

    def test_format(self):
        if False:
            print('Hello World!')
        expected = '>TestID TestDescr\nABCDEFGHIJKLMNOPQRSTUVWZYX\n'
        self.assertEqual(expected, self.record.format('fasta'))

    def test_format_str(self):
        if False:
            print('Hello World!')
        expected = '>TestID TestDescr\nABCDEFGHIJKLMNOPQRSTUVWZYX\n'
        self.assertEqual(expected, f'{self.record:fasta}')

    def test_format_str_binary(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'Binary format sff cannot be used with SeqRecord format method'):
            f'{self.record:sff}'

    def test_format_spaces(self):
        if False:
            print('Hello World!')
        rec = SeqRecord(Seq('ABCDEFGHIJKLMNOPQRSTUVWZYX'), id='TestID', name='TestName', description='TestDescr')
        rec.description = 'TestDescr     with5spaces'
        expected = '>TestID TestDescr     with5spaces\nABCDEFGHIJKLMNOPQRSTUVWZYX\n'
        self.assertEqual(expected, rec.format('fasta'))

    def test_count(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.record.count('HIJK'), 1)
        self.assertRaises(TypeError, SeqRecord(Seq('AC777GT')).count, 7)
        self.assertRaises(TypeError, SeqRecord(Seq('AC777GT')).count, None)

    def test_upper(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('ABCDEFGHIJKLMNOPQRSTUVWZYX', self.record.lower().upper().seq)

    def test_lower(self):
        if False:
            return 10
        self.assertEqual('abcdefghijklmnopqrstuvwzyx', self.record.lower().seq)

    def test_isupper(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.record.isupper())
        self.assertFalse(self.record.lower().isupper())

    def test_islower(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.record.islower())
        self.assertTrue(self.record.lower().islower())

    def test_slicing(self):
        if False:
            while True:
                i = 10
        self.assertEqual('B', self.record[1])
        self.assertEqual('BC', self.record[1:3].seq)
        with self.assertRaises(ValueError):
            c = self.record['a'].seq
        if numpy is not None:
            (start, stop) = numpy.array([1, 3])
            self.assertEqual('B', self.record[start])
            self.assertEqual('BC', self.record[start:stop].seq)

    def test_slice_variants(self):
        if False:
            return 10
        'Simple slices using different start/end values.'
        for start in list(range(-30, 30)) + [None]:
            for end in list(range(-30, 30)) + [None]:
                if start is None and end is None:
                    continue
                rec = self.record[start:end]
                seq = self.record.seq[start:end]
                seq_str = str(self.record.seq)[start:end]
                self.assertEqual(seq_str, str(seq))
                self.assertEqual(seq_str, str(rec.seq))
                self.assertEqual('X' * len(seq_str), rec.letter_annotations['fake'])

    def test_slice_simple(self):
        if False:
            i = 10
            return i + 15
        'Simple slice.'
        rec = self.record
        self.assertEqual(len(rec), 26)
        left = rec[:10]
        self.assertEqual(left.seq, rec.seq[:10])
        right = rec[-10:]
        self.assertEqual(right.seq, rec.seq[-10:])
        mid = rec[12:22]
        self.assertEqual(mid.seq, rec.seq[12:22])
        for sub in [left, right, mid]:
            self.assertEqual(len(sub), 10)
            self.assertEqual(sub.id, 'TestID')
            self.assertEqual(sub.name, 'TestName')
            self.assertEqual(sub.description, 'TestDescr')
            self.assertEqual(sub.letter_annotations, {'fake': 'X' * 10})
            self.assertEqual(sub.dbxrefs, [])
            self.assertEqual(sub.annotations, {})
            self.assertEqual(len(sub.features), 1)
            self.assertEqual(sub.features[0].extract(sub.seq), sub.seq)
            self.assertEqual(sub.features[0].extract(sub.seq), sub.seq)

    def test_slice_zero(self):
        if False:
            while True:
                i = 10
        'Zero slice.'
        rec = self.record
        self.assertEqual(len(rec), 26)
        self.assertEqual(len(rec[2:-2]), 22)
        self.assertEqual(len(rec[5:2]), 0)
        self.assertEqual(len(rec[5:2][2:-2]), 0)

    def test_add_simple(self):
        if False:
            print('Hello World!')
        'Simple addition.'
        rec = self.record + self.record
        self.assertEqual(len(rec), 52)
        self.assertEqual(rec.id, 'TestID')
        self.assertEqual(rec.name, 'TestName')
        self.assertEqual(rec.description, 'TestDescr')
        self.assertEqual(rec.dbxrefs, ['TestXRef'])
        self.assertEqual(rec.annotations, {'k': 'v'})
        self.assertEqual(rec.letter_annotations, {'fake': 'X' * 52})
        self.assertEqual(len(rec.features), 2 * len(self.record.features))

    def test_add_seq(self):
        if False:
            while True:
                i = 10
        'Simple addition of Seq or string.'
        for other in [Seq('BIO'), 'BIO']:
            rec = self.record + other
            self.assertEqual(len(rec), 26 + 3)
            self.assertEqual(rec.seq, str(self.record.seq) + 'BIO')
            self.assertEqual(rec.id, 'TestID')
            self.assertEqual(rec.name, 'TestName')
            self.assertEqual(rec.description, 'TestDescr')
            self.assertEqual(rec.dbxrefs, ['TestXRef'])
            self.assertEqual(rec.annotations, {'k': 'v'})
            self.assertEqual(rec.letter_annotations, {})
            self.assertEqual(len(rec.features), len(self.record.features))
            self.assertEqual(rec.features[0].type, 'source')
            self.assertEqual(rec.features[0].location.start, 0)
            self.assertEqual(rec.features[0].location.end, 26)

    def test_add_seqrecord(self):
        if False:
            while True:
                i = 10
        'Simple left addition of SeqRecord from genbank file.'
        other = SeqIO.read('GenBank/dbsource_wrap.gb', 'gb')
        other.dbxrefs = ['dummy']
        rec = self.record + other
        self.assertEqual(len(rec), len(self.record) + len(other))
        self.assertEqual(rec.seq, self.record.seq + other.seq)
        self.assertEqual(rec.id, '<unknown id>')
        self.assertEqual(rec.name, '<unknown name>')
        self.assertEqual(rec.description, '<unknown description>')
        self.assertEqual(rec.dbxrefs, ['TestXRef', 'dummy'])
        self.assertEqual(len(rec.annotations), 0)
        self.assertEqual(len(rec.letter_annotations), 0)
        self.assertEqual(len(rec.features), len(self.record.features) + len(other.features))
        self.assertEqual(rec.features[0].type, 'source')
        self.assertEqual(rec.features[0].location.start, 0)
        self.assertEqual(rec.features[0].location.end, len(self.record))
        i = len(self.record.features)
        self.assertEqual(rec.features[i].type, 'source')
        self.assertEqual(rec.features[i].location.start, len(self.record))
        self.assertEqual(rec.features[i].location.end, len(rec))

    def test_add_seq_left(self):
        if False:
            while True:
                i = 10
        'Simple left addition of Seq or string.'
        for other in [Seq('BIO'), 'BIO']:
            rec = other + self.record
            self.assertEqual(len(rec), 26 + 3)
            self.assertEqual(rec.seq, 'BIO' + self.record.seq)
            self.assertEqual(rec.id, 'TestID')
            self.assertEqual(rec.name, 'TestName')
            self.assertEqual(rec.description, 'TestDescr')
            self.assertEqual(rec.dbxrefs, ['TestXRef'])
            self.assertEqual(rec.annotations, {'k': 'v'})
            self.assertEqual(rec.letter_annotations, {})
            self.assertEqual(len(rec.features), len(self.record.features))
            self.assertEqual(rec.features[0].type, 'source')
            self.assertEqual(rec.features[0].location.start, 3)
            self.assertEqual(rec.features[0].location.end, 26 + 3)

    def test_slice_add_simple(self):
        if False:
            for i in range(10):
                print('nop')
        'Simple slice and add.'
        for cut in range(27):
            rec = self.record[:cut] + self.record[cut:]
            self.assertEqual(rec.seq, self.record.seq)
            self.assertEqual(len(rec), 26)
            self.assertEqual(rec.id, 'TestID')
            self.assertEqual(rec.name, 'TestName')
            self.assertEqual(rec.description, 'TestDescr')
            self.assertEqual(rec.dbxrefs, [])
            self.assertEqual(rec.annotations, {})
            self.assertEqual(rec.letter_annotations, {'fake': 'X' * 26})
            self.assertLessEqual(len(rec.features), len(self.record.features))

    def test_slice_add_shift(self):
        if False:
            while True:
                i = 10
        'Simple slice and add to shift.'
        for cut in range(27):
            rec = self.record[cut:] + self.record[:cut]
            self.assertEqual(rec.seq, self.record.seq[cut:] + self.record.seq[:cut])
            self.assertEqual(len(rec), 26)
            self.assertEqual(rec.id, 'TestID')
            self.assertEqual(rec.name, 'TestName')
            self.assertEqual(rec.description, 'TestDescr')
            self.assertEqual(rec.dbxrefs, [])
            self.assertEqual(rec.annotations, {})
            self.assertEqual(rec.letter_annotations, {'fake': 'X' * 26})
            self.assertLessEqual(len(rec.features), len(self.record.features))

class SeqRecordMethodsMore(unittest.TestCase):
    """Test SeqRecord methods cont."""

    def test_reverse_complement_seq(self):
        if False:
            for i in range(10):
                print('nop')
        s = SeqRecord(Seq('ACTG'), id='TestID', name='TestName', description='TestDescription', dbxrefs=['TestDbxrefs'], features=[SeqFeature(SimpleLocation(0, 3), type='Site')], annotations={'organism': 'bombyx'}, letter_annotations={'test': 'abcd'})
        rc = s.reverse_complement(id=True, name=True, description=True, dbxrefs=True, features=True, annotations=True, letter_annotations=True)
        self.assertEqual('CAGT', rc.seq)
        self.assertEqual('TestID', rc.id)
        self.assertEqual('TestID', s.reverse_complement(id='TestID').id)
        self.assertEqual('TestName', rc.name)
        self.assertEqual('TestName', s.reverse_complement(name='TestName').name)
        self.assertEqual('TestDescription', rc.description)
        self.assertEqual('TestDescription', s.reverse_complement(description='TestDescription').description)
        self.assertEqual(['TestDbxrefs'], rc.dbxrefs)
        self.assertEqual(['TestDbxrefs'], s.reverse_complement(dbxrefs=['TestDbxrefs']).dbxrefs)
        self.assertEqual("[SeqFeature(SimpleLocation(ExactPosition(1), ExactPosition(4)), type='Site')]", repr(rc.features))
        rc2 = s.reverse_complement(features=[SeqFeature(SimpleLocation(1, 4), type='Site')])
        self.assertEqual("[SeqFeature(SimpleLocation(ExactPosition(1), ExactPosition(4)), type='Site')]", repr(rc2.features))
        self.assertEqual({'organism': 'bombyx'}, rc.annotations)
        self.assertEqual({'organism': 'bombyx'}, s.reverse_complement(annotations={'organism': 'bombyx'}).annotations)
        self.assertEqual({'test': 'dcba'}, rc.letter_annotations)
        self.assertEqual({'test': 'abcd'}, s.reverse_complement(letter_annotations={'test': 'abcd'}).letter_annotations)

    def test_reverse_complement_mutable_seq(self):
        if False:
            for i in range(10):
                print('nop')
        s = SeqRecord(MutableSeq('ACTG'))
        self.assertEqual('CAGT', s.reverse_complement().seq)

    def test_translate(self):
        if False:
            while True:
                i = 10
        s = SeqRecord(Seq('ATGGTGTAA'), id='TestID', name='TestName', description='TestDescription', dbxrefs=['TestDbxrefs'], features=[SeqFeature(SimpleLocation(0, 3), type='Site')], annotations={'organism': 'bombyx'}, letter_annotations={'test': 'abcdefghi'})
        t = s.translate()
        self.assertEqual(t.seq, 'MV*')
        self.assertEqual(t.id, '<unknown id>')
        self.assertEqual(t.name, '<unknown name>')
        self.assertEqual(t.description, '<unknown description>')
        self.assertFalse(t.dbxrefs)
        self.assertFalse(t.features)
        self.assertEqual(t.annotations, {'molecule_type': 'protein'})
        self.assertFalse(t.letter_annotations)
        t = s.translate(cds=True, id=True, name=True, description=True, dbxrefs=True, annotations=True)
        self.assertEqual(t.seq, 'MV')
        self.assertEqual(t.id, 'TestID')
        self.assertEqual(t.name, 'TestName')
        self.assertEqual(t.description, 'TestDescription')
        self.assertEqual(t.dbxrefs, ['TestDbxrefs'])
        self.assertFalse(t.features)
        self.assertEqual(t.annotations, {'organism': 'bombyx', 'molecule_type': 'protein'})
        self.assertFalse(t.letter_annotations)

    def test_lt_exception(self):
        if False:
            for i in range(10):
                print('nop')

        def lt():
            if False:
                i = 10
                return i + 15
            return SeqRecord(Seq('A')) < SeqRecord(Seq('A'))
        self.assertRaises(NotImplementedError, lt)

    def test_le_exception(self):
        if False:
            while True:
                i = 10

        def le():
            if False:
                print('Hello World!')
            return SeqRecord(Seq('A')) <= SeqRecord(Seq('A'))
        self.assertRaises(NotImplementedError, le)

    def test_eq_exception(self):
        if False:
            while True:
                i = 10

        def equality():
            if False:
                for i in range(10):
                    print('nop')
            return SeqRecord(Seq('A')) == SeqRecord(Seq('A'))
        self.assertRaises(NotImplementedError, equality)

    def test_ne_exception(self):
        if False:
            for i in range(10):
                print('nop')

        def notequality():
            if False:
                print('Hello World!')
            return SeqRecord(Seq('A')) != SeqRecord(Seq('A'))
        self.assertRaises(NotImplementedError, notequality)

    def test_gt_exception(self):
        if False:
            return 10

        def gt():
            if False:
                print('Hello World!')
            return SeqRecord(Seq('A')) > SeqRecord(Seq('A'))
        self.assertRaises(NotImplementedError, gt)

    def test_ge_exception(self):
        if False:
            i = 10
            return i + 15

        def ge():
            if False:
                while True:
                    i = 10
            return SeqRecord(Seq('A')) >= SeqRecord(Seq('A'))
        self.assertRaises(NotImplementedError, ge)

    def test_hash_exception(self):
        if False:
            i = 10
            return i + 15

        def hash1():
            if False:
                return 10
            hash(SeqRecord(Seq('A')))
        self.assertRaises(TypeError, hash1)

        def hash2():
            if False:
                print('Hello World!')
            SeqRecord(Seq('A')).__hash__()
        self.assertRaises(TypeError, hash2)

class TestTranslation(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.s = SeqRecord(Seq('ATGGTGTAA'), id='TestID', name='TestName', description='TestDescription', dbxrefs=['TestDbxrefs'], features=[SeqFeature(SimpleLocation(0, 3), type='Site')], annotations={'organism': 'bombyx'}, letter_annotations={'test': 'abcdefghi'})

    def test_defaults(self):
        if False:
            i = 10
            return i + 15
        t = self.s.translate()
        self.assertEqual(t.seq, 'MV*')
        self.assertEqual(t.id, '<unknown id>')
        self.assertEqual(t.name, '<unknown name>')
        self.assertEqual(t.description, '<unknown description>')
        self.assertFalse(t.dbxrefs)
        self.assertFalse(t.features)
        self.assertEqual(t.annotations, {'molecule_type': 'protein'})
        self.assertFalse(t.letter_annotations)

    def test_preserve(self):
        if False:
            i = 10
            return i + 15
        t = self.s.translate(cds=True, id=True, name=True, description=True, dbxrefs=True, annotations=True)
        self.assertEqual(t.seq, 'MV')
        self.assertEqual(t.id, 'TestID')
        self.assertEqual(t.name, 'TestName')
        self.assertEqual(t.description, 'TestDescription')
        self.assertEqual(t.dbxrefs, ['TestDbxrefs'])
        self.assertFalse(t.features)
        self.assertEqual(t.annotations, {'organism': 'bombyx', 'molecule_type': 'protein'})
        self.assertFalse(t.letter_annotations)
        self.assertRaises(TypeError, self.s.translate, features=True)
        self.assertRaises(TypeError, self.s.translate, letter_annotations=True)

    def test_new_annot(self):
        if False:
            while True:
                i = 10
        t = self.s.translate(1, to_stop=True, gap='-', id='Foo', name='Bar', description='Baz', dbxrefs=['Nope'], features=[SeqFeature(SimpleLocation(0, 3), type='Site')], annotations={'a': 'team'}, letter_annotations={'aa': ['Met', 'Val']})
        self.assertEqual(t.seq, 'MV')
        self.assertEqual(t.id, 'Foo')
        self.assertEqual(t.name, 'Bar')
        self.assertEqual(t.description, 'Baz')
        self.assertEqual(t.dbxrefs, ['Nope'])
        self.assertEqual(len(t.features), 1)
        self.assertEqual(t.annotations, {'a': 'team', 'molecule_type': 'protein'})
        self.assertEqual(t.letter_annotations, {'aa': ['Met', 'Val']})
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)