"""Common code for SeqRecord object tests."""
import unittest
from Bio.Seq import UndefinedSequenceError
from Bio.SeqUtils.CheckSum import seguid
from Bio.SeqFeature import ExactPosition, UnknownPosition
from Bio.SeqFeature import SimpleLocation, CompoundLocation, SeqFeature
from Bio.SeqRecord import SeqRecord
from test_SeqIO import SeqIOTestBaseClass

class SeqRecordTestBaseClass(unittest.TestCase):

    def compare_reference(self, r1, r2):
        if False:
            return 10
        'Compare two Reference objects.\n\n        Note r2 is assumed to be a BioSQL DBSeqRecord, due to limitations\n        of the BioSQL table structure.\n        '
        self.assertEqual(r1.title, r2.title)
        self.assertEqual(r1.authors, r2.authors)
        self.assertEqual(r1.journal, r2.journal)
        self.assertEqual(r1.medline_id, r2.medline_id)
        if r1.pubmed_id and r2.pubmed_id:
            self.assertEqual(r1.pubmed_id, r2.pubmed_id)
        if r2.comment:
            self.assertEqual(r1.comment, r2.comment)
        if r2.consrtm:
            self.assertEqual(r1.consrtm, r2.consrtm)
        if len(r1.location) == 0:
            self.assertEqual(len(r2.location), 0)
        else:
            self.assertIsInstance(r1.location[0], SimpleLocation)
            self.assertIsInstance(r2.location[0], SimpleLocation)
            self.assertEqual(r1.location[0].start, r2.location[0].start)
            self.assertEqual(r1.location[0].end, r2.location[0].end)

    def compare_feature(self, old_f, new_f):
        if False:
            print('Hello World!')
        'Compare two SeqFeature objects.'
        self.assertIsInstance(old_f, SeqFeature)
        self.assertIsInstance(new_f, SeqFeature)
        self.assertEqual(old_f.type, new_f.type)
        self.assertEqual(old_f.strand, new_f.strand)
        self.assertEqual(old_f.ref, new_f.ref)
        self.assertEqual(old_f.ref_db, new_f.ref_db)
        if new_f.id != '<unknown id>':
            self.assertEqual(old_f.id, new_f.id)
        if not (isinstance(old_f.location.start, UnknownPosition) and isinstance(new_f.location.start, UnknownPosition)):
            self.assertEqual(old_f.location.start, new_f.location.start)
        if not (isinstance(old_f.location.end, UnknownPosition) and isinstance(new_f.location.end, UnknownPosition)):
            self.assertEqual(old_f.location.end, new_f.location.end)
        if isinstance(old_f.location, CompoundLocation):
            self.assertIsInstance(new_f.location, CompoundLocation)
        else:
            self.assertNotIsInstance(new_f.location, CompoundLocation)
        if isinstance(old_f.location, CompoundLocation):
            self.assertEqual(len(old_f.location.parts), len(new_f.location.parts))
            for (old_l, new_l) in zip(old_f.location.parts, new_f.location.parts):
                self.assertEqual(old_l.start, new_l.start)
                self.assertEqual(old_l.end, new_l.end)
                self.assertEqual(old_l.strand, new_l.strand)
                self.assertEqual(old_l.ref, new_l.ref)
                self.assertEqual(old_l.ref_db, new_l.ref_db)
        self.assertEqual(len(old_f.location.parts), len(new_f.location.parts))
        for (old_sub, new_sub) in zip(old_f.location.parts, new_f.location.parts):
            if isinstance(old_sub.start, UnknownPosition):
                self.assertIsInstance(new_sub.start, UnknownPosition)
            else:
                self.assertEqual(old_sub.start, new_sub.start)
            if isinstance(old_sub.end, UnknownPosition):
                self.assertIsInstance(new_sub.end, UnknownPosition)
            else:
                self.assertEqual(old_sub.end, new_sub.end)
            self.assertEqual(old_sub.strand, new_sub.strand)
        self.assertCountEqual(old_f.qualifiers, new_f.qualifiers)
        for key in old_f.qualifiers:
            if isinstance(old_f.qualifiers[key], str):
                if isinstance(new_f.qualifiers[key], str):
                    self.assertEqual(old_f.qualifiers[key], new_f.qualifiers[key])
                elif isinstance(new_f.qualifiers[key], list):
                    self.assertEqual([old_f.qualifiers[key]], new_f.qualifiers[key])
                else:
                    self.fail(f"Problem with feature's '{key}' qualifier")
            else:
                self.assertEqual(old_f.qualifiers[key], new_f.qualifiers[key])

    def compare_features(self, old_list, new_list):
        if False:
            while True:
                i = 10
        'Compare two lists of SeqFeature objects.'
        self.assertIsInstance(old_list, list)
        self.assertIsInstance(new_list, list)
        self.assertEqual(len(old_list), len(new_list))
        for (old_f, new_f) in zip(old_list, new_list):
            self.compare_feature(old_f, new_f)

    def compare_sequence(self, old, new):
        if False:
            i = 10
            return i + 15
        'Compare two Seq objects.'
        self.assertEqual(len(old), len(new))
        self.assertEqual(len(old), len(new))
        try:
            bytes(old)
        except UndefinedSequenceError:
            self.assertRaises(UndefinedSequenceError, bytes, new)
            return
        self.assertEqual(old, new)
        ln = len(old)
        s = str(old)
        if ln < 50:
            indices = list(range(-ln, ln))
        else:
            indices = [-ln, -ln + 1, -(ln // 2), -1, 0, 1, ln // 2, ln - 2, ln - 1]
        for i in indices:
            expected = s[i]
            self.assertEqual(expected, old[i])
            self.assertEqual(expected, new[i])
        indices.append(ln)
        indices.append(ln + 1000)
        for i in indices:
            for j in indices:
                expected = s[i:j]
                self.assertEqual(expected, old[i:j])
                self.assertEqual(expected, new[i:j])
                for step in [1, 3]:
                    expected = s[i:j:step]
                    self.assertEqual(expected, old[i:j:step])
                    self.assertEqual(expected, new[i:j:step])
            expected = s[i:]
            self.assertEqual(expected, old[i:])
            self.assertEqual(expected, new[i:])
            expected = s[:i]
            self.assertEqual(expected, old[:i])
            self.assertEqual(expected, new[:i])
        self.assertEqual(s, old[:])
        self.assertEqual(s, new[:])

    def compare_record(self, old, new):
        if False:
            while True:
                i = 10
        'Compare two SeqRecord or DBSeqRecord objects.'
        self.assertIsInstance(old, SeqRecord)
        self.assertIsInstance(new, SeqRecord)
        self.compare_sequence(old.seq, new.seq)
        self.assertEqual(old.id, new.id)
        self.assertEqual(old.name, new.name)
        self.assertEqual(old.description, new.description)
        self.assertEqual(old.dbxrefs, new.dbxrefs)
        self.compare_features(old.features, new.features)
        new_keys = set(new.annotations).difference(old.annotations)
        new_keys = new_keys.difference(['cross_references', 'date', 'data_file_division', 'ncbi_taxid', 'gi'])
        self.assertEqual(len(new_keys), 0, msg=f"Unexpected new annotation keys: {', '.join(new_keys)}")
        missing_keys = set(old.annotations).difference(new.annotations)
        missing_keys = missing_keys.difference(['gene_name', 'ncbi_taxid', 'structured_comment'])
        self.assertEqual(len(missing_keys), 0, msg=f"Unexpectedly missing annotation keys: {', '.join(missing_keys)}")
        for key in set(old.annotations).intersection(new.annotations):
            if key == 'references':
                self.assertEqual(len(old.annotations[key]), len(new.annotations[key]))
                for (old_r, new_r) in zip(old.annotations[key], new.annotations[key]):
                    self.compare_reference(old_r, new_r)
            elif key == 'comment':
                if isinstance(old.annotations[key], list):
                    old_comment = ' '.join(old.annotations[key])
                else:
                    old_comment = old.annotations[key]
                if isinstance(new.annotations[key], list):
                    new_comment = ' '.join(new.annotations[key])
                else:
                    new_comment = new.annotations[key]
                old_comment = old_comment.replace('\n', ' ').replace('  ', ' ')
                new_comment = new_comment.replace('\n', ' ').replace('  ', ' ')
                self.assertEqual(old_comment, new_comment, msg='Comment annotation changed by load/retrieve')
            elif key in ['taxonomy', 'organism', 'source']:
                self.assertTrue(isinstance(new.annotations[key], (list, str)))
            elif isinstance(old.annotations[key], type(new.annotations[key])):
                self.assertEqual(old.annotations[key], new.annotations[key], msg=f"Annotation '{key}' changed by load/retrieve")
            elif isinstance(old.annotations[key], str) and isinstance(new.annotations[key], list):
                self.assertEqual([old.annotations[key]], new.annotations[key], msg=f"Annotation '{key}' changed by load/retrieve")
            elif isinstance(old.annotations[key], list) and isinstance(new.annotations[key], str):
                self.assertEqual(old.annotations[key], [new.annotations[key]], msg=f"Annotation '{key}' changed by load/retrieve")

    def compare_records(self, old_list, new_list):
        if False:
            print('Hello World!')
        'Compare two lists of SeqRecord objects.'
        self.assertEqual(len(old_list), len(new_list))
        for (old_r, new_r) in zip(old_list, new_list):
            self.compare_record(old_r, new_r)