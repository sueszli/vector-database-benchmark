"""Tests for seq module."""
import copy
import unittest
import warnings
try:
    import numpy
except ImportError:
    numpy = None
from Bio import BiopythonWarning, BiopythonDeprecationWarning
from Bio import Seq
from Bio.Data.IUPACData import ambiguous_dna_complement, ambiguous_rna_complement, ambiguous_dna_values, ambiguous_rna_values
from Bio.Data.CodonTable import TranslationError, standard_dna_table
test_seqs = [Seq.Seq('TCAAAAGGATGCATCATG'), Seq.Seq('T'), Seq.Seq('ATGAAACTG'), Seq.Seq('ATGAARCTG'), Seq.Seq('AWGAARCKG'), Seq.Seq(''.join(ambiguous_rna_values)), Seq.Seq(''.join(ambiguous_dna_values)), Seq.Seq('AWGAARCKG'), Seq.Seq('AUGAAACUG'), Seq.Seq('ATGAAA-CTG'), Seq.Seq('ATGAAACTGWN'), Seq.Seq('AUGAAA==CUG'), Seq.Seq('AUGAAACUGWN'), Seq.Seq('AUGAAACTG'), Seq.MutableSeq('ATGAAACTG'), Seq.MutableSeq('AUGaaaCUG'), Seq.Seq('ACTGTCGTCT')]
protein_seqs = [Seq.Seq('ATCGPK'), Seq.Seq('T.CGPK'), Seq.Seq('T-CGPK'), Seq.Seq('MEDG-KRXR*'), Seq.MutableSeq('ME-K-DRXR*XU'), Seq.Seq('MEDG-KRXR@'), Seq.Seq('ME-KR@'), Seq.Seq('MEDG.KRXR@')]

class TestSeq(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.s = Seq.Seq('TCAAAAGGATGCATCATG')

    def test_as_string(self):
        if False:
            print('Hello World!')
        'Test converting Seq to string.'
        self.assertEqual('TCAAAAGGATGCATCATG', self.s)

    def test_seq_construction(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Seq object initialization.'
        sequence = bytes(self.s)
        s = Seq.Seq(sequence)
        self.assertIsInstance(s, Seq.Seq, 'Creating MutableSeq using bytes')
        self.assertEqual(s, self.s)
        s = Seq.Seq(bytearray(sequence))
        self.assertIsInstance(s, Seq.Seq, 'Creating MutableSeq using bytearray')
        self.assertEqual(s, self.s)
        s = Seq.Seq(sequence.decode('ASCII'))
        self.assertIsInstance(s, Seq.Seq, 'Creating MutableSeq using str')
        self.assertEqual(s, self.s)
        s = Seq.Seq(self.s)
        self.assertIsInstance(s, Seq.Seq, 'Creating MutableSeq using Seq')
        self.assertEqual(s, self.s)
        s = Seq.Seq(Seq.MutableSeq(sequence))
        self.assertIsInstance(s, Seq.Seq, 'Creating MutableSeq using MutableSeq')
        self.assertEqual(s, self.s)
        self.assertRaises(UnicodeEncodeError, Seq.Seq, 'ÄþÇÐ')
        self.assertRaises(UnicodeEncodeError, Seq.Seq, 'あいうえお')

    def test_repr(self):
        if False:
            while True:
                i = 10
        'Test representation of Seq object.'
        self.assertEqual("Seq('TCAAAAGGATGCATCATG')", repr(self.s))

    def test_truncated_repr(self):
        if False:
            return 10
        seq = 'TCAAAAGGATGCATCATGTCAAAAGGATGCATCATGTCAAAAGGATGCATCATGTCAAAAGGA'
        expected = "Seq('TCAAAAGGATGCATCATGTCAAAAGGATGCATCATGTCAAAAGGATGCATCATG...GGA')"
        self.assertEqual(expected, repr(Seq.Seq(seq)))

    def test_length(self):
        if False:
            print('Hello World!')
        'Test len method on Seq object.'
        self.assertEqual(18, len(self.s))

    def test_first_nucleotide(self):
        if False:
            i = 10
            return i + 15
        'Test getting first nucleotide of Seq.'
        self.assertEqual('T', self.s[0])

    def test_last_nucleotide(self):
        if False:
            while True:
                i = 10
        'Test getting last nucleotide of Seq.'
        self.assertEqual('G', self.s[-1])

    def test_slicing(self):
        if False:
            return 10
        'Test slicing of Seq.'
        self.assertEqual('AA', self.s[3:5])

    def test_reverse(self):
        if False:
            while True:
                i = 10
        'Test reverse using -1 stride.'
        self.assertEqual('GTACTACGTAGGAAAACT', self.s[::-1])

    def test_extract_third_nucleotide(self):
        if False:
            print('Hello World!')
        'Test extracting every third nucleotide (slicing with stride 3).'
        self.assertEqual('TAGTAA', self.s[0::3])
        self.assertEqual('CAGGTT', self.s[1::3])
        self.assertEqual('AAACCG', self.s[2::3])

    def test_concatenation_of_seq(self):
        if False:
            return 10
        t = Seq.Seq('T')
        u = self.s + t
        self.assertEqual(str(self.s) + 'T', u)
        self.assertEqual(self.s + Seq.Seq('T'), 'TCAAAAGGATGCATCATGT')

    def test_replace(self):
        if False:
            while True:
                i = 10
        self.assertEqual('ATCCCA', Seq.Seq('ATC-CCA').replace('-', ''))

    def test_cast_to_list(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(list('ATC'), list(Seq.Seq('ATC')))
        self.assertEqual(list('ATC'), list(Seq.MutableSeq('ATC')))
        self.assertEqual(list(''), list(Seq.MutableSeq('')))
        self.assertEqual(list(''), list(Seq.Seq('')))
        with self.assertRaises(Seq.UndefinedSequenceError):
            list(Seq.Seq(None, length=3))
        with self.assertRaises(Seq.UndefinedSequenceError):
            list(Seq.Seq({3: 'ACGT'}, length=10))

class TestSeqStringMethods(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.s = Seq.Seq('TCAAAAGGATGCATCATG')
        self.dna = [Seq.Seq('ATCG'), Seq.Seq('gtca'), Seq.MutableSeq('GGTCA'), Seq.Seq('CTG-CA')]
        self.rna = [Seq.Seq('AUUUCG'), Seq.MutableSeq('AUUCG'), Seq.Seq('uCAg'), Seq.MutableSeq('UC-AG'), Seq.Seq('U.CAG')]
        self.nuc = [Seq.Seq('ATCG')]
        self.protein = [Seq.Seq('ATCGPK'), Seq.Seq('atcGPK'), Seq.Seq('T.CGPK'), Seq.Seq('T-CGPK'), Seq.Seq('MEDG-KRXR*'), Seq.MutableSeq('ME-K-DRXR*XU'), Seq.Seq('MEDG-KRXR@'), Seq.Seq('ME-KR@'), Seq.Seq('MEDG.KRXR@')]
        self.test_chars = ['-', Seq.Seq('-'), Seq.Seq('*'), '-X@']

    def test_string_methods(self):
        if False:
            print('Hello World!')
        for a in self.dna + self.rna + self.nuc + self.protein:
            self.assertEqual(a.lower(), str(a).lower())
            self.assertEqual(a.upper(), str(a).upper())
            self.assertEqual(a.islower(), str(a).islower())
            self.assertEqual(a.isupper(), str(a).isupper())
            self.assertEqual(a.strip(), str(a).strip())
            self.assertEqual(a.lstrip(), str(a).lstrip())
            self.assertEqual(a.rstrip(), str(a).rstrip())

    def test_mutableseq_upper_lower(self):
        if False:
            for i in range(10):
                print('nop')
        seq = Seq.MutableSeq('ACgt')
        lseq = seq.lower()
        self.assertEqual(lseq, 'acgt')
        self.assertEqual(seq, 'ACgt')
        self.assertTrue(lseq.islower())
        self.assertFalse(seq.islower())
        lseq = seq.lower(inplace=False)
        self.assertEqual(lseq, 'acgt')
        self.assertEqual(seq, 'ACgt')
        self.assertTrue(lseq.islower())
        self.assertFalse(seq.islower())
        lseq = seq.lower(inplace=True)
        self.assertEqual(lseq, 'acgt')
        self.assertIs(lseq, seq)
        self.assertTrue(lseq.islower())
        self.assertTrue(lseq.islower())
        seq = Seq.MutableSeq('ACgt')
        useq = seq.upper()
        self.assertEqual(useq, 'ACGT')
        self.assertEqual(seq, 'ACgt')
        self.assertTrue(useq.isupper())
        self.assertFalse(seq.isupper())
        useq = seq.upper(inplace=False)
        self.assertEqual(useq, 'ACGT')
        self.assertEqual(seq, 'ACgt')
        self.assertTrue(useq.isupper())
        self.assertFalse(seq.isupper())
        useq = seq.upper(inplace=True)
        self.assertEqual(useq, 'ACGT')
        self.assertIs(useq, seq)
        self.assertTrue(useq.isupper())
        self.assertTrue(seq.isupper())

    def test_hash(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True):
            hash(self.s)

    def test_not_equal_comparsion(self):
        if False:
            for i in range(10):
                print('nop')
        'Test __ne__ comparison method.'
        self.assertNotEqual(Seq.Seq('TCAAA'), Seq.Seq('TCAAAA'))

    def test_less_than_comparison(self):
        if False:
            i = 10
            return i + 15
        'Test __lt__ comparison method.'
        self.assertLess(self.s[:-1], self.s)

    def test_less_than_comparison_of_incompatible_types(self):
        if False:
            for i in range(10):
                print('nop')
        'Test incompatible types __lt__ comparison method.'
        with self.assertRaises(TypeError):
            self.s < 1

    def test_less_than_or_equal_comparison(self):
        if False:
            while True:
                i = 10
        'Test __le__ comparison method.'
        self.assertLessEqual(self.s, self.s)

    def test_less_than_or_equal_comparison_of_incompatible_types(self):
        if False:
            print('Hello World!')
        'Test incompatible types __le__ comparison method.'
        with self.assertRaises(TypeError):
            self.s <= 1

    def test_greater_than_comparison(self):
        if False:
            for i in range(10):
                print('nop')
        'Test __gt__ comparison method.'
        self.assertGreater(self.s, self.s[:-1])

    def test_greater_than_comparison_of_incompatible_types(self):
        if False:
            print('Hello World!')
        'Test incompatible types __gt__ comparison method.'
        with self.assertRaises(TypeError):
            self.s > 1

    def test_greater_than_or_equal_comparison(self):
        if False:
            i = 10
            return i + 15
        'Test __ge__ comparison method.'
        self.assertGreaterEqual(self.s, self.s)

    def test_greater_than_or_equal_comparison_of_incompatible_types(self):
        if False:
            print('Hello World!')
        'Test incompatible types __ge__ comparison method.'
        with self.assertRaises(TypeError):
            self.s >= 1

    def test_add_method_using_wrong_object(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self.s + {}

    def test_radd_method_using_wrong_object(self):
        if False:
            return 10
        self.assertEqual(self.s.__radd__({}), NotImplemented)

    def test_contains_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIn('AAAA', self.s)

    def test_startswith(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.s.startswith('TCA'))
        self.assertTrue(self.s.startswith(('CAA', 'CTA'), 1))

    def test_endswith(self):
        if False:
            return 10
        self.assertTrue(self.s.endswith('ATG'))
        self.assertTrue(self.s.endswith(('ATG', 'CTA')))

    def test_append_nucleotides(self):
        if False:
            return 10
        self.test_chars.append(Seq.Seq('A'))
        self.assertEqual(5, len(self.test_chars))

    def test_append_proteins(self):
        if False:
            return 10
        self.test_chars.append(Seq.Seq('K'))
        self.test_chars.append(Seq.Seq('K-'))
        self.test_chars.append(Seq.Seq('K@'))
        self.assertEqual(7, len(self.test_chars))

    def test_stripping_characters(self):
        if False:
            while True:
                i = 10
        for a in self.dna + self.rna + self.nuc + self.protein:
            for char in self.test_chars:
                str_char = str(char)
                self.assertEqual(a.strip(char), str(a).strip(str_char))
                self.assertEqual(a.lstrip(char), str(a).lstrip(str_char))
                self.assertEqual(a.rstrip(char), str(a).rstrip(str_char))
                try:
                    removeprefix = str(a).removeprefix(str_char)
                    removesuffix = str(a).removesuffix(str_char)
                except AttributeError:
                    if str(a).startswith(str_char):
                        removeprefix = str(a)[len(str_char):]
                    else:
                        removeprefix = str(a)
                    if str_char and str(a).endswith(str_char):
                        removesuffix = str(a)[:-len(str_char)]
                    else:
                        removesuffix = str(a)
                self.assertEqual(a.removeprefix(char), removeprefix)
                self.assertEqual(a.removesuffix(char), removesuffix)

    def test_finding_characters(self):
        if False:
            while True:
                i = 10
        for a in self.dna + self.rna + self.nuc + self.protein:
            for char in self.test_chars:
                str_char = str(char)
                self.assertEqual(a.find(char), str(a).find(str_char))
                self.assertEqual(a.find(char, 2, -2), str(a).find(str_char, 2, -2))
                self.assertEqual(a.rfind(char), str(a).rfind(str_char))
                self.assertEqual(a.rfind(char, 2, -2), str(a).rfind(str_char, 2, -2))

    def test_counting_characters(self):
        if False:
            while True:
                i = 10
        from Bio.SeqRecord import SeqRecord
        for a in self.dna + self.rna + self.nuc + self.protein:
            r = SeqRecord(a)
            for char in self.test_chars:
                str_char = str(char)
                n = str(a).count(str_char)
                self.assertEqual(a.count(char), n)
                self.assertEqual(r.count(char), n)
                n = str(a).count(str_char, 2, -2)
                self.assertEqual(a.count(char, 2, -2), n)
                self.assertEqual(r.count(char, 2, -2), n)

    def test_splits(self):
        if False:
            i = 10
            return i + 15
        for a in self.dna + self.rna + self.nuc + self.protein:
            for char in self.test_chars:
                str_char = str(char)
                self.assertEqual(a.split(char), str(a).split(str_char))
                self.assertEqual(a.rsplit(char), str(a).rsplit(str_char))
                for max_sep in [0, 1, 2, 999]:
                    self.assertEqual(a.split(char, max_sep), str(a).split(str_char, max_sep))

class TestSeqAddition(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.dna = [Seq.Seq('ATCG'), Seq.Seq('gtca'), Seq.MutableSeq('GGTCA'), Seq.Seq('CTG-CA'), 'TGGTCA']
        self.rna = [Seq.Seq('AUUUCG'), Seq.MutableSeq('AUUCG'), Seq.Seq('uCAg'), Seq.MutableSeq('UC-AG'), Seq.Seq('U.CAG'), 'UGCAU']
        self.nuc = [Seq.Seq('ATCG'), 'UUUTTTACG']
        self.protein = [Seq.Seq('ATCGPK'), Seq.Seq('atcGPK'), Seq.Seq('T.CGPK'), Seq.Seq('T-CGPK'), Seq.Seq('MEDG-KRXR*'), Seq.MutableSeq('ME-K-DRXR*XU'), 'TEDDF']

    def test_addition_dna_rna_with_generic_nucleotides(self):
        if False:
            while True:
                i = 10
        for a in self.dna + self.rna:
            for b in self.nuc:
                c = a + b
                self.assertEqual(c, str(a) + str(b))

    def test_addition_dna_rna_with_generic_nucleotides_inplace(self):
        if False:
            return 10
        for a in self.dna + self.rna:
            for b in self.nuc:
                c = b + a
                b += a
                self.assertEqual(c, b)

    def test_addition_rna_with_rna(self):
        if False:
            print('Hello World!')
        self.rna.pop(3)
        for a in self.rna:
            for b in self.rna:
                c = a + b
                self.assertEqual(c, str(a) + str(b))

    def test_addition_rna_with_rna_inplace(self):
        if False:
            print('Hello World!')
        self.rna.pop(3)
        for a in self.rna:
            for b in self.rna:
                c = b + a
                b += a
                self.assertEqual(c, b)

    def test_addition_dna_with_dna(self):
        if False:
            print('Hello World!')
        for a in self.dna:
            for b in self.dna:
                c = a + b
                self.assertEqual(c, str(a) + str(b))

    def test_addition_dna_with_dna_inplace(self):
        if False:
            for i in range(10):
                print('nop')
        for a in self.dna:
            for b in self.dna:
                c = b + a
                b += a
                self.assertEqual(c, b)

    def test_addition_dna_with_rna(self):
        if False:
            return 10
        self.dna.pop(4)
        self.rna.pop(5)
        for a in self.dna:
            for b in self.rna:
                self.assertEqual(str(a) + str(b), a + b)
                self.assertEqual(str(b) + str(a), b + a)
                c = a
                c += b
                self.assertEqual(c, str(a) + str(b))
                c = b
                c += a
                self.assertEqual(c, str(b) + str(a))

    def test_addition_proteins(self):
        if False:
            return 10
        self.protein.pop(2)
        for a in self.protein:
            for b in self.protein:
                c = a + b
                self.assertEqual(c, str(a) + str(b))

    def test_addition_proteins_inplace(self):
        if False:
            while True:
                i = 10
        self.protein.pop(2)
        for a in self.protein:
            for b in self.protein:
                c = b + a
                b += a
                self.assertEqual(c, b)

    def test_adding_protein_with_nucleotides(self):
        if False:
            while True:
                i = 10
        for a in self.protein[0:5]:
            for b in self.dna[0:3] + self.rna[0:4]:
                self.assertEqual(str(a) + str(b), a + b)
                a += b

    def test_adding_generic_nucleotide_with_other_nucleotides(self):
        if False:
            while True:
                i = 10
        for a in self.nuc:
            for b in self.dna + self.rna + self.nuc:
                c = a + b
                self.assertEqual(c, str(a) + str(b))

    def test_adding_generic_nucleotide_with_other_nucleotides_inplace(self):
        if False:
            return 10
        for a in self.nuc:
            for b in self.dna + self.rna + self.nuc:
                c = b + a
                b += a
                self.assertEqual(c, b)

class TestSeqMultiplication(unittest.TestCase):

    def test_mul_method(self):
        if False:
            while True:
                i = 10
        'Test mul method; relies on addition method.'
        for seq in test_seqs + protein_seqs:
            self.assertEqual(seq * 3, seq + seq + seq)
        if numpy is not None:
            factor = numpy.intc(3)
            for seq in test_seqs + protein_seqs:
                self.assertEqual(seq * factor, seq + seq + seq)

    def test_mul_method_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test mul method exceptions.'
        for seq in test_seqs + protein_seqs:
            with self.assertRaises(TypeError):
                seq * 3.0
            with self.assertRaises(TypeError):
                seq * ''

    def test_rmul_method(self):
        if False:
            while True:
                i = 10
        'Test rmul method; relies on addition method.'
        for seq in test_seqs + protein_seqs:
            self.assertEqual(3 * seq, seq + seq + seq)
        if numpy is not None:
            factor = numpy.intc(3)
            for seq in test_seqs + protein_seqs:
                self.assertEqual(factor * seq, seq + seq + seq)

    def test_rmul_method_exceptions(self):
        if False:
            i = 10
            return i + 15
        'Test rmul method exceptions.'
        for seq in test_seqs + protein_seqs:
            with self.assertRaises(TypeError):
                3.0 * seq
            with self.assertRaises(TypeError):
                '' * seq

    def test_imul_method(self):
        if False:
            print('Hello World!')
        'Test imul method; relies on addition and mull methods.'
        for seq in test_seqs + protein_seqs:
            original_seq = seq * 1
            seq *= 3
            self.assertEqual(seq, original_seq + original_seq + original_seq)
        if numpy is not None:
            factor = numpy.intc(3)
            for seq in test_seqs + protein_seqs:
                original_seq = seq * 1
                seq *= factor
                self.assertEqual(seq, original_seq + original_seq + original_seq)

    def test_imul_method_exceptions(self):
        if False:
            print('Hello World!')
        'Test imul method exceptions.'
        for seq in test_seqs + protein_seqs:
            with self.assertRaises(TypeError):
                seq *= 3.0
            with self.assertRaises(TypeError):
                seq *= ''

class TestMutableSeq(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        sequence = b'TCAAAAGGATGCATCATG'
        self.s = Seq.Seq(sequence)
        self.mutable_s = Seq.MutableSeq(sequence)

    def test_mutableseq_construction(self):
        if False:
            return 10
        'Test MutableSeq object initialization.'
        sequence = bytes(self.s)
        mutable_s = Seq.MutableSeq(sequence)
        self.assertIsInstance(mutable_s, Seq.MutableSeq, 'Initializing MutableSeq from bytes')
        self.assertEqual(mutable_s, self.s)
        mutable_s = Seq.MutableSeq(bytearray(sequence))
        self.assertIsInstance(mutable_s, Seq.MutableSeq, 'Initializing MutableSeq from bytearray')
        self.assertEqual(mutable_s, self.s)
        mutable_s = Seq.MutableSeq(sequence.decode('ASCII'))
        self.assertIsInstance(mutable_s, Seq.MutableSeq, 'Initializing MutableSeq from str')
        self.assertEqual(mutable_s, self.s)
        mutable_s = Seq.MutableSeq(self.s)
        self.assertIsInstance(mutable_s, Seq.MutableSeq, 'Initializing MutableSeq from Seq')
        self.assertEqual(mutable_s, self.s)
        mutable_s = Seq.MutableSeq(Seq.MutableSeq(sequence))
        self.assertEqual(mutable_s, self.s)
        self.assertIsInstance(mutable_s, Seq.MutableSeq, 'Initializing MutableSeq from MutableSeq')
        self.assertRaises(UnicodeEncodeError, Seq.MutableSeq, 'ÄþÇÐ')
        self.assertRaises(UnicodeEncodeError, Seq.MutableSeq, 'あいうえお')

    def test_repr(self):
        if False:
            print('Hello World!')
        self.assertEqual("MutableSeq('TCAAAAGGATGCATCATG')", repr(self.mutable_s))

    def test_truncated_repr(self):
        if False:
            while True:
                i = 10
        seq = 'TCAAAAGGATGCATCATGTCAAAAGGATGCATCATGTCAAAAGGATGCATCATGTCAAAAGGA'
        expected = "MutableSeq('TCAAAAGGATGCATCATGTCAAAAGGATGCATCATGTCAAAAGGATGCATCATG...GGA')"
        self.assertEqual(expected, repr(Seq.MutableSeq(seq)))

    def test_equal_comparison(self):
        if False:
            print('Hello World!')
        'Test __eq__ comparison method.'
        self.assertEqual(self.mutable_s, 'TCAAAAGGATGCATCATG')

    def test_not_equal_comparison(self):
        if False:
            for i in range(10):
                print('nop')
        'Test __ne__ comparison method.'
        self.assertNotEqual(self.mutable_s, 'other thing')

    def test_less_than_comparison(self):
        if False:
            while True:
                i = 10
        'Test __lt__ comparison method.'
        self.assertLess(self.mutable_s[:-1], self.mutable_s)

    def test_less_than_comparison_of_incompatible_types(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            self.mutable_s < 1

    def test_less_than_comparison_with_str(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertLessEqual(self.mutable_s[:-1], 'TCAAAAGGATGCATCATG')

    def test_less_than_or_equal_comparison(self):
        if False:
            return 10
        'Test __le__ comparison method.'
        self.assertLessEqual(self.mutable_s[:-1], self.mutable_s)

    def test_less_than_or_equal_comparison_of_incompatible_types(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            self.mutable_s <= 1

    def test_less_than_or_equal_comparison_with_str(self):
        if False:
            i = 10
            return i + 15
        self.assertLessEqual(self.mutable_s[:-1], 'TCAAAAGGATGCATCATG')

    def test_greater_than_comparison(self):
        if False:
            i = 10
            return i + 15
        'Test __gt__ comparison method.'
        self.assertGreater(self.mutable_s, self.mutable_s[:-1])

    def test_greater_than_comparison_of_incompatible_types(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self.mutable_s > 1

    def test_greater_than_comparison_with_str(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertGreater(self.mutable_s, 'TCAAAAGGATGCATCAT')

    def test_greater_than_or_equal_comparison(self):
        if False:
            for i in range(10):
                print('nop')
        'Test __ge__ comparison method.'
        self.assertGreaterEqual(self.mutable_s, self.mutable_s)

    def test_greater_than_or_equal_comparison_of_incompatible_types(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            self.mutable_s >= 1

    def test_greater_than_or_equal_comparison_with_str(self):
        if False:
            while True:
                i = 10
        self.assertGreaterEqual(self.mutable_s, 'TCAAAAGGATGCATCATG')

    def test_add_method(self):
        if False:
            return 10
        'Test adding wrong type to MutableSeq.'
        with self.assertRaises(TypeError):
            self.mutable_s + 1234

    def test_radd_method_wrong_type(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.mutable_s.__radd__(1234), NotImplemented)

    def test_contains_method(self):
        if False:
            while True:
                i = 10
        self.assertIn('AAAA', self.mutable_s)

    def test_startswith(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.mutable_s.startswith('TCA'))
        self.assertTrue(self.mutable_s.startswith(('CAA', 'CTA'), 1))

    def test_endswith(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.mutable_s.endswith('ATG'))
        self.assertTrue(self.mutable_s.endswith(('ATG', 'CTA')))

    def test_as_string(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('TCAAAAGGATGCATCATG', self.mutable_s)

    def test_length(self):
        if False:
            while True:
                i = 10
        self.assertEqual(18, len(self.mutable_s))

    def test_converting_to_immutable(self):
        if False:
            while True:
                i = 10
        self.assertIsInstance(Seq.Seq(self.mutable_s), Seq.Seq)

    def test_first_nucleotide(self):
        if False:
            while True:
                i = 10
        self.assertEqual('T', self.mutable_s[0])

    def test_setting_slices(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(Seq.MutableSeq('CAAA'), self.mutable_s[1:5], 'Slice mutable seq')
        self.mutable_s[1:3] = 'GAT'
        self.assertEqual(Seq.MutableSeq('TGATAAAGGATGCATCATG'), self.mutable_s, 'Set slice with string and adding extra nucleotide')
        self.mutable_s[1:3] = self.mutable_s[5:7]
        self.assertEqual(Seq.MutableSeq('TAATAAAGGATGCATCATG'), self.mutable_s, 'Set slice with MutableSeq')
        if numpy is not None:
            (one, three, five, seven) = numpy.array([1, 3, 5, 7])
            self.assertEqual(Seq.MutableSeq('AATA'), self.mutable_s[one:five], 'Slice mutable seq')
            self.mutable_s[one:three] = 'GAT'
            self.assertEqual(Seq.MutableSeq('TGATTAAAGGATGCATCATG'), self.mutable_s, 'Set slice with string and adding extra nucleotide')
            self.mutable_s[one:three] = self.mutable_s[five:seven]
            self.assertEqual(Seq.MutableSeq('TAATTAAAGGATGCATCATG'), self.mutable_s, 'Set slice with MutableSeq')

    def test_setting_item(self):
        if False:
            for i in range(10):
                print('nop')
        self.mutable_s[3] = 'G'
        self.assertEqual(Seq.MutableSeq('TCAGAAGGATGCATCATG'), self.mutable_s)
        if numpy is not None:
            i = numpy.intc(3)
            self.mutable_s[i] = 'X'
            self.assertEqual(Seq.MutableSeq('TCAXAAGGATGCATCATG'), self.mutable_s)

    def test_deleting_slice(self):
        if False:
            print('Hello World!')
        del self.mutable_s[4:5]
        self.assertEqual(Seq.MutableSeq('TCAAAGGATGCATCATG'), self.mutable_s)

    def test_deleting_item(self):
        if False:
            i = 10
            return i + 15
        del self.mutable_s[3]
        self.assertEqual(Seq.MutableSeq('TCAAAGGATGCATCATG'), self.mutable_s)

    def test_appending(self):
        if False:
            for i in range(10):
                print('nop')
        self.mutable_s.append('C')
        self.assertEqual(Seq.MutableSeq('TCAAAAGGATGCATCATGC'), self.mutable_s)

    def test_inserting(self):
        if False:
            print('Hello World!')
        self.mutable_s.insert(4, 'G')
        self.assertEqual(Seq.MutableSeq('TCAAGAAGGATGCATCATG'), self.mutable_s)

    def test_popping_last_item(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('G', self.mutable_s.pop())

    def test_remove_items(self):
        if False:
            while True:
                i = 10
        self.mutable_s.remove('G')
        self.assertEqual(Seq.MutableSeq('TCAAAAGATGCATCATG'), self.mutable_s, 'Remove first G')
        self.assertRaises(ValueError, self.mutable_s.remove, 'Z')

    def test_count(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(7, self.mutable_s.count('A'))
        self.assertEqual(2, self.mutable_s.count('AA'))

    def test_index(self):
        if False:
            while True:
                i = 10
        self.assertEqual(2, self.mutable_s.index('A'))
        self.assertRaises(ValueError, self.mutable_s.index, '8888')

    def test_reverse(self):
        if False:
            print('Hello World!')
        'Test using reverse method.'
        self.mutable_s.reverse()
        self.assertEqual(Seq.MutableSeq('GTACTACGTAGGAAAACT'), self.mutable_s)

    def test_reverse_with_stride(self):
        if False:
            i = 10
            return i + 15
        'Test reverse using -1 stride.'
        self.assertEqual(Seq.MutableSeq('GTACTACGTAGGAAAACT'), self.mutable_s[::-1])

    def test_complement_old(self):
        if False:
            i = 10
            return i + 15
        with self.assertWarns(BiopythonDeprecationWarning):
            self.mutable_s.complement()
        self.assertEqual('AGTTTTCCTACGTAGTAC', self.mutable_s)

    def test_complement(self):
        if False:
            i = 10
            return i + 15
        self.mutable_s.complement(inplace=True)
        self.assertEqual('AGTTTTCCTACGTAGTAC', self.mutable_s)

    def test_complement_rna(self):
        if False:
            for i in range(10):
                print('nop')
        m = self.mutable_s.complement_rna()
        self.assertEqual(self.mutable_s, 'TCAAAAGGATGCATCATG')
        self.assertIsInstance(m, Seq.MutableSeq)
        self.assertEqual(m, 'AGUUUUCCUACGUAGUAC')
        m = self.mutable_s.complement_rna(inplace=True)
        self.assertEqual(self.mutable_s, 'AGUUUUCCUACGUAGUAC')
        self.assertIsInstance(m, Seq.MutableSeq)
        self.assertEqual(m, 'AGUUUUCCUACGUAGUAC')

    def test_reverse_complement_rna(self):
        if False:
            for i in range(10):
                print('nop')
        m = self.mutable_s.reverse_complement_rna()
        self.assertEqual(self.mutable_s, 'TCAAAAGGATGCATCATG')
        self.assertIsInstance(m, Seq.MutableSeq)
        self.assertEqual(m, 'CAUGAUGCAUCCUUUUGA')
        m = self.mutable_s.reverse_complement_rna(inplace=True)
        self.assertEqual(self.mutable_s, 'CAUGAUGCAUCCUUUUGA')
        self.assertIsInstance(m, Seq.MutableSeq)
        self.assertEqual(m, 'CAUGAUGCAUCCUUUUGA')

    def test_transcribe(self):
        if False:
            return 10
        r = self.mutable_s.transcribe()
        self.assertEqual(self.mutable_s, 'TCAAAAGGATGCATCATG')
        self.assertIsInstance(r, Seq.MutableSeq)
        self.assertEqual(r, 'UCAAAAGGAUGCAUCAUG')
        r = self.mutable_s.transcribe(inplace=True)
        self.assertEqual(self.mutable_s, 'UCAAAAGGAUGCAUCAUG')
        self.assertIsInstance(r, Seq.MutableSeq)
        self.assertEqual(r, 'UCAAAAGGAUGCAUCAUG')
        d = self.mutable_s.back_transcribe()
        self.assertEqual(self.mutable_s, 'UCAAAAGGAUGCAUCAUG')
        self.assertIsInstance(d, Seq.MutableSeq)
        self.assertEqual(d, 'TCAAAAGGATGCATCATG')
        d = self.mutable_s.back_transcribe(inplace=True)
        self.assertEqual(self.mutable_s, 'TCAAAAGGATGCATCATG')
        self.assertIsInstance(d, Seq.MutableSeq)
        self.assertEqual(d, 'TCAAAAGGATGCATCATG')

    def test_complement_mixed_aphabets(self):
        if False:
            for i in range(10):
                print('nop')
        seq = Seq.MutableSeq('AUGaaaCTG')
        seq.complement_rna(inplace=True)
        self.assertEqual('UACuuuGAC', seq)
        seq = Seq.MutableSeq('AUGaaaCTG')
        with self.assertWarns(BiopythonDeprecationWarning):
            with self.assertRaises(ValueError):
                seq.complement()

    def test_complement_rna_string(self):
        if False:
            while True:
                i = 10
        seq = Seq.MutableSeq('AUGaaaCUG')
        seq.complement_rna(inplace=True)
        self.assertEqual('UACuuuGAC', seq)
        seq = Seq.MutableSeq('AUGaaaCUG')
        with self.assertWarns(BiopythonDeprecationWarning):
            seq.complement()
        self.assertEqual('UACuuuGAC', seq)

    def test_complement_dna_string(self):
        if False:
            i = 10
            return i + 15
        seq = Seq.MutableSeq('ATGaaaCTG')
        seq.complement(inplace=True)
        self.assertEqual('TACtttGAC', seq)
        seq = Seq.MutableSeq('ATGaaaCTG')
        with self.assertWarns(BiopythonDeprecationWarning):
            seq.complement()
        self.assertEqual('TACtttGAC', seq)

    def test_reverse_complement(self):
        if False:
            print('Hello World!')
        self.mutable_s.reverse_complement(inplace=True)
        self.assertEqual('CATGATGCATCCTTTTGA', self.mutable_s)

    def test_reverse_complement_old(self):
        if False:
            while True:
                i = 10
        with self.assertWarns(BiopythonDeprecationWarning):
            self.mutable_s.reverse_complement()
        self.assertEqual('CATGATGCATCCTTTTGA', self.mutable_s)

    def test_extend_method(self):
        if False:
            print('Hello World!')
        self.mutable_s.extend('GAT')
        self.assertEqual(Seq.MutableSeq('TCAAAAGGATGCATCATGGAT'), self.mutable_s)

    def test_extend_with_mutable_seq(self):
        if False:
            return 10
        self.mutable_s.extend(Seq.MutableSeq('TTT'))
        self.assertEqual(Seq.MutableSeq('TCAAAAGGATGCATCATGTTT'), self.mutable_s)

    def test_delete_stride_slice(self):
        if False:
            return 10
        del self.mutable_s[4:6 - 1]
        self.assertEqual(Seq.MutableSeq('TCAAAGGATGCATCATG'), self.mutable_s)

    def test_extract_third_nucleotide(self):
        if False:
            while True:
                i = 10
        'Test extracting every third nucleotide (slicing with stride 3).'
        self.assertEqual(Seq.MutableSeq('TAGTAA'), self.mutable_s[0::3])
        self.assertEqual(Seq.MutableSeq('CAGGTT'), self.mutable_s[1::3])
        self.assertEqual(Seq.MutableSeq('AAACCG'), self.mutable_s[2::3])

    def test_set_wobble_codon_to_n(self):
        if False:
            return 10
        'Test setting wobble codon to N (set slice with stride 3).'
        self.mutable_s[2::3] = 'N' * len(self.mutable_s[2::3])
        self.assertEqual(Seq.MutableSeq('TCNAANGGNTGNATNATN'), self.mutable_s)
        if numpy is not None:
            (start, step) = numpy.array([2, 3])
            self.mutable_s[start::step] = 'X' * len(self.mutable_s[2::3])
            self.assertEqual(Seq.MutableSeq('TCXAAXGGXTGXATXATX'), self.mutable_s)

class TestAmbiguousComplements(unittest.TestCase):

    def test_ambiguous_values(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that other tests do not introduce characters to our values.'
        self.assertNotIn('-', ambiguous_dna_values)
        self.assertNotIn('?', ambiguous_dna_values)

class TestComplement(unittest.TestCase):

    def test_complement_ambiguous_dna_values(self):
        if False:
            return 10
        for (ambig_char, values) in sorted(ambiguous_dna_values.items()):
            compl_values = Seq.Seq(values).complement()
            ambig_values = ambiguous_dna_values[ambiguous_dna_complement[ambig_char]]
            self.assertCountEqual(compl_values, ambig_values)

    def test_complement_ambiguous_rna_values(self):
        if False:
            for i in range(10):
                print('nop')
        for (ambig_char, values) in sorted(ambiguous_rna_values.items()):
            if 'u' in values or 'U' in values:
                compl_values = Seq.Seq(values).complement_rna().transcribe()
            else:
                compl_values = Seq.Seq(values).complement().transcribe()
            ambig_values = ambiguous_rna_values[ambiguous_rna_complement[ambig_char]]
            self.assertCountEqual(compl_values, ambig_values)

    def test_complement_incompatible_letters(self):
        if False:
            print('Hello World!')
        seq = Seq.Seq('CAGGTU')
        dna = seq.complement(inplace=False)
        self.assertEqual('GTCCAA', dna)
        rna = seq.complement_rna()
        self.assertEqual('GUCCAA', rna)
        with self.assertWarns(BiopythonDeprecationWarning):
            with self.assertRaises(ValueError):
                seq.complement()

    def test_complement_of_mixed_dna_rna(self):
        if False:
            return 10
        seq = 'AUGAAACTG'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=BiopythonDeprecationWarning)
            self.assertRaises(ValueError, Seq.complement, seq)

    def test_complement_of_rna(self):
        if False:
            print('Hello World!')
        seq = 'AUGAAACUG'
        rna = Seq.complement_rna(seq)
        self.assertEqual('UACUUUGAC', rna)
        with self.assertWarns(BiopythonDeprecationWarning):
            rna = Seq.complement(seq)
        self.assertEqual('UACUUUGAC', rna)

    def test_complement_of_dna(self):
        if False:
            for i in range(10):
                print('nop')
        seq = 'ATGAAACTG'
        self.assertEqual('TACTTTGAC', Seq.complement(seq))

    def test_immutable(self):
        if False:
            while True:
                i = 10
        from Bio.SeqRecord import SeqRecord
        r = SeqRecord(Seq.Seq('ACGT'))
        with self.assertRaises(TypeError) as cm:
            Seq.complement(r, inplace=True)
        self.assertEqual(str(cm.exception), 'SeqRecords are immutable')
        with self.assertRaises(TypeError) as cm:
            Seq.complement('ACGT', inplace=True)
        self.assertEqual(str(cm.exception), 'strings are immutable')
        with self.assertRaises(TypeError) as cm:
            Seq.complement_rna(r, inplace=True)
        self.assertEqual(str(cm.exception), 'SeqRecords are immutable')
        with self.assertRaises(TypeError) as cm:
            Seq.complement_rna('ACGT', inplace=True)
        self.assertEqual(str(cm.exception), 'strings are immutable')

class TestReverseComplement(unittest.TestCase):

    def test_reverse_complement(self):
        if False:
            i = 10
            return i + 15
        test_seqs_copy = copy.copy(test_seqs)
        test_seqs_copy.pop(13)
        for nucleotide_seq in test_seqs_copy:
            if not isinstance(nucleotide_seq, Seq.Seq):
                continue
            if 'u' in nucleotide_seq or 'U' in nucleotide_seq:
                expected = Seq.reverse_complement_rna(nucleotide_seq)
                self.assertEqual(repr(expected), repr(nucleotide_seq.reverse_complement_rna()))
                self.assertEqual(repr(expected[::-1]), repr(nucleotide_seq.complement_rna()))
                self.assertEqual(nucleotide_seq.complement_rna(), Seq.reverse_complement_rna(nucleotide_seq)[::-1])
                self.assertEqual(nucleotide_seq.reverse_complement_rna(), Seq.reverse_complement_rna(nucleotide_seq))
            else:
                expected = Seq.reverse_complement(nucleotide_seq)
                self.assertEqual(repr(expected), repr(nucleotide_seq.reverse_complement()))
                self.assertEqual(repr(expected[::-1]), repr(nucleotide_seq.complement()))
                self.assertEqual(nucleotide_seq.complement(), Seq.reverse_complement(nucleotide_seq)[::-1])
                self.assertEqual(nucleotide_seq.reverse_complement(), Seq.reverse_complement(nucleotide_seq))

    def test_reverse_complement_of_mixed_dna_rna(self):
        if False:
            return 10
        seq = 'AUGAAACTG'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=BiopythonDeprecationWarning)
            self.assertRaises(ValueError, Seq.reverse_complement, seq)

    def test_reverse_complement_of_rna(self):
        if False:
            while True:
                i = 10
        seq = 'AUGAAACUG'
        with self.assertWarns(BiopythonDeprecationWarning):
            rna = Seq.reverse_complement(seq)
        self.assertEqual('CAGUUUCAU', rna)
        dna = Seq.reverse_complement(seq, inplace=False)
        self.assertEqual('CAGTTTCAT', dna)

    def test_reverse_complement_of_dna(self):
        if False:
            for i in range(10):
                print('nop')
        seq = 'ATGAAACTG'
        self.assertEqual('CAGTTTCAT', Seq.reverse_complement(seq))

    def test_immutable(self):
        if False:
            while True:
                i = 10
        from Bio.SeqRecord import SeqRecord
        r = SeqRecord(Seq.Seq('ACGT'))
        with self.assertRaises(TypeError) as cm:
            Seq.reverse_complement(r, inplace=True)
        self.assertEqual(str(cm.exception), 'SeqRecords are immutable')
        with self.assertRaises(TypeError) as cm:
            Seq.reverse_complement('ACGT', inplace=True)
        self.assertEqual(str(cm.exception), 'strings are immutable')
        with self.assertRaises(TypeError) as cm:
            Seq.reverse_complement_rna(r, inplace=True)
        self.assertEqual(str(cm.exception), 'SeqRecords are immutable')
        with self.assertRaises(TypeError) as cm:
            Seq.reverse_complement_rna('ACGT', inplace=True)
        self.assertEqual(str(cm.exception), 'strings are immutable')

class TestDoubleReverseComplement(unittest.TestCase):

    def test_reverse_complements(self):
        if False:
            for i in range(10):
                print('nop')
        'Test double reverse complement preserves the sequence.'
        sorted_amb_rna = sorted(ambiguous_rna_values)
        sorted_amb_dna = sorted(ambiguous_dna_values)
        for sequence in [Seq.Seq(''.join(sorted_amb_dna)), Seq.Seq(''.join(sorted_amb_dna).replace('X', '')), Seq.Seq('AWGAARCKG')]:
            reversed_sequence = sequence.reverse_complement()
            self.assertEqual(sequence, reversed_sequence.reverse_complement())
        for sequence in [Seq.Seq(''.join(sorted_amb_rna)), Seq.Seq(''.join(sorted_amb_rna).replace('X', '')), Seq.Seq('AWGAARCKG')]:
            reversed_sequence = sequence.reverse_complement_rna()
            self.assertEqual(sequence, reversed_sequence.reverse_complement_rna())

class TestTranscription(unittest.TestCase):

    def test_transcription_dna_into_rna(self):
        if False:
            while True:
                i = 10
        for nucleotide_seq in test_seqs:
            expected = Seq.transcribe(nucleotide_seq)
            self.assertEqual(str(nucleotide_seq).replace('t', 'u').replace('T', 'U'), expected)

    def test_transcription_dna_string_into_rna(self):
        if False:
            return 10
        seq = 'ATGAAACTG'
        self.assertEqual('AUGAAACUG', Seq.transcribe(seq))

    def test_seq_object_transcription_method(self):
        if False:
            for i in range(10):
                print('nop')
        for nucleotide_seq in test_seqs:
            if isinstance(nucleotide_seq, Seq.Seq):
                self.assertEqual(repr(Seq.transcribe(nucleotide_seq)), repr(nucleotide_seq.transcribe()))

    def test_back_transcribe_rna_into_dna(self):
        if False:
            return 10
        for nucleotide_seq in test_seqs:
            expected = Seq.back_transcribe(nucleotide_seq)
            self.assertEqual(str(nucleotide_seq).replace('u', 't').replace('U', 'T'), expected)

    def test_back_transcribe_rna_string_into_dna(self):
        if False:
            while True:
                i = 10
        seq = 'AUGAAACUG'
        self.assertEqual('ATGAAACTG', Seq.back_transcribe(seq))

    def test_seq_object_back_transcription_method(self):
        if False:
            while True:
                i = 10
        for nucleotide_seq in test_seqs:
            if isinstance(nucleotide_seq, Seq.Seq):
                expected = Seq.back_transcribe(nucleotide_seq)
                self.assertEqual(repr(nucleotide_seq.back_transcribe()), repr(expected))

class TestTranslating(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_seqs = [Seq.Seq('TCAAAAGGATGCATCATG'), Seq.Seq('ATGAAACTG'), Seq.Seq('ATGAARCTG'), Seq.Seq('AWGAARCKG'), Seq.Seq(''.join(ambiguous_rna_values)), Seq.Seq(''.join(ambiguous_dna_values)), Seq.Seq('AUGAAACUG'), Seq.Seq('ATGAAACTGWN'), Seq.Seq('AUGAAACUGWN'), Seq.MutableSeq('ATGAAACTG'), Seq.MutableSeq('AUGaaaCUG')]

    def test_translation(self):
        if False:
            while True:
                i = 10
        for nucleotide_seq in self.test_seqs:
            nucleotide_seq = nucleotide_seq[:3 * (len(nucleotide_seq) // 3)]
            if 'X' not in nucleotide_seq:
                expected = Seq.translate(nucleotide_seq)
                self.assertEqual(expected, nucleotide_seq.translate())

    def test_gapped_seq_with_gap_char_given(self):
        if False:
            return 10
        seq = Seq.Seq('ATG---AAACTG')
        self.assertEqual('M-KL', seq.translate(gap='-'))
        self.assertRaises(TranslationError, seq.translate, gap='~')
        seq = Seq.Seq('GTG---GCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')
        self.assertEqual('V-AIVMGR*KGAR*', seq.translate(gap='-'))
        self.assertRaises(TranslationError, seq.translate, gap=None)
        seq = Seq.Seq('ATG~~~AAACTG')
        self.assertRaises(TranslationError, seq.translate, gap='-')
        seq = Seq.Seq('ATG---AAACTGTAG')
        self.assertEqual('M-KL*', seq.translate(gap='-'))
        self.assertEqual('M-KL@', seq.translate(gap='-', stop_symbol='@'))
        self.assertRaises(TranslationError, seq.translate, gap='~')
        seq = Seq.Seq('ATG~~~AAACTGTAG')
        self.assertRaises(TranslationError, seq.translate, gap='-')

    def test_gapped_seq_no_gap_char_given(self):
        if False:
            print('Hello World!')
        seq = Seq.Seq('ATG---AAACTG')
        self.assertRaises(TranslationError, seq.translate, gap=None)

    def test_translation_wrong_type(self):
        if False:
            for i in range(10):
                print('nop')
        'Test translation table cannot be CodonTable.'
        seq = Seq.Seq('ATCGTA')
        with self.assertRaises(ValueError):
            seq.translate(table=ambiguous_dna_complement)

    def test_translation_of_string(self):
        if False:
            return 10
        seq = 'GTGGCCATTGTAATGGGCCGC'
        self.assertEqual('VAIVMGR', Seq.translate(seq))

    def test_translation_of_gapped_string_with_gap_char_given(self):
        if False:
            return 10
        seq = 'GTG---GCCATTGTAATGGGCCGC'
        expected = 'V-AIVMGR'
        self.assertEqual(expected, Seq.translate(seq, gap='-'))
        self.assertRaises(TypeError, Seq.translate, seq, gap=[])
        self.assertRaises(ValueError, Seq.translate, seq, gap='-*')

    def test_translation_of_gapped_string_no_gap_char_given(self):
        if False:
            for i in range(10):
                print('nop')
        seq = 'GTG---GCCATTGTAATGGGCCGC'
        self.assertRaises(TranslationError, Seq.translate, seq)

    def test_translation_to_stop(self):
        if False:
            i = 10
            return i + 15
        for nucleotide_seq in self.test_seqs:
            nucleotide_seq = nucleotide_seq[:3 * (len(nucleotide_seq) // 3)]
            if 'X' not in nucleotide_seq:
                short = Seq.translate(nucleotide_seq, to_stop=True)
                self.assertEqual(short, Seq.translate(nucleotide_seq).split('*')[0])
        seq = 'GTGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG'
        self.assertEqual('VAIVMGRWKGAR', Seq.translate(seq, table=2, to_stop=True))

    def test_translation_on_proteins(self):
        if False:
            return 10
        'Check translation fails on a protein.'
        for s in protein_seqs:
            if len(s) % 3 != 0:
                with self.assertWarns(BiopythonWarning):
                    with self.assertRaises(TranslationError):
                        Seq.translate(s)
                with self.assertWarns(BiopythonWarning):
                    with self.assertRaises(TranslationError):
                        s.translate()
            else:
                with self.assertRaises(TranslationError):
                    Seq.translate(s)
                with self.assertRaises(TranslationError):
                    s.translate()

    def test_translation_of_invalid_codon(self):
        if False:
            print('Hello World!')
        for codon in ['TA?', 'N-N', 'AC_', 'Ac_']:
            with self.assertRaises(TranslationError):
                Seq.translate(codon)

    def test_translation_of_glutamine(self):
        if False:
            for i in range(10):
                print('nop')
        for codon in ['SAR', 'SAG', 'SAA']:
            self.assertEqual('Z', Seq.translate(codon))

    def test_translation_of_asparagine(self):
        if False:
            return 10
        for codon in ['RAY', 'RAT', 'RAC']:
            self.assertEqual('B', Seq.translate(codon))

    def test_translation_of_leucine(self):
        if False:
            for i in range(10):
                print('nop')
        for codon in ['WTA', 'MTY', 'MTT', 'MTW', 'MTM', 'MTH', 'MTA', 'MTC', 'HTA']:
            self.assertEqual('J', Seq.translate(codon))

    def test_translation_with_bad_table_argument(self):
        if False:
            for i in range(10):
                print('nop')
        table = {}
        with self.assertRaises(ValueError) as cm:
            Seq.translate('GTGGCCATTGTAATGGGCCGC', table=table)
        self.assertEqual(str(cm.exception), 'Bad table argument')
        table = b'0x'
        with self.assertRaises(TypeError) as cm:
            Seq.translate('GTGGCCATTGTAATGGGCCGC', table=table)
        self.assertEqual(str(cm.exception), 'table argument must be integer or string')

    def test_translation_with_codon_table_as_table_argument(self):
        if False:
            for i in range(10):
                print('nop')
        table = standard_dna_table
        self.assertEqual('VAIVMGR', Seq.translate('GTGGCCATTGTAATGGGCCGC', table=table))

    def test_translation_incomplete_codon(self):
        if False:
            return 10
        with self.assertWarns(BiopythonWarning):
            Seq.translate('GTGGCCATTGTAATGGGCCG')

    def test_translation_extra_stop_codon(self):
        if False:
            print('Hello World!')
        seq = 'GTGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGTAG'
        with self.assertRaises(TranslationError):
            Seq.translate(seq, table=2, cds=True)

    def test_translation_using_cds(self):
        if False:
            print('Hello World!')
        seq = 'GTGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG'
        self.assertEqual('MAIVMGRWKGAR', Seq.translate(seq, table=2, cds=True))
        seq = 'GTGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCG'
        with self.assertRaises(TranslationError):
            Seq.translate(seq, table=2, cds=True)
        seq = 'GTGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGA'
        with self.assertRaises(TranslationError):
            Seq.translate(seq, table=2, cds=True)
        seq = 'GCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG'
        with self.assertRaises(TranslationError):
            Seq.translate(seq, table=2, cds=True)

    def test_translation_using_tables_with_ambiguous_stop_codons(self):
        if False:
            return 10
        "Check for error and warning messages.\n\n        Here, 'ambiguous stop codons' means codons of unambiguous sequence\n        but with a context sensitive encoding as STOP or an amino acid.\n        Thus, these codons appear within the codon table in the forward\n        table as well as in the list of stop codons.\n        "
        seq = 'ATGGGCTGA'
        with self.assertRaises(ValueError):
            Seq.translate(seq, table=28, to_stop=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Seq.translate(seq, table=28)
            message = str(w[-1].message)
            self.assertTrue(message.startswith('This table contains'))
            self.assertTrue(message.endswith('be translated as amino acid.'))

class TestStopCodons(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.misc_stops = 'TAATAGTGAAGAAGG'

    def test_stops(self):
        if False:
            i = 10
            return i + 15
        for nucleotide_seq in [self.misc_stops, Seq.Seq(self.misc_stops)]:
            self.assertEqual('***RR', Seq.translate(nucleotide_seq))
            self.assertEqual('***RR', Seq.translate(nucleotide_seq, table=1))
            self.assertEqual('***RR', Seq.translate(nucleotide_seq, table='SGC0'))
            self.assertEqual('**W**', Seq.translate(nucleotide_seq, table=2))
            self.assertEqual('**WRR', Seq.translate(nucleotide_seq, table='Yeast Mitochondrial'))
            self.assertEqual('**WSS', Seq.translate(nucleotide_seq, table=5))
            self.assertEqual('**WSS', Seq.translate(nucleotide_seq, table=9))
            self.assertEqual('**CRR', Seq.translate(nucleotide_seq, table='Euplotid Nuclear'))
            self.assertEqual('***RR', Seq.translate(nucleotide_seq, table=11))
            self.assertEqual('***RR', Seq.translate(nucleotide_seq, table='Bacterial'))

    def test_translation_of_stops(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(Seq.translate('TAT'), 'Y')
        self.assertEqual(Seq.translate('TAR'), '*')
        self.assertEqual(Seq.translate('TAN'), 'X')
        self.assertEqual(Seq.translate('NNN'), 'X')
        self.assertEqual(Seq.translate('TAt'), 'Y')
        self.assertEqual(Seq.translate('TaR'), '*')
        self.assertEqual(Seq.translate('TaN'), 'X')
        self.assertEqual(Seq.translate('nnN'), 'X')
        self.assertEqual(Seq.translate('tat'), 'Y')
        self.assertEqual(Seq.translate('tar'), '*')
        self.assertEqual(Seq.translate('tan'), 'X')
        self.assertEqual(Seq.translate('nnn'), 'X')

class TestAttributes(unittest.TestCase):

    def test_seq(self):
        if False:
            print('Hello World!')
        s = Seq.Seq('ACGT')
        with self.assertRaises(AttributeError):
            s.dog
        s.dog = 'woof'
        self.assertIn('dog', dir(s))
        self.assertEqual(s.dog, 'woof')
        del s.dog
        with self.assertRaises(AttributeError):
            s.dog
        self.assertNotIn('dog', dir(s))
        with self.assertRaises(AttributeError):
            s.cat
        s.dog = 'woof'
        s.cat = 'meow'
        self.assertIn('dog', dir(s))
        self.assertIn('cat', dir(s))
        self.assertEqual(s.dog, 'woof')
        self.assertEqual(s.cat, 'meow')
        del s.dog
        with self.assertRaises(AttributeError):
            s.dog
        self.assertNotIn('dog', dir(s))
        self.assertIn('cat', dir(s))
        self.assertEqual(s.cat, 'meow')
        del s.cat
        with self.assertRaises(AttributeError):
            s.cat
        self.assertNotIn('cat', dir(s))
        s.dog = 'woof'
        s.dog = 'bark'
        self.assertIn('dog', dir(s))
        self.assertEqual(s.dog, 'bark')
        del s.dog
        with self.assertRaises(AttributeError):
            s.dog
        self.assertNotIn('dog', dir(s))

    def test_mutable_seq(self):
        if False:
            return 10
        s = Seq.MutableSeq('ACGT')
        with self.assertRaises(AttributeError):
            s.dog
        s.dog = 'woof'
        self.assertIn('dog', dir(s))
        self.assertEqual(s.dog, 'woof')
        del s.dog
        with self.assertRaises(AttributeError):
            s.dog
        self.assertNotIn('dog', dir(s))
        with self.assertRaises(AttributeError):
            s.cat
        s.dog = 'woof'
        s.cat = 'meow'
        self.assertIn('dog', dir(s))
        self.assertIn('cat', dir(s))
        self.assertEqual(s.dog, 'woof')
        self.assertEqual(s.cat, 'meow')
        del s.dog
        with self.assertRaises(AttributeError):
            s.dog
        self.assertNotIn('dog', dir(s))
        self.assertIn('cat', dir(s))
        self.assertEqual(s.cat, 'meow')
        del s.cat
        with self.assertRaises(AttributeError):
            s.cat
        self.assertNotIn('cat', dir(s))
        s.dog = 'woof'
        s.dog = 'bark'
        self.assertIn('dog', dir(s))
        self.assertEqual(s.dog, 'bark')
        del s.dog
        with self.assertRaises(AttributeError):
            s.dog
        self.assertNotIn('dog', dir(s))

class TestSeqDefined(unittest.TestCase):

    def test_zero_length(self):
        if False:
            for i in range(10):
                print('nop')
        zero_length_seqs = [Seq.Seq(''), Seq.Seq(None, length=0), Seq.Seq({}, length=0), Seq.MutableSeq('')]
        for seq in zero_length_seqs:
            self.assertTrue(seq.defined, msg=repr(seq))
            self.assertEqual(seq.defined_ranges, (), msg=repr(seq))

    def test_undefined(self):
        if False:
            print('Hello World!')
        seq = Seq.Seq(None, length=1)
        self.assertFalse(seq.defined)
        self.assertEqual(seq.defined_ranges, ())
        seq = Seq.Seq({3: 'ACGT'}, length=10)
        self.assertFalse(seq.defined)
        self.assertEqual(seq.defined_ranges, ((3, 7),))

    def test_defined(self):
        if False:
            while True:
                i = 10
        seqs = [Seq.Seq('T'), Seq.Seq({0: 'A'}, length=1), Seq.Seq({0: 'A', 1: 'C'}, length=2)]
        for seq in seqs:
            self.assertTrue(seq.defined, msg=repr(seq))
            self.assertEqual(seq.defined_ranges, ((0, len(seq)),), msg=repr(seq))
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)