"""Calculate the melting temperature of nucleotide sequences.

This module contains three different methods to calculate the melting
temperature of oligonucleotides:

1. Tm_Wallace: 'Rule of thumb'
2. Tm_GC: Empirical formulas based on GC content. Salt and mismatch corrections
   can be included.
3. Tm_NN: Calculation based on nearest neighbor thermodynamics. Several tables
   for DNA/DNA, DNA/RNA and RNA/RNA hybridizations are included.
   Correction for mismatches, dangling ends, salt concentration and other
   additives are available.

General parameters for most Tm methods:
 - seq -- A Biopython sequence object or a string.
 - check -- Checks if the sequence is valid for the given method (default=
   True). In general, whitespaces and non-base characters are removed and
   characters are converted to uppercase. RNA will be backtranscribed.
 - strict -- Do not allow base characters or neighbor duplex keys (e.g.
   'AT/NA') that could not or not unambigiously be evaluated for the respective
   method (default=True). Note that W (= A or T) and S (= C or G) are not
   ambiguous for Tm_Wallace and Tm_GC. If 'False', average values (if
   applicable) will be used.

This module is not able to detect self-complementary and it will not use
alignment tools to align an oligonucleotide sequence to its target sequence.
Thus it can not detect dangling-ends and mismatches by itself (don't even think
about bulbs and loops). These parameters have to be handed over to the
respective method.

Other public methods of this module:
 - make_table     : To create a table with thermodynamic data.
 - salt_correction: To adjust Tm to a given salt concentration by different
   formulas. This method is called from Tm_GC and Tm_NN but may
   also be accessed 'manually'. It returns a correction term, not
   a corrected Tm!
 - chem_correction: To adjust Tm regarding the chemical additives DMSO and
   formaldehyde. The method returns a corrected Tm. Chemical
   correction is not an integral part of the Tm methods and must
   be called additionally.

For example:

    >>> from Bio.SeqUtils import MeltingTemp as mt
    >>> from Bio.Seq import Seq
    >>> mystring = 'CGTTCCAAAGATGTGGGCATGAGCTTAC'
    >>> myseq = Seq(mystring)
    >>> print('%0.2f' % mt.Tm_Wallace(mystring))
    84.00
    >>> print('%0.2f' % mt.Tm_Wallace(myseq))
    84.00
    >>> print('%0.2f' % mt.Tm_GC(myseq))
    58.97
    >>> print('%0.2f' % mt.Tm_NN(myseq))
    60.32

Using different thermodynamic tables, e.g. from Breslauer '86 or Sugimoto '96:

    >>> print('%0.2f' % mt.Tm_NN(myseq, nn_table=mt.DNA_NN1))  # Breslauer '86
    72.19
    >>> print('%0.2f' % mt.Tm_NN(myseq, nn_table=mt.DNA_NN2))  # Sugimoto '96
    65.47

Tables for RNA and RNA/DNA hybrids are included:

    >>> print('%0.2f' % mt.Tm_NN(myseq, nn_table=mt.RNA_NN1))  # Freier '86
    73.35
    >>> print('%0.2f' % mt.Tm_NN(myseq, nn_table=mt.R_DNA_NN1))  # Sugimoto '95
    58.45

Several types of salc correction (for Tm_NN and Tm_GC):

    >>> for i in range(1, 8):
    ...     print("Type: %d, Tm: %0.2f" % (i, Tm_NN(myseq, saltcorr=i)))
    ...
    Type: 1, Tm: 54.27
    Type: 2, Tm: 54.02
    Type: 3, Tm: 59.60
    Type: 4, Tm: 60.64
    Type: 5, Tm: 60.32
    Type: 6, Tm: 59.78
    Type: 7, Tm: 59.78

Correction for other monovalent cations (K+, Tris), Mg2+ and dNTPs according
to von Ahsen et al. (2001) or Owczarzy et al. (2008) (for Tm_NN and Tm_GC):

    >>> print('%0.2f' % mt.Tm_NN(myseq, Na=50, Tris=10))
    60.79
    >>> print('%0.2f' % mt.Tm_NN(myseq, Na=50, Tris=10, Mg=1.5))
    67.39
    >>> print('%0.2f' % mt.Tm_NN(myseq, Na=50, Tris=10, Mg=1.5, saltcorr=7))
    66.81
    >>> print('%0.2f' % mt.Tm_NN(myseq, Na=50, Tris=10, Mg=1.5, dNTPs=0.6,
    ...                          saltcorr=7))
    66.04

Dangling ends and mismatches, e.g.::

    Oligo:     CGTTCCaAAGATGTGGGCATGAGCTTAC       CGTTCCaAAGATGTGGGCATGAGCTTAC
               ::::::X:::::::::::::::::::::  or   ::::::X:::::::::::::::::::::
    Template:  GCAAGGcTTCTACACCCGTACTCGAATG      TGCAAGGcTTCTACACCCGTACTCGAATGC

Here:

    >>> print('%0.2f' % mt.Tm_NN('CGTTCCAAAGATGTGGGCATGAGCTTAC'))
    60.32
    >>> print('%0.2f' % mt.Tm_NN('CGTTCCAAAGATGTGGGCATGAGCTTAC',
    ...                    c_seq='GCAAGGcTTCTACACCCGTACTCGAATG'))
    55.39
    >>> print('%0.2f' % mt.Tm_NN('CGTTCCAAAGATGTGGGCATGAGCTTAC', shift=1,
    ...                   c_seq='TGCAAGGcTTCTACACCCGTACTCGAATGC'))
    55.69

The same for RNA:

    >>> print('%0.2f' % mt.Tm_NN('CGUUCCAAAGAUGUGGGCAUGAGCUUAC',
    ...                   c_seq='UGCAAGGcUUCUACACCCGUACUCGAAUGC',
    ...                   shift=1, nn_table=mt.RNA_NN3,
    ...                   de_table=mt.RNA_DE1))
    73.00

Note, that thermodynamic data are not available for all kind of mismatches,
e.g. most double mismatches or terminal mismatches combined with dangling ends:

    >>> print('%0.2f' % mt.Tm_NN('CGTTCCAAAGATGTGGGCATGAGCTTAC',
    ...                   c_seq='TtCAAGGcTTCTACACCCGTACTCGAATGC',
    ...                   shift=1))
    Traceback (most recent call last):
    ValueError: no thermodynamic data for neighbors '.C/TT' available

Make your own tables, or update/extend existing tables. E.g., add values for
locked nucleotides. Here, 'locked A' (and its complement) should be represented
by '1':

    >>> mytable = mt.make_table(oldtable=mt.DNA_NN3,
    ...                         values={'A1/T1':(-6.608, -17.235),
    ...                         '1A/1T':(-6.893, -15.923)})
    >>> print('%0.2f' % mt.Tm_NN('CGTTCCAAAGATGTGGGCATGAGCTTAC'))
    60.32
    >>> print('%0.2f' % mt.Tm_NN('CGTTCCA1AGATGTGGGCATGAGCTTAC',
    ...                           nn_table=mytable, check=False))
    ... # 'check' must be False, otherwise '1' would be discarded
    62.53

"""
import math
import warnings
from Bio import SeqUtils, Seq
from Bio import BiopythonWarning
DNA_NN1 = {'init': (0, 0), 'init_A/T': (0, 0), 'init_G/C': (0, 0), 'init_oneG/C': (0, -16.8), 'init_allA/T': (0, -20.1), 'init_5T/A': (0, 0), 'sym': (0, -1.3), 'AA/TT': (-9.1, -24.0), 'AT/TA': (-8.6, -23.9), 'TA/AT': (-6.0, -16.9), 'CA/GT': (-5.8, -12.9), 'GT/CA': (-6.5, -17.3), 'CT/GA': (-7.8, -20.8), 'GA/CT': (-5.6, -13.5), 'CG/GC': (-11.9, -27.8), 'GC/CG': (-11.1, -26.7), 'GG/CC': (-11.0, -26.6)}
DNA_NN2 = {'init': (0.6, -9.0), 'init_A/T': (0, 0), 'init_G/C': (0, 0), 'init_oneG/C': (0, 0), 'init_allA/T': (0, 0), 'init_5T/A': (0, 0), 'sym': (0, -1.4), 'AA/TT': (-8.0, -21.9), 'AT/TA': (-5.6, -15.2), 'TA/AT': (-6.6, -18.4), 'CA/GT': (-8.2, -21.0), 'GT/CA': (-9.4, -25.5), 'CT/GA': (-6.6, -16.4), 'GA/CT': (-8.8, -23.5), 'CG/GC': (-11.8, -29.0), 'GC/CG': (-10.5, -26.4), 'GG/CC': (-10.9, -28.4)}
DNA_NN3 = {'init': (0, 0), 'init_A/T': (2.3, 4.1), 'init_G/C': (0.1, -2.8), 'init_oneG/C': (0, 0), 'init_allA/T': (0, 0), 'init_5T/A': (0, 0), 'sym': (0, -1.4), 'AA/TT': (-7.9, -22.2), 'AT/TA': (-7.2, -20.4), 'TA/AT': (-7.2, -21.3), 'CA/GT': (-8.5, -22.7), 'GT/CA': (-8.4, -22.4), 'CT/GA': (-7.8, -21.0), 'GA/CT': (-8.2, -22.2), 'CG/GC': (-10.6, -27.2), 'GC/CG': (-9.8, -24.4), 'GG/CC': (-8.0, -19.9)}
DNA_NN4 = {'init': (0.2, -5.7), 'init_A/T': (2.2, 6.9), 'init_G/C': (0, 0), 'init_oneG/C': (0, 0), 'init_allA/T': (0, 0), 'init_5T/A': (0, 0), 'sym': (0, -1.4), 'AA/TT': (-7.6, -21.3), 'AT/TA': (-7.2, -20.4), 'TA/AT': (-7.2, -21.3), 'CA/GT': (-8.5, -22.7), 'GT/CA': (-8.4, -22.4), 'CT/GA': (-7.8, -21.0), 'GA/CT': (-8.2, -22.2), 'CG/GC': (-10.6, -27.2), 'GC/CG': (-9.8, -24.4), 'GG/CC': (-8.0, -19.9)}
RNA_NN1 = {'init': (0, -10.8), 'init_A/T': (0, 0), 'init_G/C': (0, 0), 'init_oneG/C': (0, 0), 'init_allA/T': (0, 0), 'init_5T/A': (0, 0), 'sym': (0, -1.4), 'AA/TT': (-6.6, -18.4), 'AT/TA': (-5.7, -15.5), 'TA/AT': (-8.1, -22.6), 'CA/GT': (-10.5, -27.8), 'GT/CA': (-10.2, -26.2), 'CT/GA': (-7.6, -19.2), 'GA/CT': (-13.3, -35.5), 'CG/GC': (-8.0, -19.4), 'GC/CG': (-14.2, -34.9), 'GG/CC': (-12.2, -29.7)}
RNA_NN2 = {'init': (3.61, -1.5), 'init_A/T': (3.72, 10.5), 'init_G/C': (0, 0), 'init_oneG/C': (0, 0), 'init_allA/T': (0, 0), 'init_5T/A': (0, 0), 'sym': (0, -1.4), 'AA/TT': (-6.82, -19.0), 'AT/TA': (-9.38, -26.7), 'TA/AT': (-7.69, -20.5), 'CA/GT': (-10.44, -26.9), 'GT/CA': (-11.4, -29.5), 'CT/GA': (-10.48, -27.1), 'GA/CT': (-12.44, -32.5), 'CG/GC': (-10.64, -26.7), 'GC/CG': (-14.88, -36.9), 'GG/CC': (-13.39, -32.7)}
RNA_NN3 = {'init': (6.4, 6.99), 'init_A/T': (3.85, 11.04), 'init_G/C': (0, 0), 'init_oneG/C': (0, 0), 'init_allA/T': (0, 0), 'init_5T/A': (0, 0), 'sym': (0, -1.4), 'AA/TT': (-7.09, -19.8), 'AT/TA': (-9.11, -25.8), 'TA/AT': (-8.5, -22.9), 'CA/GT': (-11.03, -28.8), 'GT/CA': (-11.98, -31.3), 'CT/GA': (-10.9, -28.5), 'GA/CT': (-13.21, -34.9), 'CG/GC': (-10.88, -27.4), 'GC/CG': (-16.04, -40.6), 'GG/CC': (-14.18, -35.0), 'GT/TG': (-13.83, -46.9), 'GG/TT': (-17.82, -56.7), 'AG/TT': (-3.96, -11.6), 'TG/AT': (-0.96, -1.8), 'TT/AG': (-10.38, -31.8), 'TG/GT': (-12.64, -38.9), 'AT/TG': (-7.39, -21.0), 'CG/GT': (-5.56, -13.9), 'CT/GG': (-9.44, -24.7), 'GG/CT': (-7.03, -16.8), 'GT/CG': (-11.09, -28.8)}
R_DNA_NN1 = {'init': (1.9, -3.9), 'init_A/T': (0, 0), 'init_G/C': (0, 0), 'init_oneG/C': (0, 0), 'init_allA/T': (0, 0), 'init_5T/A': (0, 0), 'sym': (0, 0), 'TT/AA': (-11.5, -36.4), 'GT/CA': (-7.8, -21.6), 'CT/GA': (-7.0, -19.7), 'AT/TA': (-8.3, -23.9), 'TG/AC': (-10.4, -28.4), 'GG/CC': (-12.8, -31.9), 'CG/GC': (-16.3, -47.1), 'AG/TC': (-9.1, -23.5), 'TC/AG': (-8.6, -22.9), 'GC/CG': (-8.0, -17.1), 'CC/GG': (-9.3, -23.2), 'AC/TG': (-5.9, -12.3), 'TA/AT': (-7.8, -23.2), 'GA/CT': (-5.5, -13.5), 'CA/GT': (-9.0, -26.1), 'AA/TT': (-7.8, -21.9)}
DNA_IMM1 = {'AG/TT': (1.0, 0.9), 'AT/TG': (-2.5, -8.3), 'CG/GT': (-4.1, -11.7), 'CT/GG': (-2.8, -8.0), 'GG/CT': (3.3, 10.4), 'GG/TT': (5.8, 16.3), 'GT/CG': (-4.4, -12.3), 'GT/TG': (4.1, 9.5), 'TG/AT': (-0.1, -1.7), 'TG/GT': (-1.4, -6.2), 'TT/AG': (-1.3, -5.3), 'AA/TG': (-0.6, -2.3), 'AG/TA': (-0.7, -2.3), 'CA/GG': (-0.7, -2.3), 'CG/GA': (-4.0, -13.2), 'GA/CG': (-0.6, -1.0), 'GG/CA': (0.5, 3.2), 'TA/AG': (0.7, 0.7), 'TG/AA': (3.0, 7.4), 'AC/TT': (0.7, 0.2), 'AT/TC': (-1.2, -6.2), 'CC/GT': (-0.8, -4.5), 'CT/GC': (-1.5, -6.1), 'GC/CT': (2.3, 5.4), 'GT/CC': (5.2, 13.5), 'TC/AT': (1.2, 0.7), 'TT/AC': (1.0, 0.7), 'AA/TC': (2.3, 4.6), 'AC/TA': (5.3, 14.6), 'CA/GC': (1.9, 3.7), 'CC/GA': (0.6, -0.6), 'GA/CC': (5.2, 14.2), 'GC/CA': (-0.7, -3.8), 'TA/AC': (3.4, 8.0), 'TC/AA': (7.6, 20.2), 'AA/TA': (1.2, 1.7), 'CA/GA': (-0.9, -4.2), 'GA/CA': (-2.9, -9.8), 'TA/AA': (4.7, 12.9), 'AC/TC': (0.0, -4.4), 'CC/GC': (-1.5, -7.2), 'GC/CC': (3.6, 8.9), 'TC/AC': (6.1, 16.4), 'AG/TG': (-3.1, -9.5), 'CG/GG': (-4.9, -15.3), 'GG/CG': (-6.0, -15.8), 'TG/AG': (1.6, 3.6), 'AT/TT': (-2.7, -10.8), 'CT/GT': (-5.0, -15.8), 'GT/CT': (-2.2, -8.4), 'TT/AT': (0.2, -1.5), 'AI/TC': (-8.9, -25.5), 'TI/AC': (-5.9, -17.4), 'AC/TI': (-8.8, -25.4), 'TC/AI': (-4.9, -13.9), 'CI/GC': (-5.4, -13.7), 'GI/CC': (-6.8, -19.1), 'CC/GI': (-8.3, -23.8), 'GC/CI': (-5.0, -12.6), 'AI/TA': (-8.3, -25.0), 'TI/AA': (-3.4, -11.2), 'AA/TI': (-0.7, -2.6), 'TA/AI': (-1.3, -4.6), 'CI/GA': (2.6, 8.9), 'GI/CA': (-7.8, -21.1), 'CA/GI': (-7.0, -20.0), 'GA/CI': (-7.6, -20.2), 'AI/TT': (0.49, -0.7), 'TI/AT': (-6.5, -22.0), 'AT/TI': (-5.6, -18.7), 'TT/AI': (-0.8, -4.3), 'CI/GT': (-1.0, -2.4), 'GI/CT': (-3.5, -10.6), 'CT/GI': (0.1, -1.0), 'GT/CI': (-4.3, -12.1), 'AI/TG': (-4.9, -15.8), 'TI/AG': (-1.9, -8.5), 'AG/TI': (0.1, -1.8), 'TG/AI': (1.0, 1.0), 'CI/GG': (7.1, 21.3), 'GI/CG': (-1.1, -3.2), 'CG/GI': (5.8, 16.9), 'GG/CI': (-7.6, -22.0), 'AI/TI': (-3.3, -11.9), 'TI/AI': (0.1, -2.3), 'CI/GI': (1.3, 3.0), 'GI/CI': (-0.5, -1.3)}
DNA_TMM1 = {'AA/TA': (-3.1, -7.8), 'TA/AA': (-2.5, -6.3), 'CA/GA': (-4.3, -10.7), 'GA/CA': (-8.0, -22.5), 'AC/TC': (-0.1, 0.5), 'TC/AC': (-0.7, -1.3), 'CC/GC': (-2.1, -5.1), 'GC/CC': (-3.9, -10.6), 'AG/TG': (-1.1, -2.1), 'TG/AG': (-1.1, -2.7), 'CG/GG': (-3.8, -9.5), 'GG/CG': (-0.7, -19.2), 'AT/TT': (-2.4, -6.5), 'TT/AT': (-3.2, -8.9), 'CT/GT': (-6.1, -16.9), 'GT/CT': (-7.4, -21.2), 'AA/TC': (-1.6, -4.0), 'AC/TA': (-1.8, -3.8), 'CA/GC': (-2.6, -5.9), 'CC/GA': (-2.7, -6.0), 'GA/CC': (-5.0, -13.8), 'GC/CA': (-3.2, -7.1), 'TA/AC': (-2.3, -5.9), 'TC/AA': (-2.7, -7.0), 'AC/TT': (-0.9, -1.7), 'AT/TC': (-2.3, -6.3), 'CC/GT': (-3.2, -8.0), 'CT/GC': (-3.9, -10.6), 'GC/CT': (-4.9, -13.5), 'GT/CC': (-3.0, -7.8), 'TC/AT': (-2.5, -6.3), 'TT/AC': (-0.7, -1.2), 'AA/TG': (-1.9, -4.4), 'AG/TA': (-2.5, -5.9), 'CA/GG': (-3.9, -9.6), 'CG/GA': (-6.0, -15.5), 'GA/CG': (-4.3, -11.1), 'GG/CA': (-4.6, -11.4), 'TA/AG': (-2.0, -4.7), 'TG/AA': (-2.4, -5.8), 'AG/TT': (-3.2, -8.7), 'AT/TG': (-3.5, -9.4), 'CG/GT': (-3.8, -9.0), 'CT/GG': (-6.6, -18.7), 'GG/CT': (-5.7, -15.9), 'GT/CG': (-5.9, -16.1), 'TG/AT': (-3.9, -10.5), 'TT/AG': (-3.6, -9.8)}
DNA_DE1 = {'AA/.T': (0.2, 2.3), 'AC/.G': (-6.3, -17.1), 'AG/.C': (-3.7, -10.0), 'AT/.A': (-2.9, -7.6), 'CA/.T': (0.6, 3.3), 'CC/.G': (-4.4, -12.6), 'CG/.C': (-4.0, -11.9), 'CT/.A': (-4.1, -13.0), 'GA/.T': (-1.1, -1.6), 'GC/.G': (-5.1, -14.0), 'GG/.C': (-3.9, -10.9), 'GT/.A': (-4.2, -15.0), 'TA/.T': (-6.9, -20.0), 'TC/.G': (-4.0, -10.9), 'TG/.C': (-4.9, -13.8), 'TT/.A': (-0.2, -0.5), '.A/AT': (-0.7, -0.8), '.C/AG': (-2.1, -3.9), '.G/AC': (-5.9, -16.5), '.T/AA': (-0.5, -1.1), '.A/CT': (4.4, 14.9), '.C/CG': (-0.2, -0.1), '.G/CC': (-2.6, -7.4), '.T/CA': (4.7, 14.2), '.A/GT': (-1.6, -3.6), '.C/GG': (-3.9, -11.2), '.G/GC': (-3.2, -10.4), '.T/GA': (-4.1, -13.1), '.A/TT': (2.9, 10.4), '.C/TG': (-4.4, -13.1), '.G/TC': (-5.2, -15.0), '.T/TA': (-3.8, -12.6)}
RNA_DE1 = {'.T/AA': (-4.9, -13.2), '.T/CA': (-0.9, -1.3), '.T/GA': (-5.5, -15.1), '.T/TA': (-2.3, -5.5), '.G/AC': (-9.0, -23.5), '.G/CC': (-4.1, -10.6), '.G/GC': (-8.6, -22.2), '.G/TC': (-7.5, -20.31), '.C/AG': (-7.4, -20.3), '.C/CG': (-2.8, -7.7), '.C/GG': (-6.4, -16.4), '.C/TG': (-3.6, -9.7), '.T/AG': (-4.9, -13.2), '.T/CG': (-0.9, -1.3), '.T/GG': (-5.5, -15.1), '.T/TG': (-2.3, -5.5), '.A/AT': (-5.7, -16.1), '.A/CT': (-0.7, -1.9), '.A/GT': (-5.8, -16.4), '.A/TT': (-2.2, -6.8), '.G/AT': (-5.7, -16.1), '.G/CT': (-0.7, -1.9), '.G/GT': (-5.8, -16.4), '.G/TT': (-2.2, -6.8), 'AT/.A': (-0.5, -0.6), 'CT/.A': (6.9, 22.6), 'GT/.A': (0.6, 2.6), 'TT/.A': (0.6, 2.6), 'AG/.C': (-1.6, -4.5), 'CG/.C': (0.7, 3.2), 'GG/.C': (-4.6, -14.8), 'TG/.C': (-0.4, -1.3), 'AC/.G': (-2.4, -6.1), 'CC/.G': (3.3, 11.6), 'GC/.G': (0.8, 3.2), 'TC/.G': (-1.4, -4.2), 'AT/.G': (-0.5, -0.6), 'CT/.G': (6.9, 22.6), 'GT/.G': (0.6, 2.6), 'TT/.G': (0.6, 2.6), 'AA/.T': (1.6, 6.1), 'CA/.T': (2.2, 8.1), 'GA/.T': (0.7, 3.5), 'TA/.T': (3.1, 10.6), 'AG/.T': (1.6, 6.1), 'CG/.T': (2.2, 8.1), 'GG/.T': (0.7, 3.5), 'TG/.T': (3.1, 10.6)}

def make_table(oldtable=None, values=None):
    if False:
        for i in range(10):
            print('nop')
    'Return a table with thermodynamic parameters (as dictionary).\n\n    Arguments:\n     - oldtable: An existing dictionary with thermodynamic parameters.\n     - values: A dictionary with new or updated values.\n\n    E.g., to replace the initiation parameters in the Sugimoto \'96 dataset with\n    the initiation parameters from Allawi & SantaLucia \'97:\n\n    >>> from Bio.SeqUtils.MeltingTemp import make_table, DNA_NN2\n    >>> table = DNA_NN2                               # Sugimoto \'96\n    >>> table[\'init_A/T\']\n    (0, 0)\n    >>> newtable = make_table(oldtable=DNA_NN2, values={\'init\': (0, 0),\n    ...                       \'init_A/T\': (2.3, 4.1),\n    ...                       \'init_G/C\': (0.1, -2.8)})\n    >>> print("%0.1f, %0.1f" % newtable[\'init_A/T\'])\n    2.3, 4.1\n\n    '
    if oldtable is None:
        table = {'init': (0, 0), 'init_A/T': (0, 0), 'init_G/C': (0, 0), 'init_oneG/C': (0, 0), 'init_allA/T': (0, 0), 'init_5T/A': (0, 0), 'sym': (0, 0), 'AA/TT': (0, 0), 'AT/TA': (0, 0), 'TA/AT': (0, 0), 'CA/GT': (0, 0), 'GT/CA': (0, 0), 'CT/GA': (0, 0), 'GA/CT': (0, 0), 'CG/GC': (0, 0), 'GC/CG': (0, 0), 'GG/CC': (0, 0)}
    else:
        table = oldtable.copy()
    if values:
        table.update(values)
    return table

def _check(seq, method):
    if False:
        while True:
            i = 10
    "Return a sequence which fulfills the requirements of the given method (PRIVATE).\n\n    All Tm methods in this package require the sequence in uppercase format.\n    Most methods make use of the length of the sequence (directly or\n    indirectly), which can only be expressed as len(seq) if the sequence does\n    not contain whitespaces and other non-base characters. RNA sequences are\n    backtranscribed to DNA. This method is PRIVATE.\n\n    Arguments:\n     - seq: The sequence as given by the user (passed as string).\n     - method: Tm_Wallace, Tm_GC or Tm_NN.\n\n    >>> from Bio.SeqUtils import MeltingTemp as mt\n    >>> mt._check('10 ACGTTGCAAG tccatggtac', 'Tm_NN')\n    'ACGTTGCAAGTCCATGGTAC'\n\n    "
    seq = ''.join(seq.split()).upper()
    seq = str(Seq.Seq(seq).back_transcribe())
    if method == 'Tm_Wallace':
        return seq
    if method == 'Tm_GC':
        baseset = ('A', 'B', 'C', 'D', 'G', 'H', 'I', 'K', 'M', 'N', 'R', 'S', 'T', 'V', 'W', 'X', 'Y')
    if method == 'Tm_NN':
        baseset = ('A', 'C', 'G', 'T', 'I')
    seq = ''.join([base for base in seq if base in baseset])
    return seq

def salt_correction(Na=0, K=0, Tris=0, Mg=0, dNTPs=0, method=1, seq=None):
    if False:
        i = 10
        return i + 15
    "Calculate a term to correct Tm for salt ions.\n\n    Depending on the Tm calculation, the term will correct Tm or entropy. To\n    calculate corrected Tm values, different operations need to be applied:\n\n     - methods 1-4: Tm(new) = Tm(old) + corr\n     - method 5: deltaS(new) = deltaS(old) + corr\n     - methods 6+7: Tm(new) = 1/(1/Tm(old) + corr)\n\n    Arguments:\n     - Na, K, Tris, Mg, dNTPS: Millimolar concentration of respective ion. To\n       have a simple 'salt correction', just pass Na. If any of K, Tris, Mg and\n       dNTPS is non-zero, a 'sodium-equivalent' concentration is calculated\n       according to von Ahsen et al. (2001, Clin Chem 47: 1956-1961):\n       [Na_eq] = [Na+] + [K+] + [Tris]/2 + 120*([Mg2+] - [dNTPs])^0.5\n       If [dNTPs] >= [Mg2+]: [Na_eq] = [Na+] + [K+] + [Tris]/2\n     - method: Which method to be applied. Methods 1-4 correct Tm, method 5\n       corrects deltaS, methods 6 and 7 correct 1/Tm. The methods are:\n\n       1. 16.6 x log[Na+]\n          (Schildkraut & Lifson (1965), Biopolymers 3: 195-208)\n       2. 16.6 x log([Na+]/(1.0 + 0.7*[Na+]))\n          (Wetmur (1991), Crit Rev Biochem Mol Biol 126: 227-259)\n       3. 12.5 x log(Na+]\n          (SantaLucia et al. (1996), Biochemistry 35: 3555-3562\n       4. 11.7 x log[Na+]\n          (SantaLucia (1998), Proc Natl Acad Sci USA 95: 1460-1465\n       5. Correction for deltaS: 0.368 x (N-1) x ln[Na+]\n          (SantaLucia (1998), Proc Natl Acad Sci USA 95: 1460-1465)\n       6. (4.29(%GC)-3.95)x1e-5 x ln[Na+] + 9.40e-6 x ln[Na+]^2\n          (Owczarzy et al. (2004), Biochemistry 43: 3537-3554)\n       7. Complex formula with decision tree and 7 empirical constants.\n          Mg2+ is corrected for dNTPs binding (if present)\n          (Owczarzy et al. (2008), Biochemistry 47: 5336-5353)\n\n    Examples\n    --------\n    >>> from Bio.SeqUtils.MeltingTemp import salt_correction\n    >>> print('%0.2f' % salt_correction(Na=50, method=1))\n    -21.60\n    >>> print('%0.2f' % salt_correction(Na=50, method=2))\n    -21.85\n    >>> print('%0.2f' % salt_correction(Na=100, Tris=20, method=2))\n    -16.45\n    >>> print('%0.2f' % salt_correction(Na=100, Tris=20, Mg=1.5, method=2))\n    -10.99\n\n    "
    if method in (5, 6, 7) and (not seq):
        raise ValueError('sequence is missing (is needed to calculate GC content or sequence length).')
    corr = 0
    if not method:
        return corr
    Mon = Na + K + Tris / 2.0
    mg = Mg * 0.001
    if sum((K, Mg, Tris, dNTPs)) > 0 and method != 7 and (dNTPs < Mg):
        Mon += 120 * math.sqrt(Mg - dNTPs)
    mon = Mon * 0.001
    if method in range(1, 7) and (not mon):
        raise ValueError('Total ion concentration of zero is not allowed in this method.')
    if method == 1:
        corr = 16.6 * math.log10(mon)
    if method == 2:
        corr = 16.6 * math.log10(mon / (1.0 + 0.7 * mon))
    if method == 3:
        corr = 12.5 * math.log10(mon)
    if method == 4:
        corr = 11.7 * math.log10(mon)
    if method == 5:
        corr = 0.368 * (len(seq) - 1) * math.log(mon)
    if method == 6:
        corr = (4.29 * SeqUtils.gc_fraction(seq, 'ignore') - 3.95) * 1e-05 * math.log(mon) + 9.4e-06 * math.log(mon) ** 2
    if method == 7:
        (a, b, c, d) = (3.92, -0.911, 6.26, 1.42)
        (e, f, g) = (-48.2, 52.5, 8.31)
        if dNTPs > 0:
            dntps = dNTPs * 0.001
            ka = 30000.0
            mg = (-(ka * dntps - ka * mg + 1.0) + math.sqrt((ka * dntps - ka * mg + 1.0) ** 2 + 4.0 * ka * mg)) / (2.0 * ka)
        if Mon > 0:
            R = math.sqrt(mg) / mon
            if R < 0.22:
                corr = (4.29 * SeqUtils.gc_fraction(seq, 'ignore') - 3.95) * 1e-05 * math.log(mon) + 9.4e-06 * math.log(mon) ** 2
                return corr
            elif R < 6.0:
                a = 3.92 * (0.843 - 0.352 * math.sqrt(mon) * math.log(mon))
                d = 1.42 * (1.279 - 0.00403 * math.log(mon) - 0.00803 * math.log(mon) ** 2)
                g = 8.31 * (0.486 - 0.258 * math.log(mon) + 0.00525 * math.log(mon) ** 3)
        corr = (a + b * math.log(mg) + SeqUtils.gc_fraction(seq, 'ignore') * (c + d * math.log(mg)) + 1 / (2.0 * (len(seq) - 1)) * (e + f * math.log(mg) + g * math.log(mg) ** 2)) * 1e-05
    if method > 7:
        raise ValueError("Allowed values for parameter 'method' are 1-7.")
    return corr

def chem_correction(melting_temp, DMSO=0, fmd=0, DMSOfactor=0.75, fmdfactor=0.65, fmdmethod=1, GC=None):
    if False:
        print('Hello World!')
    "Correct a given Tm for DMSO and formamide.\n\n    Please note that these corrections are +/- rough approximations.\n\n    Arguments:\n     - melting_temp: Melting temperature.\n     - DMSO: Percent DMSO.\n     - fmd: Formamide concentration in %(fmdmethod=1) or molar (fmdmethod=2).\n     - DMSOfactor: How much should Tm decreases per percent DMSO. Default=0.65\n       (von Ahsen et al. 2001). Other published values are 0.5, 0.6 and 0.675.\n     - fmdfactor: How much should Tm decrease per percent formamide.\n       Default=0.65. Several papers report factors between 0.6 and 0.72.\n     - fmdmethod:\n\n         1. Tm = Tm - factor(%formamide) (Default)\n         2. Tm = Tm + (0.453(f(GC)) - 2.88) x [formamide]\n\n       Here f(GC) is fraction of GC.\n       Note (again) that in fmdmethod=1 formamide concentration is given in %,\n       while in fmdmethod=2 it is given in molar.\n     - GC: GC content in percent.\n\n    Examples:\n        >>> from Bio.SeqUtils import MeltingTemp as mt\n        >>> mt.chem_correction(70)\n        70\n        >>> print('%0.2f' % mt.chem_correction(70, DMSO=3))\n        67.75\n        >>> print('%0.2f' % mt.chem_correction(70, fmd=5))\n        66.75\n        >>> print('%0.2f' % mt.chem_correction(70, fmdmethod=2, fmd=1.25,\n        ...                                    GC=50))\n        66.68\n\n    "
    if DMSO:
        melting_temp -= DMSOfactor * DMSO
    if fmd:
        if fmdmethod == 1:
            melting_temp -= fmdfactor * fmd
        if fmdmethod == 2:
            if GC is None or GC < 0:
                raise ValueError("'GC' is missing or negative")
            melting_temp += (0.453 * (GC / 100.0) - 2.88) * fmd
        if fmdmethod not in (1, 2):
            raise ValueError("'fmdmethod' must be 1 or 2")
    return melting_temp

def Tm_Wallace(seq, check=True, strict=True):
    if False:
        while True:
            i = 10
    "Calculate and return the Tm using the 'Wallace rule'.\n\n    Tm = 4 degC * (G + C) + 2 degC * (A+T)\n\n    The Wallace rule (Thein & Wallace 1986, in Human genetic diseases: a\n    practical approach, 33-50) is often used as rule of thumb for approximate\n    Tm calculations for primers of 14 to 20 nt length.\n\n    Non-DNA characters (e.g., E, F, J, !, 1, etc) are ignored by this method.\n\n    Examples:\n        >>> from Bio.SeqUtils import MeltingTemp as mt\n        >>> mt.Tm_Wallace('ACGTTGCAATGCCGTA')\n        48.0\n        >>> mt.Tm_Wallace('ACGT TGCA ATGC CGTA')\n        48.0\n        >>> mt.Tm_Wallace('1ACGT2TGCA3ATGC4CGTA')\n        48.0\n\n    "
    seq = str(seq)
    if check:
        seq = _check(seq, 'Tm_Wallace')
    melting_temp = 2 * sum(map(seq.count, ('A', 'T', 'W'))) + 4 * sum(map(seq.count, ('C', 'G', 'S')))
    tmp = 3 * sum(map(seq.count, ('K', 'M', 'N', 'R', 'Y'))) + 10 / 3.0 * sum(map(seq.count, ('B', 'V'))) + 8 / 3.0 * sum(map(seq.count, ('D', 'H')))
    if strict and tmp:
        raise ValueError('ambiguous bases B, D, H, K, M, N, R, V, Y not allowed when strict=True')
    else:
        melting_temp += tmp
    return melting_temp

def Tm_GC(seq, check=True, strict=True, valueset=7, userset=None, Na=50, K=0, Tris=0, Mg=0, dNTPs=0, saltcorr=0, mismatch=True):
    if False:
        print('Hello World!')
    'Return the Tm using empirical formulas based on GC content.\n\n    General format: Tm = A + B(%GC) - C/N + salt correction - D(%mismatch)\n\n    A, B, C, D: empirical constants, N: primer length\n    D (amount of decrease in Tm per % mismatch) is often 1, but sometimes other\n    values have been used (0.6-1.5). Use \'X\' to indicate the mismatch position\n    in the sequence. Note that this mismatch correction is a rough estimate.\n\n    >>> from Bio.SeqUtils import MeltingTemp as mt\n    >>> print("%0.2f" % mt.Tm_GC(\'CTGCTGATXGCACGAGGTTATGG\', valueset=2))\n    69.20\n\n    Arguments:\n     - valueset: A few often cited variants are included:\n\n        1. Tm = 69.3 + 0.41(%GC) - 650/N\n           (Marmur & Doty 1962, J Mol Biol 5: 109-118; Chester & Marshak 1993),\n           Anal Biochem 209: 284-290)\n        2. Tm = 81.5 + 0.41(%GC) - 675/N - %mismatch\n           \'QuikChange\' formula. Recommended (by the manufacturer) for the\n           design of primers for QuikChange mutagenesis.\n        3. Tm = 81.5 + 0.41(%GC) - 675/N + 16.6 x log[Na+]\n           (Marmur & Doty 1962, J Mol Biol 5: 109-118; Schildkraut & Lifson\n           1965, Biopolymers 3: 195-208)\n        4. Tm = 81.5 + 0.41(%GC) - 500/N + 16.6 x log([Na+]/(1.0 + 0.7 x\n           [Na+])) - %mismatch\n           (Wetmur 1991, Crit Rev Biochem Mol Biol 126: 227-259). This is the\n           standard formula in approximative mode of MELTING 4.3.\n        5. Tm = 78 + 0.7(%GC) - 500/N + 16.6 x log([Na+]/(1.0 + 0.7 x [Na+]))\n           - %mismatch\n           (Wetmur 1991, Crit Rev Biochem Mol Biol 126: 227-259). For RNA.\n        6. Tm = 67 + 0.8(%GC) - 500/N + 16.6 x log([Na+]/(1.0 + 0.7 x [Na+]))\n           - %mismatch\n           (Wetmur 1991, Crit Rev Biochem Mol Biol 126: 227-259). For RNA/DNA\n           hybrids.\n        7. Tm = 81.5 + 0.41(%GC) - 600/N + 16.6 x log[Na+]\n           Used by Primer3Plus to calculate the product Tm. Default set.\n        8. Tm = 77.1 + 0.41(%GC) - 528/N + 11.7 x log[Na+]\n           (von Ahsen et al. 2001, Clin Chem 47: 1956-1961). Recommended \'as a\n           tradeoff between accuracy and ease of use\'.\n\n     - userset: Tuple of four values for A, B, C, and D. Usersets override\n       valuesets.\n     - Na, K, Tris, Mg, dNTPs: Concentration of the respective ions [mM]. If\n       any of K, Tris, Mg and dNTPS is non-zero, a \'sodium-equivalent\'\n       concentration is calculated and used for salt correction (von Ahsen et\n       al., 2001).\n     - saltcorr: Type of salt correction (see method salt_correction).\n       Default=0. 0 or None means no salt correction.\n     - mismatch: If \'True\' (default) every \'X\' in the sequence is counted as\n       mismatch.\n\n    '
    if saltcorr == 5:
        raise ValueError('salt-correction method 5 not applicable to Tm_GC')
    seq = str(seq)
    if check:
        seq = _check(seq, 'Tm_GC')
    if strict and any((x in seq for x in 'KMNRYBVDH')):
        raise ValueError("ambiguous bases B, D, H, K, M, N, R, V, Y not allowed when 'strict=True'")
    percent_gc = SeqUtils.gc_fraction(seq, 'weighted') * 100
    if mismatch:
        percent_gc -= seq.count('X') * 50.0 / len(seq)
    if userset:
        (A, B, C, D) = userset
    else:
        if valueset == 1:
            (A, B, C, D) = (69.3, 0.41, 650, 1)
            saltcorr = 0
        if valueset == 2:
            (A, B, C, D) = (81.5, 0.41, 675, 1)
            saltcorr = 0
        if valueset == 3:
            (A, B, C, D) = (81.5, 0.41, 675, 1)
            saltcorr = 1
        if valueset == 4:
            (A, B, C, D) = (81.5, 0.41, 500, 1)
            saltcorr = 2
        if valueset == 5:
            (A, B, C, D) = (78.0, 0.7, 500, 1)
            saltcorr = 2
        if valueset == 6:
            (A, B, C, D) = (67.0, 0.8, 500, 1)
            saltcorr = 2
        if valueset == 7:
            (A, B, C, D) = (81.5, 0.41, 600, 1)
            saltcorr = 1
        if valueset == 8:
            (A, B, C, D) = (77.1, 0.41, 528, 1)
            saltcorr = 4
    if valueset > 8:
        raise ValueError("allowed values for parameter 'valueset' are 0-8.")
    melting_temp = A + B * percent_gc - C / len(seq)
    if saltcorr:
        melting_temp += salt_correction(Na=Na, K=K, Tris=Tris, Mg=Mg, dNTPs=dNTPs, seq=seq, method=saltcorr)
    if mismatch:
        melting_temp -= D * (seq.count('X') * 100.0 / len(seq))
    return melting_temp

def _key_error(neighbors, strict):
    if False:
        while True:
            i = 10
    'Throw an error or a warning if there is no data for the neighbors (PRIVATE).'
    if strict:
        raise ValueError(f'no thermodynamic data for neighbors {neighbors!r} available')
    else:
        warnings.warn('no themodynamic data for neighbors %r available. Calculation will be wrong' % neighbors, BiopythonWarning)

def Tm_NN(seq, check=True, strict=True, c_seq=None, shift=0, nn_table=None, tmm_table=None, imm_table=None, de_table=None, dnac1=25, dnac2=25, selfcomp=False, Na=50, K=0, Tris=0, Mg=0, dNTPs=0, saltcorr=5):
    if False:
        while True:
            i = 10
    "Return the Tm using nearest neighbor thermodynamics.\n\n    Arguments:\n     - seq: The primer/probe sequence as string or Biopython sequence object.\n       For RNA/DNA hybridizations seq must be the RNA sequence.\n     - c_seq: Complementary sequence. The sequence of the template/target in\n       3'->5' direction. c_seq is necessary for mismatch correction and\n       dangling-ends correction. Both corrections will automatically be\n       applied if mismatches or dangling ends are present. Default=None.\n     - shift: Shift of the primer/probe sequence on the template/target\n       sequence, e.g.::\n\n                           shift=0       shift=1        shift= -1\n        Primer (seq):      5' ATGC...    5'  ATGC...    5' ATGC...\n        Template (c_seq):  3' TACG...    3' CTACG...    3'  ACG...\n\n       The shift parameter is necessary to align seq and c_seq if they have\n       different lengths or if they should have dangling ends. Default=0\n     - table: Thermodynamic NN values, eight tables are implemented:\n       For DNA/DNA hybridizations:\n\n        - DNA_NN1: values from Breslauer et al. (1986)\n        - DNA_NN2: values from Sugimoto et al. (1996)\n        - DNA_NN3: values from Allawi & SantaLucia (1997) (default)\n        - DNA_NN4: values from SantaLucia & Hicks (2004)\n\n       For RNA/RNA hybridizations:\n\n        - RNA_NN1: values from Freier et al. (1986)\n        - RNA_NN2: values from Xia et al. (1998)\n        - RNA_NN3: values from Chen et al. (2012)\n\n       For RNA/DNA hybridizations:\n\n        - R_DNA_NN1: values from Sugimoto et al. (1995)\n          Note that ``seq`` must be the RNA sequence.\n\n       Use the module's maketable method to make a new table or to update one\n       one of the implemented tables.\n     - tmm_table: Thermodynamic values for terminal mismatches.\n       Default: DNA_TMM1 (SantaLucia & Peyret, 2001)\n     - imm_table: Thermodynamic values for internal mismatches, may include\n       insosine mismatches. Default: DNA_IMM1 (Allawi & SantaLucia, 1997-1998;\n       Peyret et al., 1999; Watkins & SantaLucia, 2005)\n     - de_table: Thermodynamic values for dangling ends:\n\n        - DNA_DE1: for DNA. Values from Bommarito et al. (2000) (default)\n        - RNA_DE1: for RNA. Values from Turner & Mathews (2010)\n\n     - dnac1: Concentration of the higher concentrated strand [nM]. Typically\n       this will be the primer (for PCR) or the probe. Default=25.\n     - dnac2: Concentration of the lower concentrated strand [nM]. In PCR this\n       is the template strand which concentration is typically very low and may\n       be ignored (dnac2=0). In oligo/oligo hybridization experiments, dnac1\n       equals dnac1. Default=25.\n       MELTING and Primer3Plus use k = [Oligo(Total)]/4 by default. To mimic\n       this behaviour, you have to divide [Oligo(Total)] by 2 and assign this\n       concentration to dnac1 and dnac2. E.g., Total oligo concentration of\n       50 nM in Primer3Plus means dnac1=25, dnac2=25.\n     - selfcomp: Is the sequence self-complementary? Default=False. If 'True'\n       the primer is thought binding to itself, thus dnac2 is not considered.\n     - Na, K, Tris, Mg, dNTPs: See method 'Tm_GC' for details. Defaults: Na=50,\n       K=0, Tris=0, Mg=0, dNTPs=0.\n     - saltcorr: See method 'Tm_GC'. Default=5. 0 means no salt correction.\n\n    "
    if not nn_table:
        nn_table = DNA_NN3
    if not tmm_table:
        tmm_table = DNA_TMM1
    if not imm_table:
        imm_table = DNA_IMM1
    if not de_table:
        de_table = DNA_DE1
    seq = str(seq)
    if not c_seq:
        c_seq = Seq.Seq(seq).complement()
    c_seq = str(c_seq)
    if check:
        seq = _check(seq, 'Tm_NN')
        c_seq = _check(c_seq, 'Tm_NN')
    tmp_seq = seq
    tmp_cseq = c_seq
    delta_h = 0
    delta_s = 0
    d_h = 0
    d_s = 1
    if shift or len(seq) != len(c_seq):
        if shift > 0:
            tmp_seq = '.' * shift + seq
        if shift < 0:
            tmp_cseq = '.' * abs(shift) + c_seq
        if len(tmp_cseq) > len(tmp_seq):
            tmp_seq += (len(tmp_cseq) - len(tmp_seq)) * '.'
        if len(tmp_cseq) < len(tmp_seq):
            tmp_cseq += (len(tmp_seq) - len(tmp_cseq)) * '.'
        while tmp_seq.startswith('..') or tmp_cseq.startswith('..'):
            tmp_seq = tmp_seq[1:]
            tmp_cseq = tmp_cseq[1:]
        while tmp_seq.endswith('..') or tmp_cseq.endswith('..'):
            tmp_seq = tmp_seq[:-1]
            tmp_cseq = tmp_cseq[:-1]
        if tmp_seq.startswith('.') or tmp_cseq.startswith('.'):
            left_de = tmp_seq[:2] + '/' + tmp_cseq[:2]
            try:
                delta_h += de_table[left_de][d_h]
                delta_s += de_table[left_de][d_s]
            except KeyError:
                _key_error(left_de, strict)
            tmp_seq = tmp_seq[1:]
            tmp_cseq = tmp_cseq[1:]
        if tmp_seq.endswith('.') or tmp_cseq.endswith('.'):
            right_de = tmp_cseq[-2:][::-1] + '/' + tmp_seq[-2:][::-1]
            try:
                delta_h += de_table[right_de][d_h]
                delta_s += de_table[right_de][d_s]
            except KeyError:
                _key_error(right_de, strict)
            tmp_seq = tmp_seq[:-1]
            tmp_cseq = tmp_cseq[:-1]
    left_tmm = tmp_cseq[:2][::-1] + '/' + tmp_seq[:2][::-1]
    if left_tmm in tmm_table:
        delta_h += tmm_table[left_tmm][d_h]
        delta_s += tmm_table[left_tmm][d_s]
        tmp_seq = tmp_seq[1:]
        tmp_cseq = tmp_cseq[1:]
    right_tmm = tmp_seq[-2:] + '/' + tmp_cseq[-2:]
    if right_tmm in tmm_table:
        delta_h += tmm_table[right_tmm][d_h]
        delta_s += tmm_table[right_tmm][d_s]
        tmp_seq = tmp_seq[:-1]
        tmp_cseq = tmp_cseq[:-1]
    delta_h += nn_table['init'][d_h]
    delta_s += nn_table['init'][d_s]
    if SeqUtils.gc_fraction(seq, 'ignore') == 0:
        delta_h += nn_table['init_allA/T'][d_h]
        delta_s += nn_table['init_allA/T'][d_s]
    else:
        delta_h += nn_table['init_oneG/C'][d_h]
        delta_s += nn_table['init_oneG/C'][d_s]
    if seq.startswith('T'):
        delta_h += nn_table['init_5T/A'][d_h]
        delta_s += nn_table['init_5T/A'][d_s]
    if seq.endswith('A'):
        delta_h += nn_table['init_5T/A'][d_h]
        delta_s += nn_table['init_5T/A'][d_s]
    ends = seq[0] + seq[-1]
    AT = ends.count('A') + ends.count('T')
    GC = ends.count('G') + ends.count('C')
    delta_h += nn_table['init_A/T'][d_h] * AT
    delta_s += nn_table['init_A/T'][d_s] * AT
    delta_h += nn_table['init_G/C'][d_h] * GC
    delta_s += nn_table['init_G/C'][d_s] * GC
    for basenumber in range(len(tmp_seq) - 1):
        neighbors = tmp_seq[basenumber:basenumber + 2] + '/' + tmp_cseq[basenumber:basenumber + 2]
        if neighbors in imm_table:
            delta_h += imm_table[neighbors][d_h]
            delta_s += imm_table[neighbors][d_s]
        elif neighbors[::-1] in imm_table:
            delta_h += imm_table[neighbors[::-1]][d_h]
            delta_s += imm_table[neighbors[::-1]][d_s]
        elif neighbors in nn_table:
            delta_h += nn_table[neighbors][d_h]
            delta_s += nn_table[neighbors][d_s]
        elif neighbors[::-1] in nn_table:
            delta_h += nn_table[neighbors[::-1]][d_h]
            delta_s += nn_table[neighbors[::-1]][d_s]
        else:
            _key_error(neighbors, strict)
    k = (dnac1 - dnac2 / 2.0) * 1e-09
    if selfcomp:
        k = dnac1 * 1e-09
        delta_h += nn_table['sym'][d_h]
        delta_s += nn_table['sym'][d_s]
    R = 1.987
    if saltcorr:
        corr = salt_correction(Na=Na, K=K, Tris=Tris, Mg=Mg, dNTPs=dNTPs, method=saltcorr, seq=seq)
    if saltcorr == 5:
        delta_s += corr
    melting_temp = 1000 * delta_h / (delta_s + R * math.log(k)) - 273.15
    if saltcorr in (1, 2, 3, 4):
        melting_temp += corr
    if saltcorr in (6, 7):
        melting_temp = 1 / (1 / (melting_temp + 273.15) + corr) - 273.15
    return melting_temp
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()