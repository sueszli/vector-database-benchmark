"""Bio.SeqIO support for the FASTQ and QUAL file formats.

Note that you are expected to use this code via the Bio.SeqIO interface, as
shown below.

The FASTQ file format is used frequently at the Wellcome Trust Sanger Institute
to bundle a FASTA sequence and its PHRED quality data (integers between 0 and
90).  Rather than using a single FASTQ file, often paired FASTA and QUAL files
are used containing the sequence and the quality information separately.

The PHRED software reads DNA sequencing trace files, calls bases, and
assigns a non-negative quality value to each called base using a logged
transformation of the error probability, Q = -10 log10( Pe ), for example::

    Pe = 1.0,         Q =  0
    Pe = 0.1,         Q = 10
    Pe = 0.01,        Q = 20
    ...
    Pe = 0.00000001,  Q = 80
    Pe = 0.000000001, Q = 90

In typical raw sequence reads, the PHRED quality valuea will be from 0 to 40.
In the QUAL format these quality values are held as space separated text in
a FASTA like file format.  In the FASTQ format, each quality values is encoded
with a single ASCI character using chr(Q+33), meaning zero maps to the
character "!" and for example 80 maps to "q".  For the Sanger FASTQ standard
the allowed range of PHRED scores is 0 to 93 inclusive. The sequences and
quality are then stored in pairs in a FASTA like format.

Unfortunately there is no official document describing the FASTQ file format,
and worse, several related but different variants exist. For more details,
please read this open access publication::

    The Sanger FASTQ file format for sequences with quality scores, and the
    Solexa/Illumina FASTQ variants.
    P.J.A.Cock (Biopython), C.J.Fields (BioPerl), N.Goto (BioRuby),
    M.L.Heuer (BioJava) and P.M. Rice (EMBOSS).
    Nucleic Acids Research 2010 38(6):1767-1771
    https://doi.org/10.1093/nar/gkp1137

The good news is that Roche 454 sequencers can output files in the QUAL format,
and sensibly they use PHREP style scores like Sanger.  Converting a pair of
FASTA and QUAL files into a Sanger style FASTQ file is easy. To extract QUAL
files from a Roche 454 SFF binary file, use the Roche off instrument command
line tool "sffinfo" with the -q or -qual argument.  You can extract a matching
FASTA file using the -s or -seq argument instead.

The bad news is that Solexa/Illumina did things differently - they have their
own scoring system AND their own incompatible versions of the FASTQ format.
Solexa/Illumina quality scores use Q = - 10 log10 ( Pe / (1-Pe) ), which can
be negative.  PHRED scores and Solexa scores are NOT interchangeable (but a
reasonable mapping can be achieved between them, and they are approximately
equal for higher quality reads).

Confusingly early Solexa pipelines produced a FASTQ like file but using their
own score mapping and an ASCII offset of 64. To make things worse, for the
Solexa/Illumina pipeline 1.3 onwards, they introduced a third variant of the
FASTQ file format, this time using PHRED scores (which is more consistent) but
with an ASCII offset of 64.

i.e. There are at least THREE different and INCOMPATIBLE variants of the FASTQ
file format: The original Sanger PHRED standard, and two from Solexa/Illumina.

The good news is that as of CASAVA version 1.8, Illumina sequencers will
produce FASTQ files using the standard Sanger encoding.

You are expected to use this module via the Bio.SeqIO functions, with the
following format names:

    - "qual" means simple quality files using PHRED scores (e.g. from Roche 454)
    - "fastq" means Sanger style FASTQ files using PHRED scores and an ASCII
      offset of 33 (e.g. from the NCBI Short Read Archive and Illumina 1.8+).
      These can potentially hold PHRED scores from 0 to 93.
    - "fastq-sanger" is an alias for "fastq".
    - "fastq-solexa" means old Solexa (and also very early Illumina) style FASTQ
      files, using Solexa scores with an ASCII offset 64. These can hold Solexa
      scores from -5 to 62.
    - "fastq-illumina" means newer Illumina 1.3 to 1.7 style FASTQ files, using
      PHRED scores but with an ASCII offset 64, allowing PHRED scores from 0
      to 62.

We could potentially add support for "qual-solexa" meaning QUAL files which
contain Solexa scores, but thus far there isn't any reason to use such files.

For example, consider the following short FASTQ file::

    @EAS54_6_R1_2_1_413_324
    CCCTTCTTGTCTTCAGCGTTTCTCC
    +
    ;;3;;;;;;;;;;;;7;;;;;;;88
    @EAS54_6_R1_2_1_540_792
    TTGGCAGGCCAAGGCCGATGGATCA
    +
    ;;;;;;;;;;;7;;;;;-;;;3;83
    @EAS54_6_R1_2_1_443_348
    GTTGCTTCTGGCGTGGGTGGGGGGG
    +
    ;;;;;;;;;;;9;7;;.7;393333

This contains three reads of length 25.  From the read length these were
probably originally from an early Solexa/Illumina sequencer but this file
follows the Sanger FASTQ convention (PHRED style qualities with an ASCII
offset of 33).  This means we can parse this file using Bio.SeqIO using
"fastq" as the format name:

>>> from Bio import SeqIO
>>> for record in SeqIO.parse("Quality/example.fastq", "fastq"):
...     print("%s %s" % (record.id, record.seq))
EAS54_6_R1_2_1_413_324 CCCTTCTTGTCTTCAGCGTTTCTCC
EAS54_6_R1_2_1_540_792 TTGGCAGGCCAAGGCCGATGGATCA
EAS54_6_R1_2_1_443_348 GTTGCTTCTGGCGTGGGTGGGGGGG

The qualities are held as a list of integers in each record's annotation:

>>> print(record)
ID: EAS54_6_R1_2_1_443_348
Name: EAS54_6_R1_2_1_443_348
Description: EAS54_6_R1_2_1_443_348
Number of features: 0
Per letter annotation for: phred_quality
Seq('GTTGCTTCTGGCGTGGGTGGGGGGG')
>>> print(record.letter_annotations["phred_quality"])
[26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 22, 26, 26, 13, 22, 26, 18, 24, 18, 18, 18, 18]

You can use the SeqRecord format method to show this in the QUAL format:

>>> print(record.format("qual"))
>EAS54_6_R1_2_1_443_348
26 26 26 26 26 26 26 26 26 26 26 24 26 22 26 26 13 22 26 18
24 18 18 18 18
<BLANKLINE>

Or go back to the FASTQ format, use "fastq" (or "fastq-sanger"):

>>> print(record.format("fastq"))
@EAS54_6_R1_2_1_443_348
GTTGCTTCTGGCGTGGGTGGGGGGG
+
;;;;;;;;;;;9;7;;.7;393333
<BLANKLINE>

Or, using the Illumina 1.3+ FASTQ encoding (PHRED values with an ASCII offset
of 64):

>>> print(record.format("fastq-illumina"))
@EAS54_6_R1_2_1_443_348
GTTGCTTCTGGCGTGGGTGGGGGGG
+
ZZZZZZZZZZZXZVZZMVZRXRRRR
<BLANKLINE>

You can also get Biopython to convert the scores and show a Solexa style
FASTQ file:

>>> print(record.format("fastq-solexa"))
@EAS54_6_R1_2_1_443_348
GTTGCTTCTGGCGTGGGTGGGGGGG
+
ZZZZZZZZZZZXZVZZMVZRXRRRR
<BLANKLINE>

Notice that this is actually the same output as above using "fastq-illumina"
as the format! The reason for this is all these scores are high enough that
the PHRED and Solexa scores are almost equal. The differences become apparent
for poor quality reads. See the functions solexa_quality_from_phred and
phred_quality_from_solexa for more details.

If you wanted to trim your sequences (perhaps to remove low quality regions,
or to remove a primer sequence), try slicing the SeqRecord objects.  e.g.

>>> sub_rec = record[5:15]
>>> print(sub_rec)
ID: EAS54_6_R1_2_1_443_348
Name: EAS54_6_R1_2_1_443_348
Description: EAS54_6_R1_2_1_443_348
Number of features: 0
Per letter annotation for: phred_quality
Seq('TTCTGGCGTG')
>>> print(sub_rec.letter_annotations["phred_quality"])
[26, 26, 26, 26, 26, 26, 24, 26, 22, 26]
>>> print(sub_rec.format("fastq"))
@EAS54_6_R1_2_1_443_348
TTCTGGCGTG
+
;;;;;;9;7;
<BLANKLINE>

If you wanted to, you could read in this FASTQ file, and save it as a QUAL file:

>>> from Bio import SeqIO
>>> record_iterator = SeqIO.parse("Quality/example.fastq", "fastq")
>>> with open("Quality/temp.qual", "w") as out_handle:
...     SeqIO.write(record_iterator, out_handle, "qual")
3

You can of course read in a QUAL file, such as the one we just created:

>>> from Bio import SeqIO
>>> for record in SeqIO.parse("Quality/temp.qual", "qual"):
...     print("%s read of length %d" % (record.id, len(record.seq)))
EAS54_6_R1_2_1_413_324 read of length 25
EAS54_6_R1_2_1_540_792 read of length 25
EAS54_6_R1_2_1_443_348 read of length 25

Notice that QUAL files don't have a proper sequence present!  But the quality
information is there:

>>> print(record)
ID: EAS54_6_R1_2_1_443_348
Name: EAS54_6_R1_2_1_443_348
Description: EAS54_6_R1_2_1_443_348
Number of features: 0
Per letter annotation for: phred_quality
Undefined sequence of length 25
>>> print(record.letter_annotations["phred_quality"])
[26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 22, 26, 26, 13, 22, 26, 18, 24, 18, 18, 18, 18]

Just to keep things tidy, if you are following this example yourself, you can
delete this temporary file now:

>>> import os
>>> os.remove("Quality/temp.qual")

Sometimes you won't have a FASTQ file, but rather just a pair of FASTA and QUAL
files.  Because the Bio.SeqIO system is designed for reading single files, you
would have to read the two in separately and then combine the data.  However,
since this is such a common thing to want to do, there is a helper iterator
defined in this module that does this for you - PairedFastaQualIterator.

Alternatively, if you have enough RAM to hold all the records in memory at once,
then a simple dictionary approach would work:

>>> from Bio import SeqIO
>>> reads = SeqIO.to_dict(SeqIO.parse("Quality/example.fasta", "fasta"))
>>> for rec in SeqIO.parse("Quality/example.qual", "qual"):
...     reads[rec.id].letter_annotations["phred_quality"]=rec.letter_annotations["phred_quality"]

You can then access any record by its key, and get both the sequence and the
quality scores.

>>> print(reads["EAS54_6_R1_2_1_540_792"].format("fastq"))
@EAS54_6_R1_2_1_540_792
TTGGCAGGCCAAGGCCGATGGATCA
+
;;;;;;;;;;;7;;;;;-;;;3;83
<BLANKLINE>

It is important that you explicitly tell Bio.SeqIO which FASTQ variant you are
using ("fastq" or "fastq-sanger" for the Sanger standard using PHRED values,
"fastq-solexa" for the original Solexa/Illumina variant, or "fastq-illumina"
for the more recent variant), as this cannot be detected reliably
automatically.

To illustrate this problem, let's consider an artificial example:

>>> from Bio.Seq import Seq
>>> from Bio.SeqRecord import SeqRecord
>>> test = SeqRecord(Seq("NACGTACGTA"), id="Test", description="Made up!")
>>> print(test.format("fasta"))
>Test Made up!
NACGTACGTA
<BLANKLINE>
>>> print(test.format("fastq"))
Traceback (most recent call last):
 ...
ValueError: No suitable quality scores found in letter_annotations of SeqRecord (id=Test).

We created a sample SeqRecord, and can show it in FASTA format - but for QUAL
or FASTQ format we need to provide some quality scores. These are held as a
list of integers (one for each base) in the letter_annotations dictionary:

>>> test.letter_annotations["phred_quality"] = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40]
>>> print(test.format("qual"))
>Test Made up!
0 1 2 3 4 5 10 20 30 40
<BLANKLINE>
>>> print(test.format("fastq"))
@Test Made up!
NACGTACGTA
+
!"#$%&+5?I
<BLANKLINE>

We can check this FASTQ encoding - the first PHRED quality was zero, and this
mapped to a exclamation mark, while the final score was 40 and this mapped to
the letter "I":

>>> ord('!') - 33
0
>>> ord('I') - 33
40
>>> [ord(letter)-33 for letter in '!"#$%&+5?I']
[0, 1, 2, 3, 4, 5, 10, 20, 30, 40]

Similarly, we could produce an Illumina 1.3 to 1.7 style FASTQ file using PHRED
scores with an offset of 64:

>>> print(test.format("fastq-illumina"))
@Test Made up!
NACGTACGTA
+
@ABCDEJT^h
<BLANKLINE>

And we can check this too - the first PHRED score was zero, and this mapped to
"@", while the final score was 40 and this mapped to "h":

>>> ord("@") - 64
0
>>> ord("h") - 64
40
>>> [ord(letter)-64 for letter in "@ABCDEJT^h"]
[0, 1, 2, 3, 4, 5, 10, 20, 30, 40]

Notice how different the standard Sanger FASTQ and the Illumina 1.3 to 1.7 style
FASTQ files look for the same data! Then we have the older Solexa/Illumina
format to consider which encodes Solexa scores instead of PHRED scores.

First let's see what Biopython says if we convert the PHRED scores into Solexa
scores (rounding to one decimal place):

>>> for q in [0, 1, 2, 3, 4, 5, 10, 20, 30, 40]:
...     print("PHRED %i maps to Solexa %0.1f" % (q, solexa_quality_from_phred(q)))
PHRED 0 maps to Solexa -5.0
PHRED 1 maps to Solexa -5.0
PHRED 2 maps to Solexa -2.3
PHRED 3 maps to Solexa -0.0
PHRED 4 maps to Solexa 1.8
PHRED 5 maps to Solexa 3.3
PHRED 10 maps to Solexa 9.5
PHRED 20 maps to Solexa 20.0
PHRED 30 maps to Solexa 30.0
PHRED 40 maps to Solexa 40.0

Now here is the record using the old Solexa style FASTQ file:

>>> print(test.format("fastq-solexa"))
@Test Made up!
NACGTACGTA
+
;;>@BCJT^h
<BLANKLINE>

Again, this is using an ASCII offset of 64, so we can check the Solexa scores:

>>> [ord(letter)-64 for letter in ";;>@BCJT^h"]
[-5, -5, -2, 0, 2, 3, 10, 20, 30, 40]

This explains why the last few letters of this FASTQ output matched that using
the Illumina 1.3 to 1.7 format - high quality PHRED scores and Solexa scores
are approximately equal.

"""
import warnings
from math import log
from Bio import BiopythonParserWarning
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio import StreamModeError
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from typing import Any, Callable, Iterator, IO, List, Mapping, Optional, Sequence, Tuple, Union
SANGER_SCORE_OFFSET = 33
SOLEXA_SCORE_OFFSET = 64

def solexa_quality_from_phred(phred_quality: float) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Convert a PHRED quality (range 0 to about 90) to a Solexa quality.\n\n    PHRED and Solexa quality scores are both log transformations of a\n    probality of error (high score = low probability of error). This function\n    takes a PHRED score, transforms it back to a probability of error, and\n    then re-expresses it as a Solexa score. This assumes the error estimates\n    are equivalent.\n\n    How does this work exactly? Well the PHRED quality is minus ten times the\n    base ten logarithm of the probability of error::\n\n        phred_quality = -10*log(error,10)\n\n    Therefore, turning this round::\n\n        error = 10 ** (- phred_quality / 10)\n\n    Now, Solexa qualities use a different log transformation::\n\n        solexa_quality = -10*log(error/(1-error),10)\n\n    After substitution and a little manipulation we get::\n\n         solexa_quality = 10*log(10**(phred_quality/10.0) - 1, 10)\n\n    However, real Solexa files use a minimum quality of -5. This does have a\n    good reason - a random base call would be correct 25% of the time,\n    and thus have a probability of error of 0.75, which gives 1.25 as the PHRED\n    quality, or -4.77 as the Solexa quality. Thus (after rounding), a random\n    nucleotide read would have a PHRED quality of 1, or a Solexa quality of -5.\n\n    Taken literally, this logarithic formula would map a PHRED quality of zero\n    to a Solexa quality of minus infinity. Of course, taken literally, a PHRED\n    score of zero means a probability of error of one (i.e. the base call is\n    definitely wrong), which is worse than random! In practice, a PHRED quality\n    of zero usually means a default value, or perhaps random - and therefore\n    mapping it to the minimum Solexa score of -5 is reasonable.\n\n    In conclusion, we follow EMBOSS, and take this logarithmic formula but also\n    apply a minimum value of -5.0 for the Solexa quality, and also map a PHRED\n    quality of zero to -5.0 as well.\n\n    Note this function will return a floating point number, it is up to you to\n    round this to the nearest integer if appropriate.  e.g.\n\n    >>> print("%0.2f" % round(solexa_quality_from_phred(80), 2))\n    80.00\n    >>> print("%0.2f" % round(solexa_quality_from_phred(50), 2))\n    50.00\n    >>> print("%0.2f" % round(solexa_quality_from_phred(20), 2))\n    19.96\n    >>> print("%0.2f" % round(solexa_quality_from_phred(10), 2))\n    9.54\n    >>> print("%0.2f" % round(solexa_quality_from_phred(5), 2))\n    3.35\n    >>> print("%0.2f" % round(solexa_quality_from_phred(4), 2))\n    1.80\n    >>> print("%0.2f" % round(solexa_quality_from_phred(3), 2))\n    -0.02\n    >>> print("%0.2f" % round(solexa_quality_from_phred(2), 2))\n    -2.33\n    >>> print("%0.2f" % round(solexa_quality_from_phred(1), 2))\n    -5.00\n    >>> print("%0.2f" % round(solexa_quality_from_phred(0), 2))\n    -5.00\n\n    Notice that for high quality reads PHRED and Solexa scores are numerically\n    equal. The differences are important for poor quality reads, where PHRED\n    has a minimum of zero but Solexa scores can be negative.\n\n    Finally, as a special case where None is used for a "missing value", None\n    is returned:\n\n    >>> print(solexa_quality_from_phred(None))\n    None\n    '
    if phred_quality is None:
        return None
    elif phred_quality > 0:
        return max(-5.0, 10 * log(10 ** (phred_quality / 10.0) - 1, 10))
    elif phred_quality == 0:
        return -5.0
    else:
        raise ValueError(f'PHRED qualities must be positive (or zero), not {phred_quality!r}')

def phred_quality_from_solexa(solexa_quality: float) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Convert a Solexa quality (which can be negative) to a PHRED quality.\n\n    PHRED and Solexa quality scores are both log transformations of a\n    probality of error (high score = low probability of error). This function\n    takes a Solexa score, transforms it back to a probability of error, and\n    then re-expresses it as a PHRED score. This assumes the error estimates\n    are equivalent.\n\n    The underlying formulas are given in the documentation for the sister\n    function solexa_quality_from_phred, in this case the operation is::\n\n        phred_quality = 10*log(10**(solexa_quality/10.0) + 1, 10)\n\n    This will return a floating point number, it is up to you to round this to\n    the nearest integer if appropriate.  e.g.\n\n    >>> print("%0.2f" % round(phred_quality_from_solexa(80), 2))\n    80.00\n    >>> print("%0.2f" % round(phred_quality_from_solexa(20), 2))\n    20.04\n    >>> print("%0.2f" % round(phred_quality_from_solexa(10), 2))\n    10.41\n    >>> print("%0.2f" % round(phred_quality_from_solexa(0), 2))\n    3.01\n    >>> print("%0.2f" % round(phred_quality_from_solexa(-5), 2))\n    1.19\n\n    Note that a solexa_quality less then -5 is not expected, will trigger a\n    warning, but will still be converted as per the logarithmic mapping\n    (giving a number between 0 and 1.19 back).\n\n    As a special case where None is used for a "missing value", None is\n    returned:\n\n    >>> print(phred_quality_from_solexa(None))\n    None\n    '
    if solexa_quality is None:
        return None
    if solexa_quality < -5:
        warnings.warn(f'Solexa quality less than -5 passed, {solexa_quality!r}', BiopythonWarning)
    return 10 * log(10 ** (solexa_quality / 10.0) + 1, 10)

def _get_phred_quality(record: SeqRecord) -> Union[List[float], List[int]]:
    if False:
        while True:
            i = 10
    "Extract PHRED qualities from a SeqRecord's letter_annotations (PRIVATE).\n\n    If there are no PHRED qualities, but there are Solexa qualities, those are\n    used instead after conversion.\n    "
    try:
        return record.letter_annotations['phred_quality']
    except KeyError:
        pass
    try:
        return [phred_quality_from_solexa(q) for q in record.letter_annotations['solexa_quality']]
    except KeyError:
        raise ValueError('No suitable quality scores found in letter_annotations of SeqRecord (id=%s).' % record.id) from None
_phred_to_sanger_quality_str = {qp: chr(min(126, qp + SANGER_SCORE_OFFSET)) for qp in range(93 + 1)}
_solexa_to_sanger_quality_str = {qs: chr(min(126, int(round(phred_quality_from_solexa(qs)) + SANGER_SCORE_OFFSET))) for qs in range(-5, 93 + 1)}

def _get_sanger_quality_str(record: SeqRecord) -> str:
    if False:
        print('Hello World!')
    'Return a Sanger FASTQ encoded quality string (PRIVATE).\n\n    >>> from Bio.Seq import Seq\n    >>> from Bio.SeqRecord import SeqRecord\n    >>> r = SeqRecord(Seq("ACGTAN"), id="Test",\n    ...               letter_annotations = {"phred_quality":[50, 40, 30, 20, 10, 0]})\n    >>> _get_sanger_quality_str(r)\n    \'SI?5+!\'\n\n    If as in the above example (or indeed a SeqRecord parser with Bio.SeqIO),\n    the PHRED qualities are integers, this function is able to use a very fast\n    pre-cached mapping. However, if they are floats which differ slightly, then\n    it has to do the appropriate rounding - which is slower:\n\n    >>> r2 = SeqRecord(Seq("ACGTAN"), id="Test2",\n    ...      letter_annotations = {"phred_quality":[50.0, 40.05, 29.99, 20, 9.55, 0.01]})\n    >>> _get_sanger_quality_str(r2)\n    \'SI?5+!\'\n\n    If your scores include a None value, this raises an exception:\n\n    >>> r3 = SeqRecord(Seq("ACGTAN"), id="Test3",\n    ...               letter_annotations = {"phred_quality":[50, 40, 30, 20, 10, None]})\n    >>> _get_sanger_quality_str(r3)\n    Traceback (most recent call last):\n       ...\n    TypeError: A quality value of None was found\n\n    If (strangely) your record has both PHRED and Solexa scores, then the PHRED\n    scores are used in preference:\n\n    >>> r4 = SeqRecord(Seq("ACGTAN"), id="Test4",\n    ...               letter_annotations = {"phred_quality":[50, 40, 30, 20, 10, 0],\n    ...                                     "solexa_quality":[-5, -4, 0, None, 0, 40]})\n    >>> _get_sanger_quality_str(r4)\n    \'SI?5+!\'\n\n    If there are no PHRED scores, but there are Solexa scores, these are used\n    instead (after the appropriate conversion):\n\n    >>> r5 = SeqRecord(Seq("ACGTAN"), id="Test5",\n    ...      letter_annotations = {"solexa_quality":[40, 30, 20, 10, 0, -5]})\n    >>> _get_sanger_quality_str(r5)\n    \'I?5+$"\'\n\n    Again, integer Solexa scores can be looked up in a pre-cached mapping making\n    this very fast. You can still use approximate floating point scores:\n\n    >>> r6 = SeqRecord(Seq("ACGTAN"), id="Test6",\n    ...      letter_annotations = {"solexa_quality":[40.1, 29.7, 20.01, 10, 0.0, -4.9]})\n    >>> _get_sanger_quality_str(r6)\n    \'I?5+$"\'\n\n    Notice that due to the limited range of printable ASCII characters, a\n    PHRED quality of 93 is the maximum that can be held in an Illumina FASTQ\n    file (using ASCII 126, the tilde). This function will issue a warning\n    in this situation.\n    '
    try:
        qualities = record.letter_annotations['phred_quality']
    except KeyError:
        pass
    else:
        try:
            return ''.join((_phred_to_sanger_quality_str[qp] for qp in qualities))
        except KeyError:
            pass
        if None in qualities:
            raise TypeError('A quality value of None was found')
        if max(qualities) >= 93.5:
            warnings.warn('Data loss - max PHRED quality 93 in Sanger FASTQ', BiopythonWarning)
        return ''.join((chr(min(126, int(round(qp)) + SANGER_SCORE_OFFSET)) for qp in qualities))
    try:
        qualities = record.letter_annotations['solexa_quality']
    except KeyError:
        raise ValueError('No suitable quality scores found in letter_annotations of SeqRecord (id=%s).' % record.id) from None
    try:
        return ''.join((_solexa_to_sanger_quality_str[qs] for qs in qualities))
    except KeyError:
        pass
    if None in qualities:
        raise TypeError('A quality value of None was found')
    if max(qualities) >= 93.5:
        warnings.warn('Data loss - max PHRED quality 93 in Sanger FASTQ', BiopythonWarning)
    return ''.join((chr(min(126, int(round(phred_quality_from_solexa(qs))) + SANGER_SCORE_OFFSET)) for qs in qualities))
assert 62 + SOLEXA_SCORE_OFFSET == 126
_phred_to_illumina_quality_str = {qp: chr(qp + SOLEXA_SCORE_OFFSET) for qp in range(62 + 1)}
_solexa_to_illumina_quality_str = {qs: chr(int(round(phred_quality_from_solexa(qs))) + SOLEXA_SCORE_OFFSET) for qs in range(-5, 62 + 1)}

def _get_illumina_quality_str(record: SeqRecord) -> str:
    if False:
        print('Hello World!')
    'Return an Illumina 1.3 to 1.7 FASTQ encoded quality string (PRIVATE).\n\n    Notice that due to the limited range of printable ASCII characters, a\n    PHRED quality of 62 is the maximum that can be held in an Illumina FASTQ\n    file (using ASCII 126, the tilde). This function will issue a warning\n    in this situation.\n    '
    try:
        qualities = record.letter_annotations['phred_quality']
    except KeyError:
        pass
    else:
        try:
            return ''.join((_phred_to_illumina_quality_str[qp] for qp in qualities))
        except KeyError:
            pass
        if None in qualities:
            raise TypeError('A quality value of None was found')
        if max(qualities) >= 62.5:
            warnings.warn('Data loss - max PHRED quality 62 in Illumina FASTQ', BiopythonWarning)
        return ''.join((chr(min(126, int(round(qp)) + SOLEXA_SCORE_OFFSET)) for qp in qualities))
    try:
        qualities = record.letter_annotations['solexa_quality']
    except KeyError:
        raise ValueError('No suitable quality scores found in letter_annotations of SeqRecord (id=%s).' % record.id) from None
    try:
        return ''.join((_solexa_to_illumina_quality_str[qs] for qs in qualities))
    except KeyError:
        pass
    if None in qualities:
        raise TypeError('A quality value of None was found')
    if max(qualities) >= 62.5:
        warnings.warn('Data loss - max PHRED quality 62 in Illumina FASTQ', BiopythonWarning)
    return ''.join((chr(min(126, int(round(phred_quality_from_solexa(qs))) + SOLEXA_SCORE_OFFSET)) for qs in qualities))
assert 62 + SOLEXA_SCORE_OFFSET == 126
_solexa_to_solexa_quality_str = {qs: chr(min(126, qs + SOLEXA_SCORE_OFFSET)) for qs in range(-5, 62 + 1)}
_phred_to_solexa_quality_str = {qp: chr(min(126, int(round(solexa_quality_from_phred(qp))) + SOLEXA_SCORE_OFFSET)) for qp in range(62 + 1)}

def _get_solexa_quality_str(record: SeqRecord) -> str:
    if False:
        return 10
    'Return a Solexa FASTQ encoded quality string (PRIVATE).\n\n    Notice that due to the limited range of printable ASCII characters, a\n    Solexa quality of 62 is the maximum that can be held in a Solexa FASTQ\n    file (using ASCII 126, the tilde). This function will issue a warning\n    in this situation.\n    '
    try:
        qualities = record.letter_annotations['solexa_quality']
    except KeyError:
        pass
    else:
        try:
            return ''.join((_solexa_to_solexa_quality_str[qs] for qs in qualities))
        except KeyError:
            pass
        if None in qualities:
            raise TypeError('A quality value of None was found')
        if max(qualities) >= 62.5:
            warnings.warn('Data loss - max Solexa quality 62 in Solexa FASTQ', BiopythonWarning)
        return ''.join((chr(min(126, int(round(qs)) + SOLEXA_SCORE_OFFSET)) for qs in qualities))
    try:
        qualities = record.letter_annotations['phred_quality']
    except KeyError:
        raise ValueError('No suitable quality scores found in letter_annotations of SeqRecord (id=%s).' % record.id) from None
    try:
        return ''.join((_phred_to_solexa_quality_str[qp] for qp in qualities))
    except KeyError:
        pass
    if None in qualities:
        raise TypeError('A quality value of None was found')
    if max(qualities) >= 62.5:
        warnings.warn('Data loss - max Solexa quality 62 in Solexa FASTQ', BiopythonWarning)
    return ''.join((chr(min(126, int(round(solexa_quality_from_phred(qp))) + SOLEXA_SCORE_OFFSET)) for qp in qualities))

def FastqGeneralIterator(source: _TextIOSource) -> Iterator[Tuple[str, str, str]]:
    if False:
        for i in range(10):
            print('nop')
    'Iterate over Fastq records as string tuples (not as SeqRecord objects).\n\n    Arguments:\n     - source - input stream opened in text mode, or a path to a file\n\n    This code does not try to interpret the quality string numerically.  It\n    just returns tuples of the title, sequence and quality as strings.  For\n    the sequence and quality, any whitespace (such as new lines) is removed.\n\n    Our SeqRecord based FASTQ iterators call this function internally, and then\n    turn the strings into a SeqRecord objects, mapping the quality string into\n    a list of numerical scores.  If you want to do a custom quality mapping,\n    then you might consider calling this function directly.\n\n    For parsing FASTQ files, the title string from the "@" line at the start\n    of each record can optionally be omitted on the "+" lines.  If it is\n    repeated, it must be identical.\n\n    The sequence string and the quality string can optionally be split over\n    multiple lines, although several sources discourage this.  In comparison,\n    for the FASTA file format line breaks between 60 and 80 characters are\n    the norm.\n\n    **WARNING** - Because the "@" character can appear in the quality string,\n    this can cause problems as this is also the marker for the start of\n    a new sequence.  In fact, the "+" sign can also appear as well.  Some\n    sources recommended having no line breaks in the  quality to avoid this,\n    but even that is not enough, consider this example::\n\n        @071113_EAS56_0053:1:1:998:236\n        TTTCTTGCCCCCATAGACTGAGACCTTCCCTAAATA\n        +071113_EAS56_0053:1:1:998:236\n        IIIIIIIIIIIIIIIIIIIIIIIIIIIIICII+III\n        @071113_EAS56_0053:1:1:182:712\n        ACCCAGCTAATTTTTGTATTTTTGTTAGAGACAGTG\n        +\n        @IIIIIIIIIIIIIIICDIIIII<%<6&-*).(*%+\n        @071113_EAS56_0053:1:1:153:10\n        TGTTCTGAAGGAAGGTGTGCGTGCGTGTGTGTGTGT\n        +\n        IIIIIIIIIIIICIIGIIIII>IAIIIE65I=II:6\n        @071113_EAS56_0053:1:3:990:501\n        TGGGAGGTTTTATGTGGA\n        AAGCAGCAATGTACAAGA\n        +\n        IIIIIII.IIIIII1@44\n        @-7.%<&+/$/%4(++(%\n\n    This is four PHRED encoded FASTQ entries originally from an NCBI source\n    (given the read length of 36, these are probably Solexa Illumina reads where\n    the quality has been mapped onto the PHRED values).\n\n    This example has been edited to illustrate some of the nasty things allowed\n    in the FASTQ format.  Firstly, on the "+" lines most but not all of the\n    (redundant) identifiers are omitted.  In real files it is likely that all or\n    none of these extra identifiers will be present.\n\n    Secondly, while the first three sequences have been shown without line\n    breaks, the last has been split over multiple lines.  In real files any line\n    breaks are likely to be consistent.\n\n    Thirdly, some of the quality string lines start with an "@" character.  For\n    the second record this is unavoidable.  However for the fourth sequence this\n    only happens because its quality string is split over two lines.  A naive\n    parser could wrongly treat any line starting with an "@" as the beginning of\n    a new sequence!  This code copes with this possible ambiguity by keeping\n    track of the length of the sequence which gives the expected length of the\n    quality string.\n\n    Using this tricky example file as input, this short bit of code demonstrates\n    what this parsing function would return:\n\n    >>> with open("Quality/tricky.fastq") as handle:\n    ...     for (title, sequence, quality) in FastqGeneralIterator(handle):\n    ...         print(title)\n    ...         print("%s %s" % (sequence, quality))\n    ...\n    071113_EAS56_0053:1:1:998:236\n    TTTCTTGCCCCCATAGACTGAGACCTTCCCTAAATA IIIIIIIIIIIIIIIIIIIIIIIIIIIIICII+III\n    071113_EAS56_0053:1:1:182:712\n    ACCCAGCTAATTTTTGTATTTTTGTTAGAGACAGTG @IIIIIIIIIIIIIIICDIIIII<%<6&-*).(*%+\n    071113_EAS56_0053:1:1:153:10\n    TGTTCTGAAGGAAGGTGTGCGTGCGTGTGTGTGTGT IIIIIIIIIIIICIIGIIIII>IAIIIE65I=II:6\n    071113_EAS56_0053:1:3:990:501\n    TGGGAGGTTTTATGTGGAAAGCAGCAATGTACAAGA IIIIIII.IIIIII1@44@-7.%<&+/$/%4(++(%\n\n    Finally we note that some sources state that the quality string should\n    start with "!" (which using the PHRED mapping means the first letter always\n    has a quality score of zero).  This rather restrictive rule is not widely\n    observed, so is therefore ignored here.  One plus point about this "!" rule\n    is that (provided there are no line breaks in the quality sequence) it\n    would prevent the above problem with the "@" character.\n    '
    with as_handle(source) as handle:
        if handle.read(0) != '':
            raise StreamModeError('Fastq files must be opened in text mode') from None
        try:
            line = next(handle)
        except StopIteration:
            return
        while True:
            if line[0] != '@':
                raise ValueError("Records in Fastq files should start with '@' character")
            title_line = line[1:].rstrip()
            seq_string = ''
            for line in handle:
                if line[0] == '+':
                    break
                seq_string += line.rstrip()
            else:
                if seq_string:
                    raise ValueError('End of file without quality information.')
                else:
                    raise ValueError('Unexpected end of file')
            second_title = line[1:].rstrip()
            if second_title and second_title != title_line:
                raise ValueError('Sequence and quality captions differ.')
            if ' ' in seq_string or '\t' in seq_string:
                raise ValueError('Whitespace is not allowed in the sequence.')
            seq_len = len(seq_string)
            line = None
            quality_string = ''
            for line in handle:
                if line[0] == '@':
                    if len(quality_string) >= seq_len:
                        break
                quality_string += line.rstrip()
            else:
                if line is None:
                    raise ValueError('Unexpected end of file')
                line = None
            if seq_len != len(quality_string):
                raise ValueError('Lengths of sequence and quality values differs for %s (%i and %i).' % (title_line, seq_len, len(quality_string)))
            yield (title_line, seq_string, quality_string)
            if line is None:
                break

class FastqPhredIterator(SequenceIterator[str]):
    """Parser for FASTQ files."""

    def __init__(self, source: _TextIOSource, alphabet: None=None, title2ids: Optional[Callable[[str], Tuple[str, str, str]]]=None):
        if False:
            print('Hello World!')
        'Iterate over FASTQ records as SeqRecord objects.\n\n        Arguments:\n         - source - input stream opened in text mode, or a path to a file\n         - alphabet - optional alphabet, no longer used. Leave as None.\n         - title2ids (DEPRECATED) - A function that, when given the title line\n           from the FASTQ file (without the beginning >), will return the id,\n           name and description (in that order) for the record as a tuple of\n           strings.  If this is not given, then the entire title line will be\n           used as the description, and the first word as the id and name.\n\n        The use of title2ids matches that of Bio.SeqIO.FastaIO.\n\n        For each sequence in a (Sanger style) FASTQ file there is a matching string\n        encoding the PHRED qualities (integers between 0 and about 90) using ASCII\n        values with an offset of 33.\n\n        For example, consider a file containing three short reads::\n\n            @EAS54_6_R1_2_1_413_324\n            CCCTTCTTGTCTTCAGCGTTTCTCC\n            +\n            ;;3;;;;;;;;;;;;7;;;;;;;88\n            @EAS54_6_R1_2_1_540_792\n            TTGGCAGGCCAAGGCCGATGGATCA\n            +\n            ;;;;;;;;;;;7;;;;;-;;;3;83\n            @EAS54_6_R1_2_1_443_348\n            GTTGCTTCTGGCGTGGGTGGGGGGG\n            +\n            ;;;;;;;;;;;9;7;;.7;393333\n\n        For each sequence (e.g. "CCCTTCTTGTCTTCAGCGTTTCTCC") there is a matching\n        string encoding the PHRED qualities using a ASCII values with an offset of\n        33 (e.g. ";;3;;;;;;;;;;;;7;;;;;;;88").\n\n        Using this module directly you might run:\n\n        >>> with open("Quality/example.fastq") as handle:\n        ...     for record in FastqPhredIterator(handle):\n        ...         print("%s %s" % (record.id, record.seq))\n        EAS54_6_R1_2_1_413_324 CCCTTCTTGTCTTCAGCGTTTCTCC\n        EAS54_6_R1_2_1_540_792 TTGGCAGGCCAAGGCCGATGGATCA\n        EAS54_6_R1_2_1_443_348 GTTGCTTCTGGCGTGGGTGGGGGGG\n\n        Typically however, you would call this via Bio.SeqIO instead with "fastq"\n        (or "fastq-sanger") as the format:\n\n        >>> from Bio import SeqIO\n        >>> with open("Quality/example.fastq") as handle:\n        ...     for record in SeqIO.parse(handle, "fastq"):\n        ...         print("%s %s" % (record.id, record.seq))\n        EAS54_6_R1_2_1_413_324 CCCTTCTTGTCTTCAGCGTTTCTCC\n        EAS54_6_R1_2_1_540_792 TTGGCAGGCCAAGGCCGATGGATCA\n        EAS54_6_R1_2_1_443_348 GTTGCTTCTGGCGTGGGTGGGGGGG\n\n        If you want to look at the qualities, they are record in each record\'s\n        per-letter-annotation dictionary as a simple list of integers:\n\n        >>> print(record.letter_annotations["phred_quality"])\n        [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 22, 26, 26, 13, 22, 26, 18, 24, 18, 18, 18, 18]\n\n        The title2ids argument is deprecated. Instead, please use a generator\n        function to modify the records returned by the parser. For example, to\n        store the mean PHRED quality in the record description, use\n\n        >>> from statistics import mean\n        >>> def modify_records(records):\n        ...     for record in records:\n        ...         record.description = mean(record.letter_annotations[\'phred_quality\'])\n        ...         yield record\n        ...\n        >>> with open(\'Quality/example.fastq\') as handle:\n        ...     for record in modify_records(FastqPhredIterator(handle)):\n        ...         print(record.id, record.description)\n        ...\n        EAS54_6_R1_2_1_413_324 25.28\n        EAS54_6_R1_2_1_540_792 24.52\n        EAS54_6_R1_2_1_443_348 23.4\n\n        '
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        if title2ids is not None:
            warnings.warn("The title2ids argument is deprecated. Instead, please use a generator function to modify records returned by the parser. For example, to change the record description to a counter, use\n\n>>> from statistics import mean\n>>> def modify_records(records):\n...     for record in records:\n...         record.description = mean(record.letter_annotations['phred_quality'])\n...         yield record\n...\n>>> with open('Quality/example.fastq') as handle:\n...     for record in modify_records(FastqPhredIterator(handle)):\n...         print(record.id, record.description)\n\n", BiopythonDeprecationWarning)
        self.title2ids = title2ids
        super().__init__(source, mode='t', fmt='Fastq')

    def parse(self, handle: IO[str]) -> Iterator[SeqRecord]:
        if False:
            print('Hello World!')
        'Start parsing the file, and return a SeqRecord iterator.'
        records = self.iterate(handle)
        return records

    def iterate(self, handle: IO[str]) -> Iterator[SeqRecord]:
        if False:
            return 10
        'Parse the file and generate SeqRecord objects.'
        title2ids = self.title2ids
        assert SANGER_SCORE_OFFSET == ord('!')
        q_mapping = {chr(letter): letter - SANGER_SCORE_OFFSET for letter in range(SANGER_SCORE_OFFSET, 94 + SANGER_SCORE_OFFSET)}
        for (title_line, seq_string, quality_string) in FastqGeneralIterator(handle):
            if title2ids:
                (id, name, descr) = title2ids(title_line)
            else:
                descr = title_line
                id = descr.split()[0]
                name = id
            record = SeqRecord(Seq(seq_string), id=id, name=name, description=descr)
            try:
                qualities = [q_mapping[letter2] for letter2 in quality_string]
            except KeyError:
                raise ValueError('Invalid character in quality string') from None
            dict.__setitem__(record._per_letter_annotations, 'phred_quality', qualities)
            yield record

def FastqSolexaIterator(source: _TextIOSource, alphabet: None=None, title2ids: Optional[Callable[[str], Tuple[str, str, str]]]=None) -> Iterator[SeqRecord]:
    if False:
        while True:
            i = 10
    'Parse old Solexa/Illumina FASTQ like files (which differ in the quality mapping).\n\n    The optional arguments are the same as those for the FastqPhredIterator.\n\n    For each sequence in Solexa/Illumina FASTQ files there is a matching string\n    encoding the Solexa integer qualities using ASCII values with an offset\n    of 64.  Solexa scores are scaled differently to PHRED scores, and Biopython\n    will NOT perform any automatic conversion when loading.\n\n    NOTE - This file format is used by the OLD versions of the Solexa/Illumina\n    pipeline. See also the FastqIlluminaIterator function for the NEW version.\n\n    For example, consider a file containing these five records::\n\n        @SLXA-B3_649_FC8437_R1_1_1_610_79\n        GATGTGCAATACCTTTGTAGAGGAA\n        +SLXA-B3_649_FC8437_R1_1_1_610_79\n        YYYYYYYYYYYYYYYYYYWYWYYSU\n        @SLXA-B3_649_FC8437_R1_1_1_397_389\n        GGTTTGAGAAAGAGAAATGAGATAA\n        +SLXA-B3_649_FC8437_R1_1_1_397_389\n        YYYYYYYYYWYYYYWWYYYWYWYWW\n        @SLXA-B3_649_FC8437_R1_1_1_850_123\n        GAGGGTGTTGATCATGATGATGGCG\n        +SLXA-B3_649_FC8437_R1_1_1_850_123\n        YYYYYYYYYYYYYWYYWYYSYYYSY\n        @SLXA-B3_649_FC8437_R1_1_1_362_549\n        GGAAACAAAGTTTTTCTCAACATAG\n        +SLXA-B3_649_FC8437_R1_1_1_362_549\n        YYYYYYYYYYYYYYYYYYWWWWYWY\n        @SLXA-B3_649_FC8437_R1_1_1_183_714\n        GTATTATTTAATGGCATACACTCAA\n        +SLXA-B3_649_FC8437_R1_1_1_183_714\n        YYYYYYYYYYWYYYYWYWWUWWWQQ\n\n    Using this module directly you might run:\n\n    >>> with open("Quality/solexa_example.fastq") as handle:\n    ...     for record in FastqSolexaIterator(handle):\n    ...         print("%s %s" % (record.id, record.seq))\n    SLXA-B3_649_FC8437_R1_1_1_610_79 GATGTGCAATACCTTTGTAGAGGAA\n    SLXA-B3_649_FC8437_R1_1_1_397_389 GGTTTGAGAAAGAGAAATGAGATAA\n    SLXA-B3_649_FC8437_R1_1_1_850_123 GAGGGTGTTGATCATGATGATGGCG\n    SLXA-B3_649_FC8437_R1_1_1_362_549 GGAAACAAAGTTTTTCTCAACATAG\n    SLXA-B3_649_FC8437_R1_1_1_183_714 GTATTATTTAATGGCATACACTCAA\n\n    Typically however, you would call this via Bio.SeqIO instead with\n    "fastq-solexa" as the format:\n\n    >>> from Bio import SeqIO\n    >>> with open("Quality/solexa_example.fastq") as handle:\n    ...     for record in SeqIO.parse(handle, "fastq-solexa"):\n    ...         print("%s %s" % (record.id, record.seq))\n    SLXA-B3_649_FC8437_R1_1_1_610_79 GATGTGCAATACCTTTGTAGAGGAA\n    SLXA-B3_649_FC8437_R1_1_1_397_389 GGTTTGAGAAAGAGAAATGAGATAA\n    SLXA-B3_649_FC8437_R1_1_1_850_123 GAGGGTGTTGATCATGATGATGGCG\n    SLXA-B3_649_FC8437_R1_1_1_362_549 GGAAACAAAGTTTTTCTCAACATAG\n    SLXA-B3_649_FC8437_R1_1_1_183_714 GTATTATTTAATGGCATACACTCAA\n\n    If you want to look at the qualities, they are recorded in each record\'s\n    per-letter-annotation dictionary as a simple list of integers:\n\n    >>> print(record.letter_annotations["solexa_quality"])\n    [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 23, 25, 25, 25, 25, 23, 25, 23, 23, 21, 23, 23, 23, 17, 17]\n\n    These scores aren\'t very good, but they are high enough that they map\n    almost exactly onto PHRED scores:\n\n    >>> print("%0.2f" % phred_quality_from_solexa(25))\n    25.01\n\n    Let\'s look at faked example read which is even worse, where there are\n    more noticeable differences between the Solexa and PHRED scores::\n\n         @slxa_0001_1_0001_01\n         ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n         +slxa_0001_1_0001_01\n         hgfedcba`_^]\\[ZYXWVUTSRQPONMLKJIHGFEDCBA@?>=<;\n\n    Again, you would typically use Bio.SeqIO to read this file in (rather than\n    calling the Bio.SeqIO.QualtityIO module directly).  Most FASTQ files will\n    contain thousands of reads, so you would normally use Bio.SeqIO.parse()\n    as shown above.  This example has only as one entry, so instead we can\n    use the Bio.SeqIO.read() function:\n\n    >>> from Bio import SeqIO\n    >>> with open("Quality/solexa_faked.fastq") as handle:\n    ...     record = SeqIO.read(handle, "fastq-solexa")\n    >>> print("%s %s" % (record.id, record.seq))\n    slxa_0001_1_0001_01 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n    >>> print(record.letter_annotations["solexa_quality"])\n    [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]\n\n    These quality scores are so low that when converted from the Solexa scheme\n    into PHRED scores they look quite different:\n\n    >>> print("%0.2f" % phred_quality_from_solexa(-1))\n    2.54\n    >>> print("%0.2f" % phred_quality_from_solexa(-5))\n    1.19\n\n    Note you can use the Bio.SeqIO.write() function or the SeqRecord\'s format\n    method to output the record(s):\n\n    >>> print(record.format("fastq-solexa"))\n    @slxa_0001_1_0001_01\n    ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n    +\n    hgfedcba`_^]\\[ZYXWVUTSRQPONMLKJIHGFEDCBA@?>=<;\n    <BLANKLINE>\n\n    Note this output is slightly different from the input file as Biopython\n    has left out the optional repetition of the sequence identifier on the "+"\n    line.  If you want the to use PHRED scores, use "fastq" or "qual" as the\n    output format instead, and Biopython will do the conversion for you:\n\n    >>> print(record.format("fastq"))\n    @slxa_0001_1_0001_01\n    ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n    +\n    IHGFEDCBA@?>=<;:9876543210/.-,++*)(\'&&%%$$##""\n    <BLANKLINE>\n\n    >>> print(record.format("qual"))\n    >slxa_0001_1_0001_01\n    40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21\n    20 19 18 17 16 15 14 13 12 11 10 10 9 8 7 6 5 5 4 4 3 3 2 2\n    1 1\n    <BLANKLINE>\n\n    As shown above, the poor quality Solexa reads have been mapped to the\n    equivalent PHRED score (e.g. -5 to 1 as shown earlier).\n    '
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    q_mapping = {chr(letter): letter - SOLEXA_SCORE_OFFSET for letter in range(SOLEXA_SCORE_OFFSET - 5, 63 + SOLEXA_SCORE_OFFSET)}
    for (title_line, seq_string, quality_string) in FastqGeneralIterator(source):
        if title2ids:
            (id, name, descr) = title2ids(title_line)
        else:
            descr = title_line
            id = descr.split()[0]
            name = id
        record = SeqRecord(Seq(seq_string), id=id, name=name, description=descr)
        try:
            qualities = [q_mapping[letter2] for letter2 in quality_string]
        except KeyError:
            raise ValueError('Invalid character in quality string') from None
        dict.__setitem__(record._per_letter_annotations, 'solexa_quality', qualities)
        yield record

def FastqIlluminaIterator(source: _TextIOSource, alphabet: None=None, title2ids: Optional[Callable[[str], Tuple[str, str, str]]]=None) -> Iterator[SeqRecord]:
    if False:
        print('Hello World!')
    'Parse Illumina 1.3 to 1.7 FASTQ like files (which differ in the quality mapping).\n\n    The optional arguments are the same as those for the FastqPhredIterator.\n\n    For each sequence in Illumina 1.3+ FASTQ files there is a matching string\n    encoding PHRED integer qualities using ASCII values with an offset of 64.\n\n    >>> from Bio import SeqIO\n    >>> record = SeqIO.read("Quality/illumina_faked.fastq", "fastq-illumina")\n    >>> print("%s %s" % (record.id, record.seq))\n    Test ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTN\n    >>> max(record.letter_annotations["phred_quality"])\n    40\n    >>> min(record.letter_annotations["phred_quality"])\n    0\n\n    NOTE - Older versions of the Solexa/Illumina pipeline encoded Solexa scores\n    with an ASCII offset of 64. They are approximately equal but only for high\n    quality reads. If you have an old Solexa/Illumina file with negative\n    Solexa scores, and try and read this as an Illumina 1.3+ file it will fail:\n\n    >>> record2 = SeqIO.read("Quality/solexa_faked.fastq", "fastq-illumina")\n    Traceback (most recent call last):\n       ...\n    ValueError: Invalid character in quality string\n\n    NOTE - True Sanger style FASTQ files use PHRED scores with an offset of 33.\n    '
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    q_mapping = {chr(letter): letter - SOLEXA_SCORE_OFFSET for letter in range(SOLEXA_SCORE_OFFSET, 63 + SOLEXA_SCORE_OFFSET)}
    for (title_line, seq_string, quality_string) in FastqGeneralIterator(source):
        if title2ids:
            (id, name, descr) = title2ids(title_line)
        else:
            descr = title_line
            id = descr.split()[0]
            name = id
        record = SeqRecord(Seq(seq_string), id=id, name=name, description=descr)
        try:
            qualities = [q_mapping[letter2] for letter2 in quality_string]
        except KeyError:
            raise ValueError('Invalid character in quality string') from None
        dict.__setitem__(record._per_letter_annotations, 'phred_quality', qualities)
        yield record

class QualPhredIterator(SequenceIterator):
    """Parser for QUAL files with PHRED quality scores but no sequence."""

    def __init__(self, source: _TextIOSource, alphabet: None=None, title2ids: Optional[Callable[[str], Tuple[str, str, str]]]=None) -> None:
        if False:
            while True:
                i = 10
        'For QUAL files which include PHRED quality scores, but no sequence.\n\n        For example, consider this short QUAL file::\n\n            >EAS54_6_R1_2_1_413_324\n            26 26 18 26 26 26 26 26 26 26 26 26 26 26 26 22 26 26 26 26\n            26 26 26 23 23\n            >EAS54_6_R1_2_1_540_792\n            26 26 26 26 26 26 26 26 26 26 26 22 26 26 26 26 26 12 26 26\n            26 18 26 23 18\n            >EAS54_6_R1_2_1_443_348\n            26 26 26 26 26 26 26 26 26 26 26 24 26 22 26 26 13 22 26 18\n            24 18 18 18 18\n\n        Using this module directly you might run:\n\n        >>> with open("Quality/example.qual") as handle:\n        ...     for record in QualPhredIterator(handle):\n        ...         print("%s read of length %d" % (record.id, len(record.seq)))\n        EAS54_6_R1_2_1_413_324 read of length 25\n        EAS54_6_R1_2_1_540_792 read of length 25\n        EAS54_6_R1_2_1_443_348 read of length 25\n\n        Typically however, you would call this via Bio.SeqIO instead with "qual"\n        as the format:\n\n        >>> from Bio import SeqIO\n        >>> with open("Quality/example.qual") as handle:\n        ...     for record in SeqIO.parse(handle, "qual"):\n        ...         print("%s read of length %d" % (record.id, len(record.seq)))\n        EAS54_6_R1_2_1_413_324 read of length 25\n        EAS54_6_R1_2_1_540_792 read of length 25\n        EAS54_6_R1_2_1_443_348 read of length 25\n\n        Only the sequence length is known, as the QUAL file does not contain\n        the sequence string itself.\n\n        The quality scores themselves are available as a list of integers\n        in each record\'s per-letter-annotation:\n\n        >>> print(record.letter_annotations["phred_quality"])\n        [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 22, 26, 26, 13, 22, 26, 18, 24, 18, 18, 18, 18]\n\n        You can still slice one of these SeqRecord objects:\n\n        >>> sub_record = record[5:10]\n        >>> print("%s %s" % (sub_record.id, sub_record.letter_annotations["phred_quality"]))\n        EAS54_6_R1_2_1_443_348 [26, 26, 26, 26, 26]\n\n        As of Biopython 1.59, this parser will accept files with negatives quality\n        scores but will replace them with the lowest possible PHRED score of zero.\n        This will trigger a warning, previously it raised a ValueError exception.\n        '
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        self.title2ids = title2ids
        super().__init__(source, mode='t', fmt='QUAL')

    def parse(self, handle: IO) -> Iterator[SeqRecord]:
        if False:
            return 10
        'Start parsing the file, and return a SeqRecord iterator.'
        records = self.iterate(handle)
        return records

    def iterate(self, handle: IO) -> Iterator[SeqRecord]:
        if False:
            for i in range(10):
                print('nop')
        'Parse the file and generate SeqRecord objects.'
        for line in handle:
            if line[0] == '>':
                break
        else:
            return
        while True:
            if line[0] != '>':
                raise ValueError("Records in Fasta files should start with '>' character")
            if self.title2ids:
                (id, name, descr) = self.title2ids(line[1:].rstrip())
            else:
                descr = line[1:].rstrip()
                id = descr.split()[0]
                name = id
            qualities: List[int] = []
            for line in handle:
                if line[0] == '>':
                    break
                qualities.extend((int(word) for word in line.split()))
            else:
                line = None
            if qualities and min(qualities) < 0:
                warnings.warn('Negative quality score %i found, substituting PHRED zero instead.' % min(qualities), BiopythonParserWarning)
                qualities = [max(0, q) for q in qualities]
            sequence = Seq(None, length=len(qualities))
            record = SeqRecord(sequence, id=id, name=name, description=descr)
            dict.__setitem__(record._per_letter_annotations, 'phred_quality', qualities)
            yield record
            if line is None:
                return
        raise ValueError('Unrecognised QUAL record format.')
assert SANGER_SCORE_OFFSET == ord('!')

class FastqPhredWriter(SequenceWriter):
    """Class to write standard FASTQ format files (using PHRED quality scores) (OBSOLETE).

    Although you can use this class directly, you are strongly encouraged
    to use the ``as_fastq`` function, or top level ``Bio.SeqIO.write()``
    function instead via the format name "fastq" or the alias "fastq-sanger".

    For example, this code reads in a standard Sanger style FASTQ file
    (using PHRED scores) and re-saves it as another Sanger style FASTQ file:

    >>> from Bio import SeqIO
    >>> record_iterator = SeqIO.parse("Quality/example.fastq", "fastq")
    >>> with open("Quality/temp.fastq", "w") as out_handle:
    ...     SeqIO.write(record_iterator, out_handle, "fastq")
    3

    You might want to do this if the original file included extra line breaks,
    which while valid may not be supported by all tools.  The output file from
    Biopython will have each sequence on a single line, and each quality
    string on a single line (which is considered desirable for maximum
    compatibility).

    In this next example, an old style Solexa/Illumina FASTQ file (using Solexa
    quality scores) is converted into a standard Sanger style FASTQ file using
    PHRED qualities:

    >>> from Bio import SeqIO
    >>> record_iterator = SeqIO.parse("Quality/solexa_example.fastq", "fastq-solexa")
    >>> with open("Quality/temp.fastq", "w") as out_handle:
    ...     SeqIO.write(record_iterator, out_handle, "fastq")
    5

    This code is also called if you use the .format("fastq") method of a
    SeqRecord, or .format("fastq-sanger") if you prefer that alias.

    Note that Sanger FASTQ files have an upper limit of PHRED quality 93, which is
    encoded as ASCII 126, the tilde. If your quality scores are truncated to fit, a
    warning is issued.

    P.S. To avoid cluttering up your working directory, you can delete this
    temporary file now:

    >>> import os
    >>> os.remove("Quality/temp.fastq")
    """

    def write_record(self, record: SeqRecord) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write a single FASTQ record to the file.'
        self._record_written = True
        seq = record.seq
        if seq is None:
            raise ValueError(f'No sequence for record {record.id}')
        qualities_str = _get_sanger_quality_str(record)
        if len(qualities_str) != len(seq):
            raise ValueError('Record %s has sequence length %i but %i quality scores' % (record.id, len(seq), len(qualities_str)))
        id_ = self.clean(record.id) if record.id else ''
        description = self.clean(record.description)
        if description and description.split(None, 1)[0] == id_:
            title = description
        elif description:
            title = f'{id_} {description}'
        else:
            title = id_
        self.handle.write(f'@{title}\n{seq}\n+\n{qualities_str}\n')

def as_fastq(record: SeqRecord) -> str:
    if False:
        i = 10
        return i + 15
    'Turn a SeqRecord into a Sanger FASTQ formatted string.\n\n    This is used internally by the SeqRecord\'s .format("fastq")\n    method and by the SeqIO.write(..., ..., "fastq") function,\n    and under the format alias "fastq-sanger" as well.\n    '
    seq_str = _get_seq_string(record)
    qualities_str = _get_sanger_quality_str(record)
    if len(qualities_str) != len(seq_str):
        raise ValueError('Record %s has sequence length %i but %i quality scores' % (record.id, len(seq_str), len(qualities_str)))
    id_ = _clean(record.id) if record.id else ''
    description = _clean(record.description)
    if description and description.split(None, 1)[0] == id_:
        title = description
    elif description:
        title = f'{id_} {description}'
    else:
        title = id_
    return f'@{title}\n{seq_str}\n+\n{qualities_str}\n'

class QualPhredWriter(SequenceWriter):
    """Class to write QUAL format files (using PHRED quality scores) (OBSOLETE).

    Although you can use this class directly, you are strongly encouraged
    to use the ``as_qual`` function, or top level ``Bio.SeqIO.write()``
    function instead.

    For example, this code reads in a FASTQ file and saves the quality scores
    into a QUAL file:

    >>> from Bio import SeqIO
    >>> record_iterator = SeqIO.parse("Quality/example.fastq", "fastq")
    >>> with open("Quality/temp.qual", "w") as out_handle:
    ...     SeqIO.write(record_iterator, out_handle, "qual")
    3

    This code is also called if you use the .format("qual") method of a
    SeqRecord.

    P.S. Don't forget to clean up the temp file if you don't need it anymore:

    >>> import os
    >>> os.remove("Quality/temp.qual")
    """

    def __init__(self, handle: _TextIOSource, wrap: int=60, record2title: Optional[Callable[[SeqRecord], str]]=None) -> None:
        if False:
            print('Hello World!')
        'Create a QUAL writer.\n\n        Arguments:\n         - handle - Handle to an output file, e.g. as returned\n           by open(filename, "w")\n         - wrap   - Optional line length used to wrap sequence lines.\n           Defaults to wrapping the sequence at 60 characters. Use\n           zero (or None) for no wrapping, giving a single long line\n           for the sequence.\n         - record2title - Optional function to return the text to be\n           used for the title line of each record.  By default a\n           combination of the record.id and record.description is\n           used.  If the record.description starts with the record.id,\n           then just the record.description is used.\n\n        The record2title argument is present for consistency with the\n        Bio.SeqIO.FastaIO writer class.\n        '
        super().__init__(handle)
        self.wrap: Optional[int] = None
        if wrap:
            if wrap < 1:
                raise ValueError
            self.wrap = wrap
        self.record2title = record2title

    def write_record(self, record: SeqRecord) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write a single QUAL record to the file.'
        self._record_written = True
        handle = self.handle
        wrap = self.wrap
        if self.record2title:
            title = self.clean(self.record2title(record))
        else:
            id_ = self.clean(record.id) if record.id else ''
            description = self.clean(record.description)
            if description and description.split(None, 1)[0] == id_:
                title = description
            elif description:
                title = f'{id} {description}'
            else:
                title = id_
        handle.write(f'>{title}\n')
        qualities = _get_phred_quality(record)
        try:
            qualities_strs = ['%i' % round(q, 0) for q in qualities]
        except TypeError:
            if None in qualities:
                raise TypeError('A quality value of None was found') from None
            else:
                raise
        if wrap is not None and wrap > 5:
            data = ' '.join(qualities_strs)
            while True:
                if len(data) <= wrap:
                    self.handle.write(data + '\n')
                    break
                else:
                    i = data.rfind(' ', 0, wrap)
                    handle.write(data[:i] + '\n')
                    data = data[i + 1:]
        elif wrap:
            while qualities_strs:
                line = qualities_strs.pop(0)
                while qualities_strs and len(line) + 1 + len(qualities_strs[0]) < wrap:
                    line += ' ' + qualities_strs.pop(0)
                handle.write(line + '\n')
        else:
            data = ' '.join(qualities_strs)
            handle.write(data + '\n')

def as_qual(record: SeqRecord) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Turn a SeqRecord into a QUAL formatted string.\n\n    This is used internally by the SeqRecord\'s .format("qual")\n    method and by the SeqIO.write(..., ..., "qual") function.\n    '
    id_ = _clean(record.id) if record.id else ''
    description = _clean(record.description)
    if description and description.split(None, 1)[0] == id_:
        title = description
    elif description:
        title = f'{id_} {description}'
    else:
        title = id_
    lines = [f'>{title}\n']
    qualities = _get_phred_quality(record)
    try:
        qualities_strs = ['%i' % round(q, 0) for q in qualities]
    except TypeError:
        if None in qualities:
            raise TypeError('A quality value of None was found') from None
        else:
            raise
    while qualities_strs:
        line = qualities_strs.pop(0)
        while qualities_strs and len(line) + 1 + len(qualities_strs[0]) < 60:
            line += ' ' + qualities_strs.pop(0)
        lines.append(line + '\n')
    return ''.join(lines)

class FastqSolexaWriter(SequenceWriter):
    """Write old style Solexa/Illumina FASTQ format files (with Solexa qualities) (OBSOLETE).

    This outputs FASTQ files like those from the early Solexa/Illumina
    pipeline, using Solexa scores and an ASCII offset of 64. These are
    NOT compatible with the standard Sanger style PHRED FASTQ files.

    If your records contain a "solexa_quality" entry under letter_annotations,
    this is used, otherwise any "phred_quality" entry will be used after
    conversion using the solexa_quality_from_phred function. If neither style
    of quality scores are present, an exception is raised.

    Although you can use this class directly, you are strongly encouraged
    to use the ``as_fastq_solexa`` function, or top-level ``Bio.SeqIO.write()``
    function instead.  For example, this code reads in a FASTQ file and re-saves
    it as another FASTQ file:

    >>> from Bio import SeqIO
    >>> record_iterator = SeqIO.parse("Quality/solexa_example.fastq", "fastq-solexa")
    >>> with open("Quality/temp.fastq", "w") as out_handle:
    ...     SeqIO.write(record_iterator, out_handle, "fastq-solexa")
    5

    You might want to do this if the original file included extra line breaks,
    which (while valid) may not be supported by all tools.  The output file
    from Biopython will have each sequence on a single line, and each quality
    string on a single line (which is considered desirable for maximum
    compatibility).

    This code is also called if you use the .format("fastq-solexa") method of
    a SeqRecord. For example,

    >>> record = SeqIO.read("Quality/sanger_faked.fastq", "fastq-sanger")
    >>> print(record.format("fastq-solexa"))
    @Test PHRED qualities from 40 to 0 inclusive
    ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTN
    +
    hgfedcba`_^]\\[ZYXWVUTSRQPONMLKJHGFECB@>;;
    <BLANKLINE>

    Note that Solexa FASTQ files have an upper limit of Solexa quality 62, which is
    encoded as ASCII 126, the tilde.  If your quality scores must be truncated to fit,
    a warning is issued.

    P.S. Don't forget to delete the temp file if you don't need it anymore:

    >>> import os
    >>> os.remove("Quality/temp.fastq")
    """

    def write_record(self, record: SeqRecord) -> None:
        if False:
            return 10
        'Write a single FASTQ record to the file.'
        self._record_written = True
        seq = record.seq
        if seq is None:
            raise ValueError(f'No sequence for record {record.id}')
        qualities_str = _get_solexa_quality_str(record)
        if len(qualities_str) != len(seq):
            raise ValueError('Record %s has sequence length %i but %i quality scores' % (record.id, len(seq), len(qualities_str)))
        id_ = self.clean(record.id) if record.id else ''
        description = self.clean(record.description)
        if description and description.split(None, 1)[0] == id_:
            title = description
        elif description:
            title = f'{id_} {description}'
        else:
            title = id_
        self.handle.write(f'@{title}\n{seq}\n+\n{qualities_str}\n')

def as_fastq_solexa(record: SeqRecord) -> str:
    if False:
        while True:
            i = 10
    'Turn a SeqRecord into a Solexa FASTQ formatted string.\n\n    This is used internally by the SeqRecord\'s .format("fastq-solexa")\n    method and by the SeqIO.write(..., ..., "fastq-solexa") function.\n    '
    seq_str = _get_seq_string(record)
    qualities_str = _get_solexa_quality_str(record)
    if len(qualities_str) != len(seq_str):
        raise ValueError('Record %s has sequence length %i but %i quality scores' % (record.id, len(seq_str), len(qualities_str)))
    id_ = _clean(record.id) if record.id else ''
    description = _clean(record.description)
    if description and description.split(None, 1)[0] == id_:
        title = description
    elif description:
        title = f'{id_} {description}'
    else:
        title = id_
    return f'@{title}\n{seq_str}\n+\n{qualities_str}\n'

class FastqIlluminaWriter(SequenceWriter):
    """Write Illumina 1.3+ FASTQ format files (with PHRED quality scores) (OBSOLETE).

    This outputs FASTQ files like those from the Solexa/Illumina 1.3+ pipeline,
    using PHRED scores and an ASCII offset of 64. Note these files are NOT
    compatible with the standard Sanger style PHRED FASTQ files which use an
    ASCII offset of 32.

    Although you can use this class directly, you are strongly encouraged to
    use the ``as_fastq_illumina`` or top-level ``Bio.SeqIO.write()`` function
    with format name "fastq-illumina" instead. This code is also called if you
    use the .format("fastq-illumina") method of a SeqRecord. For example,

    >>> from Bio import SeqIO
    >>> record = SeqIO.read("Quality/sanger_faked.fastq", "fastq-sanger")
    >>> print(record.format("fastq-illumina"))
    @Test PHRED qualities from 40 to 0 inclusive
    ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTN
    +
    hgfedcba`_^]\\[ZYXWVUTSRQPONMLKJIHGFEDCBA@
    <BLANKLINE>

    Note that Illumina FASTQ files have an upper limit of PHRED quality 62, which is
    encoded as ASCII 126, the tilde. If your quality scores are truncated to fit, a
    warning is issued.
    """

    def write_record(self, record: SeqRecord) -> None:
        if False:
            return 10
        'Write a single FASTQ record to the file.'
        self._record_written = True
        seq = record.seq
        if seq is None:
            raise ValueError(f'No sequence for record {record.id}')
        qualities_str = _get_illumina_quality_str(record)
        if len(qualities_str) != len(seq):
            raise ValueError('Record %s has sequence length %i but %i quality scores' % (record.id, len(seq), len(qualities_str)))
        id_ = self.clean(record.id) if record.id else ''
        description = self.clean(record.description)
        if description and description.split(None, 1)[0] == id_:
            title = description
        elif description:
            title = f'{id_} {description}'
        else:
            title = id_
        self.handle.write(f'@{title}\n{seq}\n+\n{qualities_str}\n')

def as_fastq_illumina(record: SeqRecord) -> str:
    if False:
        i = 10
        return i + 15
    'Turn a SeqRecord into an Illumina FASTQ formatted string.\n\n    This is used internally by the SeqRecord\'s .format("fastq-illumina")\n    method and by the SeqIO.write(..., ..., "fastq-illumina") function.\n    '
    seq_str = _get_seq_string(record)
    qualities_str = _get_illumina_quality_str(record)
    if len(qualities_str) != len(seq_str):
        raise ValueError('Record %s has sequence length %i but %i quality scores' % (record.id, len(seq_str), len(qualities_str)))
    id_ = _clean(record.id) if record.id else ''
    description = _clean(record.description)
    if description and description.split(None, 1)[0] == id_:
        title = description
    elif description:
        title = f'{id_} {description}'
    else:
        title = id_
    return f'@{title}\n{seq_str}\n+\n{qualities_str}\n'

def PairedFastaQualIterator(fasta_source: _TextIOSource, qual_source: _TextIOSource, alphabet: None=None, title2ids: Optional[Callable[[str], Tuple[str, str, str]]]=None) -> Iterator[SeqRecord]:
    if False:
        for i in range(10):
            print('nop')
    'Iterate over matched FASTA and QUAL files as SeqRecord objects.\n\n    For example, consider this short QUAL file with PHRED quality scores::\n\n        >EAS54_6_R1_2_1_413_324\n        26 26 18 26 26 26 26 26 26 26 26 26 26 26 26 22 26 26 26 26\n        26 26 26 23 23\n        >EAS54_6_R1_2_1_540_792\n        26 26 26 26 26 26 26 26 26 26 26 22 26 26 26 26 26 12 26 26\n        26 18 26 23 18\n        >EAS54_6_R1_2_1_443_348\n        26 26 26 26 26 26 26 26 26 26 26 24 26 22 26 26 13 22 26 18\n        24 18 18 18 18\n\n    And a matching FASTA file::\n\n        >EAS54_6_R1_2_1_413_324\n        CCCTTCTTGTCTTCAGCGTTTCTCC\n        >EAS54_6_R1_2_1_540_792\n        TTGGCAGGCCAAGGCCGATGGATCA\n        >EAS54_6_R1_2_1_443_348\n        GTTGCTTCTGGCGTGGGTGGGGGGG\n\n    You can parse these separately using Bio.SeqIO with the "qual" and\n    "fasta" formats, but then you\'ll get a group of SeqRecord objects with\n    no sequence, and a matching group with the sequence but not the\n    qualities.  Because it only deals with one input file handle, Bio.SeqIO\n    can\'t be used to read the two files together - but this function can!\n    For example,\n\n    >>> with open("Quality/example.fasta") as f:\n    ...     with open("Quality/example.qual") as q:\n    ...         for record in PairedFastaQualIterator(f, q):\n    ...             print("%s %s" % (record.id, record.seq))\n    ...\n    EAS54_6_R1_2_1_413_324 CCCTTCTTGTCTTCAGCGTTTCTCC\n    EAS54_6_R1_2_1_540_792 TTGGCAGGCCAAGGCCGATGGATCA\n    EAS54_6_R1_2_1_443_348 GTTGCTTCTGGCGTGGGTGGGGGGG\n\n    As with the FASTQ or QUAL parsers, if you want to look at the qualities,\n    they are in each record\'s per-letter-annotation dictionary as a simple\n    list of integers:\n\n    >>> print(record.letter_annotations["phred_quality"])\n    [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 24, 26, 22, 26, 26, 13, 22, 26, 18, 24, 18, 18, 18, 18]\n\n    If you have access to data as a FASTQ format file, using that directly\n    would be simpler and more straight forward.  Note that you can easily use\n    this function to convert paired FASTA and QUAL files into FASTQ files:\n\n    >>> from Bio import SeqIO\n    >>> with open("Quality/example.fasta") as f:\n    ...     with open("Quality/example.qual") as q:\n    ...         SeqIO.write(PairedFastaQualIterator(f, q), "Quality/temp.fastq", "fastq")\n    ...\n    3\n\n    And don\'t forget to clean up the temp file if you don\'t need it anymore:\n\n    >>> import os\n    >>> os.remove("Quality/temp.fastq")\n    '
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    from Bio.SeqIO.FastaIO import FastaIterator
    fasta_iter = FastaIterator(fasta_source, title2ids=title2ids)
    qual_iter = QualPhredIterator(qual_source, title2ids=title2ids)
    while True:
        try:
            f_rec = next(fasta_iter)
        except StopIteration:
            f_rec = None
        try:
            q_rec = next(qual_iter)
        except StopIteration:
            q_rec = None
        if f_rec is None and q_rec is None:
            break
        if f_rec is None:
            raise ValueError('FASTA file has more entries than the QUAL file.')
        if q_rec is None:
            raise ValueError('QUAL file has more entries than the FASTA file.')
        if f_rec.id != q_rec.id:
            raise ValueError(f'FASTA and QUAL entries do not match ({f_rec.id} vs {q_rec.id}).')
        if len(f_rec) != len(q_rec.letter_annotations['phred_quality']):
            raise ValueError(f'Sequence length and number of quality scores disagree for {f_rec.id}')
        f_rec.letter_annotations['phred_quality'] = q_rec.letter_annotations['phred_quality']
        yield f_rec

def _fastq_generic(in_file: _TextIOSource, out_file: _TextIOSource, mapping: Union[Sequence[str], Mapping[int, Optional[Union[str, int]]]]) -> int:
    if False:
        return 10
    "FASTQ helper function where can't have data loss by truncation (PRIVATE)."
    count = 0
    null = chr(0)
    with as_handle(out_file, 'w') as out_handle:
        for (title, seq, old_qual) in FastqGeneralIterator(in_file):
            count += 1
            qual = old_qual.translate(mapping)
            if null in qual:
                raise ValueError('Invalid character in quality string')
            out_handle.write(f'@{title}\n{seq}\n+\n{qual}\n')
    return count

def _fastq_generic2(in_file: _TextIOSource, out_file: _TextIOSource, mapping: Union[Sequence[str], Mapping[int, Optional[Union[str, int]]]], truncate_char: str, truncate_msg: str) -> int:
    if False:
        i = 10
        return i + 15
    'FASTQ helper function where there could be data loss by truncation (PRIVATE).'
    count = 0
    null = chr(0)
    with as_handle(out_file, 'w') as out_handle:
        for (title, seq, old_qual) in FastqGeneralIterator(in_file):
            count += 1
            qual = old_qual.translate(mapping)
            if null in qual:
                raise ValueError('Invalid character in quality string')
            if truncate_char in qual:
                qual = qual.replace(truncate_char, chr(126))
                warnings.warn(truncate_msg, BiopythonWarning)
            out_handle.write(f'@{title}\n{seq}\n+\n{qual}\n')
    return count

def _fastq_sanger_convert_fastq_sanger(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        i = 10
        return i + 15
    'Fast Sanger FASTQ to Sanger FASTQ conversion (PRIVATE).\n\n    Useful for removing line wrapping and the redundant second identifier\n    on the plus lines. Will check also check the quality string is valid.\n\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion.\n    '
    mapping = ''.join([chr(0) for ascii in range(33)] + [chr(ascii) for ascii in range(33, 127)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)

def _fastq_solexa_convert_fastq_solexa(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        i = 10
        return i + 15
    'Fast Solexa FASTQ to Solexa FASTQ conversion (PRIVATE).\n\n    Useful for removing line wrapping and the redundant second identifier\n    on the plus lines. Will check also check the quality string is valid.\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion.\n    '
    mapping = ''.join([chr(0) for ascii in range(59)] + [chr(ascii) for ascii in range(59, 127)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)

def _fastq_illumina_convert_fastq_illumina(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        return 10
    'Fast Illumina 1.3+ FASTQ to Illumina 1.3+ FASTQ conversion (PRIVATE).\n\n    Useful for removing line wrapping and the redundant second identifier\n    on the plus lines. Will check also check the quality string is valid.\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion.\n    '
    mapping = ''.join([chr(0) for ascii in range(64)] + [chr(ascii) for ascii in range(64, 127)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)

def _fastq_illumina_convert_fastq_sanger(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Fast Illumina 1.3+ FASTQ to Sanger FASTQ conversion (PRIVATE).\n\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion.\n    '
    mapping = ''.join([chr(0) for ascii in range(64)] + [chr(33 + q) for q in range(62 + 1)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)

def _fastq_sanger_convert_fastq_illumina(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        i = 10
        return i + 15
    'Fast Sanger FASTQ to Illumina 1.3+ FASTQ conversion (PRIVATE).\n\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion. Will issue a warning if the scores had to be truncated at 62\n    (maximum possible in the Illumina 1.3+ FASTQ format)\n    '
    trunc_char = chr(1)
    mapping = ''.join([chr(0) for ascii in range(33)] + [chr(64 + q) for q in range(62 + 1)] + [trunc_char for ascii in range(96, 127)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic2(in_file, out_file, mapping, trunc_char, 'Data loss - max PHRED quality 62 in Illumina 1.3+ FASTQ')

def _fastq_solexa_convert_fastq_sanger(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        return 10
    'Fast Solexa FASTQ to Sanger FASTQ conversion (PRIVATE).\n\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion.\n    '
    mapping = ''.join([chr(0) for ascii in range(59)] + [chr(33 + int(round(phred_quality_from_solexa(q)))) for q in range(-5, 62 + 1)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)

def _fastq_sanger_convert_fastq_solexa(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        i = 10
        return i + 15
    'Fast Sanger FASTQ to Solexa FASTQ conversion (PRIVATE).\n\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion. Will issue a warning if the scores had to be truncated at 62\n    (maximum possible in the Solexa FASTQ format)\n    '
    trunc_char = chr(1)
    mapping = ''.join([chr(0) for ascii in range(33)] + [chr(64 + int(round(solexa_quality_from_phred(q)))) for q in range(62 + 1)] + [trunc_char for ascii in range(96, 127)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic2(in_file, out_file, mapping, trunc_char, 'Data loss - max Solexa quality 62 in Solexa FASTQ')

def _fastq_solexa_convert_fastq_illumina(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Fast Solexa FASTQ to Illumina 1.3+ FASTQ conversion (PRIVATE).\n\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion.\n    '
    mapping = ''.join([chr(0) for ascii in range(59)] + [chr(64 + int(round(phred_quality_from_solexa(q)))) for q in range(-5, 62 + 1)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)

def _fastq_illumina_convert_fastq_solexa(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Fast Illumina 1.3+ FASTQ to Solexa FASTQ conversion (PRIVATE).\n\n    Avoids creating SeqRecord and Seq objects in order to speed up this\n    conversion.\n    '
    mapping = ''.join([chr(0) for ascii in range(64)] + [chr(64 + int(round(solexa_quality_from_phred(q)))) for q in range(62 + 1)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic(in_file, out_file, mapping)

def _fastq_convert_fasta(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Fast FASTQ to FASTA conversion (PRIVATE).\n\n    Avoids dealing with the FASTQ quality encoding, and creating SeqRecord and\n    Seq objects in order to speed up this conversion.\n\n    NOTE - This does NOT check the characters used in the FASTQ quality string\n    are valid!\n    '
    count = 0
    with as_handle(out_file, 'w') as out_handle:
        for (title, seq, qual) in FastqGeneralIterator(in_file):
            count += 1
            out_handle.write(f'>{title}\n')
            for i in range(0, len(seq), 60):
                out_handle.write(seq[i:i + 60] + '\n')
    return count

def _fastq_convert_tab(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        print('Hello World!')
    'Fast FASTQ to simple tabbed conversion (PRIVATE).\n\n    Avoids dealing with the FASTQ quality encoding, and creating SeqRecord and\n    Seq objects in order to speed up this conversion.\n\n    NOTE - This does NOT check the characters used in the FASTQ quality string\n    are valid!\n    '
    count = 0
    with as_handle(out_file, 'w') as out_handle:
        for (title, seq, qual) in FastqGeneralIterator(in_file):
            count += 1
            out_handle.write(f'{title.split(None, 1)[0]}\t{seq}\n')
    return count

def _fastq_convert_qual(in_file: _TextIOSource, out_file: _TextIOSource, mapping: Mapping[str, str]) -> int:
    if False:
        return 10
    'FASTQ helper function for QUAL output (PRIVATE).\n\n    Mapping should be a dictionary mapping expected ASCII characters from the\n    FASTQ quality string to PHRED quality scores (as strings).\n    '
    count = 0
    with as_handle(out_file, 'w') as out_handle:
        for (title, seq, qual) in FastqGeneralIterator(in_file):
            count += 1
            out_handle.write(f'>{title}\n')
            try:
                qualities_strs = [mapping[ascii_] for ascii_ in qual]
            except KeyError:
                raise ValueError('Invalid character in quality string') from None
            data = ' '.join(qualities_strs)
            while len(data) > 60:
                if data[60] == ' ':
                    out_handle.write(data[:60] + '\n')
                    data = data[61:]
                elif data[59] == ' ':
                    out_handle.write(data[:59] + '\n')
                    data = data[60:]
                else:
                    assert data[58] == ' ', 'Internal logic failure in wrapping'
                    out_handle.write(data[:58] + '\n')
                    data = data[59:]
            out_handle.write(data + '\n')
    return count

def _fastq_sanger_convert_qual(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        i = 10
        return i + 15
    'Fast Sanger FASTQ to QUAL conversion (PRIVATE).'
    mapping = {chr(q + 33): str(q) for q in range(93 + 1)}
    return _fastq_convert_qual(in_file, out_file, mapping)

def _fastq_solexa_convert_qual(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        while True:
            i = 10
    'Fast Solexa FASTQ to QUAL conversion (PRIVATE).'
    mapping = {chr(q + 64): str(int(round(phred_quality_from_solexa(q)))) for q in range(-5, 62 + 1)}
    return _fastq_convert_qual(in_file, out_file, mapping)

def _fastq_illumina_convert_qual(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    if False:
        i = 10
        return i + 15
    'Fast Illumina 1.3+ FASTQ to QUAL conversion (PRIVATE).'
    mapping = {chr(q + 64): str(q) for q in range(62 + 1)}
    return _fastq_convert_qual(in_file, out_file, mapping)
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)