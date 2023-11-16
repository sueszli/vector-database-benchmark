"""Bio.SeqIO support for the "ig" (IntelliGenetics or MASE) file format.

This module is for reading and writing IntelliGenetics format files as
SeqRecord objects.  This file format appears to be the same as the MASE
multiple sequence alignment format.

You are expected to use this module via the Bio.SeqIO functions.
"""
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator

class IgIterator(SequenceIterator):
    """Parser for IntelliGenetics files."""

    def __init__(self, source):
        if False:
            i = 10
            return i + 15
        'Iterate over IntelliGenetics records (as SeqRecord objects).\n\n        source - file-like object opened in text mode, or a path to a file\n\n        The optional free format file header lines (which start with two\n        semi-colons) are ignored.\n\n        The free format commentary lines at the start of each record (which\n        start with a semi-colon) are recorded as a single string with embedded\n        new line characters in the SeqRecord\'s annotations dictionary under the\n        key \'comment\'.\n\n        Examples\n        --------\n        >>> with open("IntelliGenetics/TAT_mase_nuc.txt") as handle:\n        ...     for record in IgIterator(handle):\n        ...         print("%s length %i" % (record.id, len(record)))\n        ...\n        A_U455 length 303\n        B_HXB2R length 306\n        C_UG268A length 267\n        D_ELI length 309\n        F_BZ163A length 309\n        O_ANT70 length 342\n        O_MVP5180 length 348\n        CPZGAB length 309\n        CPZANT length 309\n        A_ROD length 390\n        B_EHOA length 420\n        D_MM251 length 390\n        STM_STM length 387\n        VER_AGM3 length 354\n        GRI_AGM677 length 264\n        SAB_SAB1C length 219\n        SYK_SYK length 330\n\n        '
        super().__init__(source, mode='t', fmt='IntelliGenetics')

    def parse(self, handle):
        if False:
            i = 10
            return i + 15
        'Start parsing the file, and return a SeqRecord generator.'
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        if False:
            print('Hello World!')
        'Iterate over the records in the IntelliGenetics file.'
        for line in handle:
            if not line.startswith(';;'):
                break
        else:
            return
        if line[0] != ';':
            raise ValueError(f"Records should start with ';' and not:\n{line!r}")
        while line:
            comment_lines = []
            while line.startswith(';'):
                comment_lines.append(line[1:].strip())
                line = next(handle)
            title = line.rstrip()
            seq_lines = []
            for line in handle:
                if line[0] == ';':
                    break
                seq_lines.append(line.rstrip().replace(' ', ''))
            else:
                line = None
            seq_str = ''.join(seq_lines)
            if seq_str.endswith('1'):
                seq_str = seq_str[:-1]
            if '1' in seq_str:
                raise ValueError('Potential terminator digit one found within sequence.')
            yield SeqRecord(Seq(seq_str), id=title, name=title, annotations={'comment': '\n'.join(comment_lines)})
        assert not line
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)