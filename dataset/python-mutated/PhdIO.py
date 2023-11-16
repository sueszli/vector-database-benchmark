"""Bio.SeqIO support for the "phd" file format.

PHD files are output by PHRED and used by PHRAP and CONSED.

You are expected to use this module via the Bio.SeqIO functions, under the
format name "phd". See also the underlying Bio.Sequencing.Phd module.

For example, using Bio.SeqIO we can read in one of the example PHRED files
from the Biopython unit tests:

    >>> from Bio import SeqIO
    >>> for record in SeqIO.parse("Phd/phd1", "phd"):
    ...     print(record.id)
    ...     print("%s..." % record.seq[:10])
    ...     print("%s..." % record.letter_annotations["phred_quality"][:10])
    34_222_(80-A03-19).b.ab1
    ctccgtcgga...
    [9, 9, 10, 19, 22, 37, 28, 28, 24, 22]...
    425_103_(81-A03-19).g.ab1
    cgggatccca...
    [14, 17, 22, 10, 10, 10, 15, 8, 8, 9]...
    425_7_(71-A03-19).b.ab1
    acataaatca...
    [10, 10, 10, 10, 8, 8, 6, 6, 6, 6]...

Since PHRED files contain quality scores, you can save them as FASTQ or as
QUAL files, for example using Bio.SeqIO.write(...), or simply with the format
method of the SeqRecord object:

    >>> print(record[:50].format("fastq"))
    @425_7_(71-A03-19).b.ab1
    acataaatcaaattactnaccaacacacaaaccngtctcgcgtagtggag
    +
    ++++))'''')(''')$!$''')''''(+.''$!$))))+)))'''''''
    <BLANKLINE>

Or,

    >>> print(record[:50].format("qual"))
    >425_7_(71-A03-19).b.ab1
    10 10 10 10 8 8 6 6 6 6 8 7 6 6 6 8 3 0 3 6 6 6 8 6 6 6 6 7
    10 13 6 6 3 0 3 8 8 8 8 10 8 8 8 6 6 6 6 6 6 6
    <BLANKLINE>

Note these examples only show the first 50 bases to keep the output short.
"""
from typing import Iterator
from Bio.SeqRecord import SeqRecord
from Bio.Sequencing import Phd
from .QualityIO import _get_phred_quality
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from .Interfaces import _IOSource

def PhdIterator(source: _TextIOSource) -> Iterator[SeqRecord]:
    if False:
        for i in range(10):
            print('nop')
    'Return SeqRecord objects from a PHD file.\n\n    Arguments:\n     - source - input stream opened in text mode, or a path to a file\n\n    This uses the Bio.Sequencing.Phd module to do the hard work.\n    '
    phd_records = Phd.parse(source)
    for phd_record in phd_records:
        name = phd_record.file_name.split(None, 1)[0]
        seq_record = SeqRecord(phd_record.seq, id=name, name=name, description=phd_record.file_name)
        seq_record.annotations = phd_record.comments
        seq_record.annotations['molecule_type'] = 'DNA'
        seq_record.letter_annotations['phred_quality'] = [int(site[1]) for site in phd_record.sites]
        try:
            seq_record.letter_annotations['peak_location'] = [int(site[2]) for site in phd_record.sites]
        except IndexError:
            pass
        yield seq_record

class PhdWriter(SequenceWriter):
    """Class to write Phd format files."""

    def __init__(self, handle: _IOSource) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        super().__init__(handle)

    def write_record(self, record):
        if False:
            print('Hello World!')
        'Write a single Phd record to the file.'
        assert record.seq, 'No sequence present in SeqRecord'
        phred_qualities = _get_phred_quality(record)
        peak_locations = record.letter_annotations.get('peak_location')
        if len(record.seq) != len(phred_qualities):
            raise ValueError('Number of phd quality scores does not match length of sequence')
        if peak_locations:
            if len(record.seq) != len(peak_locations):
                raise ValueError('Number of peak location scores does not match length of sequence')
        if None in phred_qualities:
            raise ValueError('A quality value of None was found')
        if record.description.startswith(f'{record.id} '):
            title = record.description
        else:
            title = f'{record.id} {record.description}'
        self.handle.write(f'BEGIN_SEQUENCE {self.clean(title)}\nBEGIN_COMMENT\n')
        for annot in [k.lower() for k in Phd.CKEYWORDS]:
            value = None
            if annot == 'trim':
                if record.annotations.get('trim'):
                    value = '%s %s %.4f' % record.annotations['trim']
            elif annot == 'trace_peak_area_ratio':
                if record.annotations.get('trace_peak_area_ratio'):
                    value = f"{record.annotations['trace_peak_area_ratio']:.4f}"
            else:
                value = record.annotations.get(annot)
            if value or value == 0:
                self.handle.write(f'{annot.upper()}: {value}\n')
        self.handle.write('END_COMMENT\nBEGIN_DNA\n')
        for (i, site) in enumerate(record.seq):
            if peak_locations:
                self.handle.write('%s %i %i\n' % (site, round(phred_qualities[i]), peak_locations[i]))
            else:
                self.handle.write('%s %i\n' % (site, round(phred_qualities[i])))
        self.handle.write('END_DNA\nEND_SEQUENCE\n')
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()