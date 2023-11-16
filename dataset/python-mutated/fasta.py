"""Bio.Align support for aligned FASTA files.

Aligned FASTA files are FASTA files in which alignment gaps in a sequence are
represented by dashes. Each sequence line in an aligned FASTA should have the
same length.
"""
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

class AlignmentWriter(interfaces.AlignmentWriter):
    """Alignment file writer for the aligned FASTA file format."""
    fmt = 'FASTA'

    def format_alignment(self, alignment):
        if False:
            i = 10
            return i + 15
        'Return a string with the alignment in aligned FASTA format.'
        if not isinstance(alignment, Alignment):
            raise TypeError('Expected an Alignment object')
        lines = []
        for (sequence, line) in zip(alignment.sequences, alignment):
            try:
                name = sequence.id
            except AttributeError:
                lines.append('>')
            else:
                if sequence.description:
                    lines.append(f'>{sequence.id} {sequence.description}')
                else:
                    lines.append(f'>{sequence.id}')
            lines.append(line)
        return '\n'.join(lines) + '\n'

class AlignmentIterator(interfaces.AlignmentIterator):
    """Alignment iterator for aligned FASTA files.

    An aligned FASTA file contains one multiple alignment. Alignment gaps are
    represented by dashes in the sequence lines. Header lines start with '>'
    followed by the name of the sequence, and optionally a description.
    """
    fmt = 'FASTA'

    def _read_next_alignment(self, stream):
        if False:
            i = 10
            return i + 15
        names = []
        descriptions = []
        lines = []
        for line in stream:
            if line.startswith('>'):
                parts = line[1:].rstrip().split(None, 1)
                if len(parts) == 2:
                    (name, description) = parts
                else:
                    description = ''
                    if len(parts) == 1:
                        name = parts[0]
                    else:
                        name = ''
                names.append(name)
                descriptions.append(description)
                lines.append('')
            else:
                lines[-1] += line.strip()
        if not lines:
            if self._stream.tell() == 0:
                raise ValueError('Empty file.')
            return
        coordinates = Alignment.infer_coordinates(lines)
        records = []
        for (name, description, line) in zip(names, descriptions, lines):
            line = line.replace('-', '')
            sequence = Seq(line)
            record = SeqRecord(sequence, id=name, description=description)
            records.append(record)
        alignment = Alignment(records, coordinates)
        return alignment