"""Bio.Align support for GCG MSF format.

The file format was produced by the GCG PileUp and LocalPileUp tools, and later
tools such as T-COFFEE and MUSCLE support it as an optional output format.

You are expected to use this module via the Bio.Align functions.
"""
import warnings
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning

class AlignmentIterator(interfaces.AlignmentIterator):
    """GCG MSF alignment iterator."""
    fmt = 'MSF'

    def _read_next_alignment(self, stream):
        if False:
            i = 10
            return i + 15
        try:
            line = next(stream)
        except StopIteration:
            if stream.tell() == 0:
                raise ValueError('Empty file.') from None
            return
        known_headers = ['!!NA_MULTIPLE_ALIGNMENT', '!!AA_MULTIPLE_ALIGNMENT', 'PileUp']
        if line.strip().split()[0] not in known_headers:
            raise ValueError('%s is not a known GCG MSF header: %s' % (line.strip().split()[0], ', '.join(known_headers)))
        for line in stream:
            line = line.rstrip('\n')
            if 'MSF: ' in line and line.endswith('..'):
                break
        else:
            raise ValueError('Reached end of file without MSF/Type/Check header line')
        parts = line.split()
        offset = parts.index('MSF:')
        if parts[offset + 2] != 'Type:' or parts[-3] not in ('Check:', 'CompCheck:'):
            raise ValueError("GCG MSF header line should be '<optional text> MSF: <int> Type: <letter> <optional date> Check: <int> ..',  not: %r" % line)
        try:
            aln_length = int(parts[offset + 1])
        except ValueError:
            raise ValueError('GCG MSF header line should have MSF: <int> for column count, not %r' % parts[offset + 1]) from None
        seq_type = parts[offset + 3]
        if seq_type not in ['P', 'N']:
            raise ValueError("GCG MSF header line should have 'Type: P' (protein) or 'Type: N' (nucleotide), not 'Type: %s'" % seq_type)
        names = []
        remaining = []
        checks = []
        weights = []
        for line in stream:
            line = line.strip()
            if line == '//':
                break
            if line.startswith('Name: '):
                words = line.split()
                try:
                    index_name = words.index('Name:')
                    index_len = words.index('Len:')
                    index_weight = words.index('Weight:')
                    index_check = words.index('Check:')
                except ValueError:
                    raise ValueError(f'Malformed GCG MSF name line: {line!r}') from None
                name = words[index_name + 1]
                length = int(words[index_len + 1])
                weight = float(words[index_weight + 1])
                check = words[index_check + 1]
                if name in names:
                    raise ValueError(f'Duplicated ID of {name!r}')
                names.append(name)
                remaining.append(length)
                checks.append(check)
                weights.append(weight)
        else:
            raise ValueError('End of file while looking for end of header // line.')
        try:
            line = next(stream)
        except StopIteration:
            raise ValueError('End of file after // line, expected sequences.') from None
        if line.strip():
            raise ValueError('After // line, expected blank line before sequences.')
        seqs = [''] * len(names)
        for line in stream:
            words = line.split()
            if not words:
                continue
            name = words[0]
            try:
                index = names.index(name)
            except ValueError:
                for word in words:
                    if not word.isdigit():
                        break
                else:
                    continue
                raise ValueError(f"Unexpected line '{line}' in input") from None
            seq = ''.join(words[1:])
            length = remaining[index] - (len(seq) - seq.count('-'))
            if length < 0:
                raise ValueError('Received longer sequence than expected for %s' % name)
            seqs[index] += seq
            remaining[index] = length
            if all((length == 0 for length in remaining)):
                break
        else:
            raise ValueError('End of file where expecting sequence data.')
        for line in stream:
            assert line.strip() == ''
        length = max((len(seq) for seq in seqs))
        if length != aln_length:
            warnings.warn('GCG MSF headers said alignment length %i, but found %i' % (aln_length, length), BiopythonParserWarning, stacklevel=2)
            aln_length = length
        for (index, seq) in enumerate(seqs):
            seq = ''.join(seq).replace('~', '-').replace('.', '-')
            if len(seq) < aln_length:
                seq += '-' * (aln_length - len(seq))
            seqs[index] = seq
        coordinates = Alignment.infer_coordinates(seqs)
        seqs = (Seq(seq.replace('-', '')) for seq in seqs)
        records = [SeqRecord(seq, id=name, name=name, description=name, annotations={'weight': weight}) for (name, seq, weight) in zip(names, seqs, weights)]
        alignment = Alignment(records, coordinates)
        columns = alignment.length
        if columns != aln_length:
            raise ValueError('GCG MSF headers said alignment length %i, but found %i' % (aln_length, columns))
        return alignment