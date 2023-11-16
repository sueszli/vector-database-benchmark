"""Bio.AlignIO support for GCG MSF format.

The file format was produced by the GCG PileUp and and LocalPileUp tools,
and later tools such as T-COFFEE and MUSCLE support it as an optional
output format.

The original GCG tool would write gaps at ends of each sequence which could
be missing data as tildes (``~``), whereas internal gaps were periods (``.``)
instead. This parser replaces both with minus signs (``-``) for consistency
with the rest of ``Bio.AlignIO``.

You are expected to use this module via the Bio.AlignIO functions (or the
Bio.SeqIO functions if you want to work directly with the gapped sequences).
"""
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator

class MsfIterator(AlignmentIterator):
    """GCG MSF alignment iterator."""
    _header = None

    def __next__(self):
        if False:
            print('Hello World!')
        'Parse the next alignment from the handle.'
        handle = self.handle
        if self._header is None:
            line = handle.readline()
        else:
            line = self._header
            self._header = None
        if not line:
            raise StopIteration
        known_headers = ['!!NA_MULTIPLE_ALIGNMENT', '!!AA_MULTIPLE_ALIGNMENT', 'PileUp']
        if line.strip().split()[0] not in known_headers:
            raise ValueError('%s is not a known GCG MSF header: %s' % (line.strip().split()[0], ', '.join(known_headers)))
        while line and ' MSF: ' not in line:
            line = handle.readline()
        if not line:
            raise ValueError('Reached end of file without MSF/Type/Check header line')
        parts = line.strip('\n').split()
        offset = parts.index('MSF:')
        if parts[offset + 2] != 'Type:' or parts[-3] not in ('Check:', 'CompCheck:') or parts[-1] != '..':
            raise ValueError("GCG MSF header line should be '<optional text> MSF: <int> Type: <letter> <optional date> Check: <int> ..',  not: %r" % line)
        try:
            aln_length = int(parts[offset + 1])
        except ValueError:
            aln_length = -1
        if aln_length < 0:
            raise ValueError('GCG MSF header line should have MDF: <int> for column count, not %r' % parts[offset + 1])
        seq_type = parts[offset + 3]
        if seq_type not in ['P', 'N']:
            raise ValueError("GCG MSF header line should have 'Type: P' (protein) or 'Type: N' (nucleotide), not 'Type: %s'" % seq_type)
        ids = []
        lengths = []
        checks = []
        weights = []
        line = handle.readline()
        while line and line.strip() != '//':
            line = handle.readline()
            if line.strip().startswith('Name: '):
                if ' Len: ' in line and ' Check: ' in line and (' Weight: ' in line):
                    rest = line[line.index('Name: ') + 6:].strip()
                    (name, rest) = rest.split(' Len: ')
                    (length, rest) = rest.split(' Check: ')
                    (check, weight) = rest.split(' Weight: ')
                    name = name.strip()
                    if name.endswith(' oo'):
                        name = name[:-3]
                    if name in ids:
                        raise ValueError(f'Duplicated ID of {name!r}')
                    if ' ' in name:
                        raise NotImplementedError(f'Space in ID {name!r}')
                    ids.append(name)
                    lengths.append(int(length.strip()))
                    checks.append(int(check.strip()))
                    weights.append(float(weight.strip()))
                else:
                    raise ValueError(f'Malformed GCG MSF name line: {line!r}')
        if not line:
            raise ValueError('End of file while looking for end of header // line.')
        if aln_length != max(lengths):
            max_length = max(lengths)
            max_count = sum((1 for _ in lengths if _ == max_length))
            raise ValueError('GCG MSF header said alignment length %i, but %s of %i sequences said Len: %s' % (aln_length, max_count, len(ids), max_length))
        line = handle.readline()
        if not line:
            raise ValueError('End of file after // line, expected sequences.')
        if line.strip():
            raise ValueError('After // line, expected blank line before sequences.')
        seqs = [[] for _ in ids]
        completed_length = 0
        while completed_length < aln_length:
            for (idx, name) in enumerate(ids):
                line = handle.readline()
                if idx == 0 and (not line.strip()):
                    while line and (not line.strip()):
                        line = handle.readline()
                if not line:
                    raise ValueError('End of file where expecting sequence data.')
                words = line.strip().split()
                if idx == 0 and words and (words[0] != name):
                    try:
                        i = int(words[0])
                    except ValueError:
                        i = -1
                    if i != completed_length + 1:
                        raise ValueError('Expected GCG MSF coordinate line starting %i, got: %r' % (completed_length + 1, line))
                    if len(words) > 1:
                        if len(words) != 2:
                            i = -1
                        else:
                            try:
                                i = int(words[1])
                            except ValueError:
                                i = -1
                        if i != (completed_length + 50 if completed_length + 50 < aln_length else aln_length):
                            raise ValueError('Expected GCG MSF coordinate line %i to %i, got: %r' % (completed_length + 1, completed_length + 50 if completed_length + 50 < aln_length else aln_length, line))
                    line = handle.readline()
                    words = line.strip().split()
                if not words:
                    if lengths[idx] < aln_length and len(''.join(seqs[idx])) == lengths[idx]:
                        pass
                    else:
                        raise ValueError(f'Expected sequence for {name}, got: {line!r}')
                elif words[0] == name:
                    assert len(words) > 1, line
                    seqs[idx].extend(words[1:])
                else:
                    raise ValueError(f'Expected sequence for {name!r}, got: {line!r}')
            completed_length += 50
            line = handle.readline()
            if line.strip():
                raise ValueError(f'Expected blank line, got: {line!r}')
        while True:
            line = handle.readline()
            if not line:
                break
            elif not line.strip():
                pass
            elif line.strip().split()[0] in known_headers:
                self._header = line
                break
            else:
                raise ValueError(f'Unexpected line after GCG MSF alignment: {line!r}')
        seqs = [''.join(s).replace('~', '-').replace('.', '-') for s in seqs]
        padded = False
        for (idx, (length, s)) in enumerate(zip(lengths, seqs)):
            if len(s) < aln_length and len(s) == length:
                padded = True
                seqs[idx] = s + '-' * (aln_length - len(s))
        if padded:
            import warnings
            from Bio import BiopythonParserWarning
            warnings.warn('One of more alignment sequences were truncated and have been gap padded', BiopythonParserWarning)
        records = (SeqRecord(Seq(s), id=i, name=i, description=i, annotations={'weight': w}) for (i, s, w) in zip(ids, seqs, weights))
        align = MultipleSeqAlignment(records)
        if align.get_alignment_length() != aln_length:
            raise ValueError('GCG MSF headers said alignment length %i, but have %i' % (aln_length, align.get_alignment_length()))
        return align