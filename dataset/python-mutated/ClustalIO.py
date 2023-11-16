"""Bio.AlignIO support for "clustal" output from CLUSTAL W and other tools.

You are expected to use this module via the Bio.AlignIO functions (or the
Bio.SeqIO functions if you want to work directly with the gapped sequences).
"""
from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.Interfaces import AlignmentIterator
from Bio.AlignIO.Interfaces import SequentialAlignmentWriter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

class ClustalWriter(SequentialAlignmentWriter):
    """Clustalw alignment writer."""

    def write_alignment(self, alignment):
        if False:
            for i in range(10):
                print('nop')
        'Use this to write (another) single alignment to an open file.'
        if len(alignment) == 0:
            raise ValueError('Must have at least one sequence')
        if alignment.get_alignment_length() == 0:
            raise ValueError('Non-empty sequences are required')
        try:
            version = str(alignment._version)
        except AttributeError:
            version = ''
        if not version:
            version = '1.81'
        if version.startswith('2.'):
            output = f'CLUSTAL {version} multiple sequence alignment\n\n\n'
        else:
            output = f'CLUSTAL X ({version}) multiple sequence alignment\n\n\n'
        cur_char = 0
        max_length = len(alignment[0])
        if max_length <= 0:
            raise ValueError('Non-empty sequences are required')
        if 'clustal_consensus' in alignment.column_annotations:
            star_info = alignment.column_annotations['clustal_consensus']
        else:
            try:
                star_info = alignment._star_info
            except AttributeError:
                star_info = None
        while cur_char != max_length:
            if cur_char + 50 > max_length:
                show_num = max_length - cur_char
            else:
                show_num = 50
            for record in alignment:
                line = record.id[0:30].replace(' ', '_').ljust(36)
                line += str(record.seq[cur_char:cur_char + show_num])
                output += line + '\n'
            if star_info:
                output += ' ' * 36 + star_info[cur_char:cur_char + show_num] + '\n'
            output += '\n'
            cur_char += show_num
        self.handle.write(output + '\n')

class ClustalIterator(AlignmentIterator):
    """Clustalw alignment iterator."""
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
        known_headers = ['CLUSTAL', 'PROBCONS', 'MUSCLE', 'MSAPROBS', 'Kalign', 'Biopython']
        if line.strip().split()[0] not in known_headers:
            raise ValueError('%s is not a known CLUSTAL header: %s' % (line.strip().split()[0], ', '.join(known_headers)))
        version = None
        for word in line.split():
            if word[0] == '(' and word[-1] == ')':
                word = word[1:-1]
            if word[0] in '0123456789':
                version = word
                break
        line = handle.readline()
        while line.strip() == '':
            line = handle.readline()
        ids = []
        seqs = []
        consensus = ''
        seq_cols = None
        while True:
            if line[0] != ' ' and line.strip() != '':
                fields = line.rstrip().split()
                if len(fields) < 2 or len(fields) > 3:
                    raise ValueError(f'Could not parse line:\n{line}')
                ids.append(fields[0])
                seqs.append(fields[1])
                if seq_cols is None:
                    start = len(fields[0]) + line[len(fields[0]):].find(fields[1])
                    end = start + len(fields[1])
                    seq_cols = slice(start, end)
                    del start, end
                assert fields[1] == line[seq_cols]
                if len(fields) == 3:
                    try:
                        letters = int(fields[2])
                    except ValueError:
                        raise ValueError(f'Could not parse line, bad sequence number:\n{line}') from None
                    if len(fields[1].replace('-', '')) != letters:
                        raise ValueError(f'Could not parse line, invalid sequence number:\n{line}')
            elif line[0] == ' ':
                assert len(ids) == len(seqs)
                assert len(ids) > 0
                assert seq_cols is not None
                consensus = line[seq_cols]
                assert not line[:seq_cols.start].strip()
                assert not line[seq_cols.stop:].strip()
                line = handle.readline()
                assert line.strip() == ''
                break
            else:
                break
            line = handle.readline()
            if not line:
                break
        assert line.strip() == ''
        assert seq_cols is not None
        for s in seqs:
            assert len(s) == len(seqs[0])
        if consensus:
            assert len(consensus) == len(seqs[0])
        done = False
        while not done:
            while not line or line.strip() == '':
                line = handle.readline()
                if not line:
                    break
            if not line:
                break
            if line.split(None, 1)[0] in known_headers:
                self._header = line
                break
            for i in range(len(ids)):
                if line[0] == ' ':
                    raise ValueError(f'Unexpected line:\n{line!r}')
                fields = line.rstrip().split()
                if len(fields) < 2 or len(fields) > 3:
                    raise ValueError(f'Could not parse line:\n{line!r}')
                if fields[0] != ids[i]:
                    raise ValueError("Identifiers out of order? Got '%s' but expected '%s'" % (fields[0], ids[i]))
                if fields[1] != line[seq_cols]:
                    start = len(fields[0]) + line[len(fields[0]):].find(fields[1])
                    if start != seq_cols.start:
                        raise ValueError('Old location %s -> %i:XX' % (seq_cols, start))
                    end = start + len(fields[1])
                    seq_cols = slice(start, end)
                    del start, end
                seqs[i] += fields[1]
                assert len(seqs[i]) == len(seqs[0])
                if len(fields) == 3:
                    try:
                        letters = int(fields[2])
                    except ValueError:
                        raise ValueError(f'Could not parse line, bad sequence number:\n{line}') from None
                    if len(seqs[i].replace('-', '')) != letters:
                        raise ValueError(f'Could not parse line, invalid sequence number:\n{line}')
                line = handle.readline()
            if consensus:
                assert line[0] == ' '
                assert seq_cols is not None
                consensus += line[seq_cols]
                assert len(consensus) == len(seqs[0])
                assert not line[:seq_cols.start].strip()
                assert not line[seq_cols.stop:].strip()
                line = handle.readline()
        assert len(ids) == len(seqs)
        if len(seqs) == 0 or len(seqs[0]) == 0:
            raise StopIteration
        if self.records_per_alignment is not None and self.records_per_alignment != len(ids):
            raise ValueError('Found %i records in this alignment, told to expect %i' % (len(ids), self.records_per_alignment))
        records = (SeqRecord(Seq(s), id=i, description=i) for (i, s) in zip(ids, seqs))
        alignment = MultipleSeqAlignment(records)
        if version:
            alignment._version = version
        if consensus:
            alignment_length = len(seqs[0])
            if len(consensus) != alignment_length:
                raise ValueError("Alignment length is %i, consensus length is %i, '%s'" % (alignment_length, len(consensus), consensus))
            alignment.column_annotations['clustal_consensus'] = consensus
            alignment._star_info = consensus
        return alignment