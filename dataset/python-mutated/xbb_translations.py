"""Translation code for graphical Xbbtools tool."""
import time
from Bio.Seq import Seq, reverse_complement, translate
from Bio.SeqUtils import gc_fraction

class xbb_translations:
    """A class for doing translations."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initialize the class.'

    def frame1(self, seq, translation_table=1):
        if False:
            print('Hello World!')
        'Translate first reading frame.'
        return translate(seq, table=translation_table)

    def complement(self, seq):
        if False:
            return 10
        'Return complementary DNA Seq object.'
        return Seq.complement(Seq(seq))

    def reverse(self, seq):
        if False:
            print('Hello World!')
        'Reverse the sequence.'
        return seq[::-1]

    def antiparallel(self, seq):
        if False:
            for i in range(10):
                print('nop')
        'Return reverse complementary sequence.'
        return reverse_complement(seq)

    def frame(self, seq, frame, translation_table=1):
        if False:
            for i in range(10):
                print('nop')
        'Translate DNA sequence in a chosen frame.'
        if frame < 0:
            seq = reverse_complement(seq)
        seq = seq[abs(frame) - 1:]
        return translate(seq, table=translation_table)

    def header_nice(self, txt, seq):
        if False:
            while True:
                i = 10
        'Print a short header for the translation window.'
        length = len(seq)
        if length > 20:
            short = f'{seq[:10]} ... {seq[-10:]}'
        else:
            short = seq
        date = time.strftime('%y %b %d, %X', time.localtime(time.time()))
        res = f'{txt}: {date}, '
        for nt in ['a', 't', 'g', 'c']:
            res += f'{nt}:{seq.count(nt.upper()):d} '
        res += f'\nSequence: {short.lower()}, {length:d} nt, %0.2f %{self.gc(seq):G}C\n'
        res += '\n\n'
        return res

    def frame_nice(self, seq, frame, translation_table=1):
        if False:
            return 10
        'Print a pretty print single frame translation.'
        length = len(seq)
        protein = self.frame(seq, frame, translation_table)
        protein_length = len(protein)
        protein = '  '.join(list(protein))
        protein += ((length - (abs(frame) - 1)) % 3 + 2) * ' '
        if frame < 0:
            protein = protein[::-1]
        res = self.header_nice(f'Frame {frame} translation', seq)
        for i in range(0, length, 60):
            subseq = seq[i:i + 60]
            p = i // 3
            if frame > 0:
                res += '%d/%d\n' % (i + 1, p + 1)
                res += ' ' * (frame - 1) + protein[i:i + 60] + '\n'
                res += subseq.lower() + '%5d %%\n' % int(self.gc(subseq)) + '\n'
            else:
                res += '%d/%d\n' % (i + 1, protein_length - len(protein[:i].split()))
                res += subseq.lower() + '%5d %%\n' % int(self.gc(subseq))
                res += protein[i:i + 60] + '\n\n'
        return res

    def gc(self, seq):
        if False:
            for i in range(10):
                print('nop')
        'Calculate GC content in percent (0-100).'
        return 100 * gc_fraction(seq)

    def gcframe(self, seq, translation_table=1, direction='both'):
        if False:
            while True:
                i = 10
        'Print a pretty print translation in several frames.'
        comp = self.complement(seq)
        anti = self.reverse(comp)
        length = len(seq)
        frames = {}
        for i in range(3):
            frames[i + 1] = self.frame1(seq[i:], translation_table)
            frames[-(i + 1)] = self.reverse(self.frame1(anti[i:], translation_table))
        res = self.header_nice('GCFrame', seq)
        for i in range(0, length, 60):
            subseq = seq[i:i + 60]
            csubseq = comp[i:i + 60]
            p = i // 3
            if direction in ('plus', 'both'):
                res += '%d/%d\n' % (i + 1, i // 3 + 1)
                res += '  ' + '  '.join(frames[3][p:p + 20]) + '\n'
                res += ' ' + '  '.join(frames[2][p:p + 20]) + '\n'
                res += '  '.join(frames[1][p:p + 20]) + '\n'
            res += subseq.lower() + '%5d %%\n' % int(self.gc(subseq))
            res += csubseq.lower() + '\n'
            if direction == 'plus':
                res += '\n'
            if direction in ('minus', 'both'):
                res += '  '.join(frames[-2][p:p + 20]) + ' \n'
                res += ' ' + '  '.join(frames[-1][p:p + 20]) + '\n'
                res += '  ' + '  '.join(frames[-3][p:p + 20]) + '\n\n'
        return res
if __name__ == '__main__':
    seq = 'ATTCCGGTTGATCCTGCCGGACCCGACCGCTATCGGGGTAGGGATAAGCCATGGGAGTCTTACACTCCCGGGTAAGGGAGTGTGGCGGACGGCTGAGTAACACGTGGCTAACCTACCCTCGGGACGGGGATAACCCCGGGAAACTGGGGATAATCCCCGATAGGGAAGGAGTCCTGGAATGGTTCCTTCCCTAAAGGGCTATAGGCTATTTCCCGTTTGTAGCCGCCCGAGGATGGGGCTACGGCCCATCAGGCTGTCGGTGGGGTAAAGGCCCACCGAACCTATAACGGGTAGGGGCCGTGGAAGCGGGAGCCTCCAGTTGGGCACTGAGACAAGGGCCCAGGCCCTACGGGGCGCACCAGGCGCGAAACGTCCCCAATGCGCGAAAGCGTGAGGGCGCTACCCCGAGTGCCTCCGCAAGGAGGCTTTTCCCCGCTCTAAAAAGGCGGGGGAATAAGCGGGGGGCAAGTCTGGTGTCAGCCGCCGCGGTAATACCAGCTCCGCGAGTGGTCGGGGTGATTACTGGGCCTAAAGCGCCTGTAGCCGGCCCACCAAGTCGCCCCTTAAAGTCCCCGGCTCAACCGGGGAACTGGGGGCGATACTGGTGGGCTAGGGGGCGGGAGAGGCGGGGGGTACTCCCGGAGTAGGGGCGAAATCCTTAGATACCGGGAGGACCACCAGTGGCGGAAGCGCCCCGCTA'
    test = xbb_translations()
    print('============================================================')
    print(test.gcframe(seq))