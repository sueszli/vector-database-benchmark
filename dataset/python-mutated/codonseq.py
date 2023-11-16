"""Code for dealing with coding sequence.

CodonSeq class is inherited from Seq class. This is the core class to
deal with sequences in CodonAlignment in biopython.

"""
from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable

class CodonSeq(Seq):
    """CodonSeq is designed to be within the SeqRecords of a CodonAlignment class.

    CodonSeq is useful as it allows the user to specify
    reading frame when translate CodonSeq

    CodonSeq also accepts codon style slice by calling
    get_codon() method.

    **Important:** Ungapped CodonSeq can be any length if you
    specify the rf_table. Gapped CodonSeq should be a
    multiple of three.

    >>> codonseq = CodonSeq("AAATTTGGGCCAAATTT", rf_table=(0,3,6,8,11,14))
    >>> print(codonseq.translate())
    KFGAKF

    test get_full_rf_table method

    >>> p = CodonSeq('AAATTTCCCGG-TGGGTTTAA', rf_table=(0, 3, 6, 9, 11, 14, 17))
    >>> full_rf_table = p.get_full_rf_table()
    >>> print(full_rf_table)
    [0, 3, 6, 9, 12, 15, 18]
    >>> print(p.translate(rf_table=full_rf_table, ungap_seq=False))
    KFPPWV*
    >>> p = CodonSeq('AAATTTCCCGGGAA-TTTTAA', rf_table=(0, 3, 6, 9, 14, 17))
    >>> print(p.get_full_rf_table())
    [0, 3, 6, 9, 12.0, 15, 18]
    >>> p = CodonSeq('AAA------------TAA', rf_table=(0, 3))
    >>> print(p.get_full_rf_table())
    [0, 3.0, 6.0, 9.0, 12.0, 15]

    """

    def __init__(self, data='', gap_char='-', rf_table=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        Seq.__init__(self, data.upper())
        self.gap_char = gap_char
        if rf_table is None:
            length = len(self)
            if length % 3 != 0:
                raise ValueError('Sequence length is not a multiple of three (i.e. a whole number of codons)')
            self.rf_table = list(range(0, length - self.count(gap_char), 3))
        else:
            if not isinstance(rf_table, (tuple, list)):
                raise TypeError('rf_table should be a tuple or list object')
            if not all((isinstance(i, int) for i in rf_table)):
                raise TypeError('Elements in rf_table should be int that specify the codon positions of the sequence')
            self.rf_table = rf_table

    def get_codon(self, index):
        if False:
            while True:
                i = 10
        'Get the index codon from the sequence.'
        if len({i % 3 for i in self.rf_table}) != 1:
            raise RuntimeError('frameshift detected. CodonSeq object is not able to deal with codon sequence with frameshift. Please use normal slice option.')
        if isinstance(index, int):
            if index != -1:
                return str(self[index * 3:(index + 1) * 3])
            else:
                return str(self[index * 3:])
        else:
            aa_index = range(len(self) // 3)

            def cslice(p):
                if False:
                    print('Hello World!')
                aa_slice = aa_index[p]
                codon_slice = ''
                for i in aa_slice:
                    codon_slice += self[i * 3:i * 3 + 3]
                return str(codon_slice)
            codon_slice = cslice(index)
            return CodonSeq(codon_slice)

    def get_codon_num(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of codons in the CodonSeq.'
        return len(self.rf_table)

    def translate(self, codon_table=None, stop_symbol='*', rf_table=None, ungap_seq=True):
        if False:
            for i in range(10):
                print('nop')
        'Translate the CodonSeq based on the reading frame in rf_table.\n\n        It is possible for the user to specify\n        a rf_table at this point. If you want to include\n        gaps in the translated sequence, this is the only\n        way. ungap_seq should be set to true for this\n        purpose.\n        '
        if codon_table is None:
            codon_table = CodonTable.generic_by_id[1]
        amino_acids = []
        if ungap_seq:
            tr_seq = str(self).replace(self.gap_char, '')
        else:
            tr_seq = str(self)
        if rf_table is None:
            rf_table = self.rf_table
        p = -1
        for i in rf_table:
            if isinstance(i, float):
                amino_acids.append('-')
                continue
            elif '-' in tr_seq[i:i + 3]:
                if p == -1 or p - i == 3:
                    p = i
                    codon = tr_seq[i:i + 6].replace('-', '')[:3]
                elif p - i > 3:
                    codon = tr_seq[i:i + 3]
                    p = i
            else:
                codon = tr_seq[i:i + 3]
                p = i
            if codon in codon_table.stop_codons:
                amino_acids.append(stop_symbol)
                continue
            try:
                amino_acids.append(codon_table.forward_table[codon])
            except KeyError:
                raise RuntimeError(f'Unknown codon detected ({codon}). Did you forget to specify the ungap_seq argument?')
        return ''.join(amino_acids)

    def toSeq(self):
        if False:
            print('Hello World!')
        'Convert DNA to seq object.'
        return Seq(str(self))

    def get_full_rf_table(self):
        if False:
            for i in range(10):
                print('nop')
        'Return full rf_table of the CodonSeq records.\n\n        A full rf_table is different from a normal rf_table in that\n        it translate gaps in CodonSeq. It is helpful to construct\n        alignment containing frameshift.\n        '
        ungap_seq = str(self).replace('-', '')
        relative_pos = [self.rf_table[0]]
        for i in range(1, len(self.rf_table[1:]) + 1):
            relative_pos.append(self.rf_table[i] - self.rf_table[i - 1])
        full_rf_table = []
        codon_num = 0
        for i in range(0, len(self), 3):
            if self[i:i + 3] == self.gap_char * 3:
                full_rf_table.append(i + 0.0)
            elif relative_pos[codon_num] == 0:
                full_rf_table.append(i)
                codon_num += 1
            elif relative_pos[codon_num] in (-1, -2):
                gap_stat = 3 - self.count('-', i - 3, i)
                if gap_stat == 3:
                    full_rf_table.append(i + relative_pos[codon_num])
                elif gap_stat == 2:
                    full_rf_table.append(i + 1 + relative_pos[codon_num])
                elif gap_stat == 1:
                    full_rf_table.append(i + 2 + relative_pos[codon_num])
                codon_num += 1
            elif relative_pos[codon_num] > 0:
                full_rf_table.append(i + 0.0)
            try:
                this_len = 3 - self.count('-', i, i + 3)
                relative_pos[codon_num] -= this_len
            except Exception:
                pass
        return full_rf_table

    def full_translate(self, codon_table=None, stop_symbol='*'):
        if False:
            while True:
                i = 10
        'Apply full translation with gaps considered.'
        if codon_table is None:
            codon_table = CodonTable.generic_by_id[1]
        full_rf_table = self.get_full_rf_table()
        return self.translate(codon_table=codon_table, stop_symbol=stop_symbol, rf_table=full_rf_table, ungap_seq=False)

    def ungap(self, gap='-'):
        if False:
            i = 10
            return i + 15
        'Return a copy of the sequence without the gap character(s).'
        if len(gap) != 1 or not isinstance(gap, str):
            raise ValueError(f'Unexpected gap character, {gap!r}')
        return CodonSeq(str(self).replace(gap, ''), rf_table=self.rf_table)

    @classmethod
    def from_seq(cls, seq, rf_table=None):
        if False:
            return 10
        'Get codon sequence from sequence data.'
        if rf_table is None:
            return cls(str(seq))
        else:
            return cls(str(seq), rf_table=rf_table)

def _get_codon_list(codonseq):
    if False:
        i = 10
        return i + 15
    'List of codons according to full_rf_table for counting (PRIVATE).'
    full_rf_table = codonseq.get_full_rf_table()
    codon_lst = []
    for (i, k) in enumerate(full_rf_table):
        if isinstance(k, int):
            start = k
            try:
                end = int(full_rf_table[i + 1])
            except IndexError:
                end = start + 3
            this_codon = str(codonseq[start:end])
            if len(this_codon) == 3:
                codon_lst.append(this_codon)
            else:
                codon_lst.append(str(this_codon.ungap()))
        elif str(codonseq[int(k):int(k) + 3]) == '---':
            codon_lst.append('---')
        else:
            codon_lst.append(codonseq[int(k):int(k) + 3])
    return codon_lst

def cal_dn_ds(codon_seq1, codon_seq2, method='NG86', codon_table=None, k=1, cfreq=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculate dN and dS of the given two sequences.\n\n    Available methods:\n        - NG86  - `Nei and Gojobori (1986)`_ (PMID 3444411).\n        - LWL85 - `Li et al. (1985)`_ (PMID 3916709).\n        - ML    - `Goldman and Yang (1994)`_ (PMID 7968486).\n        - YN00  - `Yang and Nielsen (2000)`_ (PMID 10666704).\n\n    .. _`Nei and Gojobori (1986)`: http://www.ncbi.nlm.nih.gov/pubmed/3444411\n    .. _`Li et al. (1985)`: http://www.ncbi.nlm.nih.gov/pubmed/3916709\n    .. _`Goldman and Yang (1994)`: http://mbe.oxfordjournals.org/content/11/5/725\n    .. _`Yang and Nielsen (2000)`: https://doi.org/10.1093/oxfordjournals.molbev.a026236\n\n    Arguments:\n     - codon_seq1 - CodonSeq or or SeqRecord that contains a CodonSeq\n     - codon_seq2 - CodonSeq or or SeqRecord that contains a CodonSeq\n     - w  - transition/transversion ratio\n     - cfreq - Current codon frequency vector can only be specified\n       when you are using ML method. Possible ways of\n       getting cfreq are: F1x4, F3x4 and F61.\n\n    '
    if isinstance(codon_seq1, CodonSeq) and isinstance(codon_seq2, CodonSeq):
        pass
    elif isinstance(codon_seq1, SeqRecord) and isinstance(codon_seq2, SeqRecord):
        codon_seq1 = codon_seq1.seq
        codon_seq2 = codon_seq2.seq
    else:
        raise TypeError('cal_dn_ds accepts two CodonSeq objects or SeqRecord that contains CodonSeq as its seq!')
    if len(codon_seq1.get_full_rf_table()) != len(codon_seq2.get_full_rf_table()):
        raise RuntimeError(f'full_rf_table length of seq1 ({len(codon_seq1.get_full_rf_table())}) and seq2 ({len(codon_seq2.get_full_rf_table())}) are not the same')
    if cfreq is None:
        cfreq = 'F3x4'
    elif cfreq is not None and method != 'ML':
        raise RuntimeError('cfreq can only be specified when you are using ML method')
    if cfreq not in ('F1x4', 'F3x4', 'F61'):
        import warnings
        warnings.warn(f'Unknown cfreq ({cfreq}). Only F1x4, F3x4 and F61 are acceptable. Used F3x4 in the following.')
        cfreq = 'F3x4'
    if codon_table is None:
        codon_table = CodonTable.generic_by_id[1]
    seq1_codon_lst = _get_codon_list(codon_seq1)
    seq2_codon_lst = _get_codon_list(codon_seq2)
    seq1 = []
    seq2 = []
    for (i, j) in zip(seq1_codon_lst, seq2_codon_lst):
        if '-' not in i and '-' not in j:
            seq1.append(i)
            seq2.append(j)
    dnds_func = {'ML': _ml, 'NG86': _ng86, 'LWL85': _lwl85, 'YN00': _yn00}
    if method == 'ML':
        return dnds_func[method](seq1, seq2, cfreq, codon_table)
    else:
        return dnds_func[method](seq1, seq2, k, codon_table)

def _ng86(seq1, seq2, k, codon_table):
    if False:
        for i in range(10):
            print('nop')
    'NG86 method main function (PRIVATE).'
    (S_sites1, N_sites1) = _count_site_NG86(seq1, codon_table=codon_table, k=k)
    (S_sites2, N_sites2) = _count_site_NG86(seq2, codon_table=codon_table, k=k)
    S_sites = (S_sites1 + S_sites2) / 2.0
    N_sites = (N_sites1 + N_sites2) / 2.0
    SN = [0, 0]
    for (i, j) in zip(seq1, seq2):
        SN = [m + n for (m, n) in zip(SN, _count_diff_NG86(i, j, codon_table=codon_table))]
    ps = SN[0] / S_sites
    pn = SN[1] / N_sites
    if ps < 3 / 4:
        dS = abs(-3.0 / 4 * log(1 - 4.0 / 3 * ps))
    else:
        dS = -1
    if pn < 3 / 4:
        dN = abs(-3.0 / 4 * log(1 - 4.0 / 3 * pn))
    else:
        dN = -1
    return (dN, dS)

def _count_site_NG86(codon_lst, codon_table, k=1):
    if False:
        while True:
            i = 10
    'Count synonymous and non-synonymous sites of a list of codons (PRIVATE).\n\n    Arguments:\n     - codon_lst - A three letter codon list from a CodonSeq object.\n       This can be returned from _get_codon_list method.\n     - k - transition/transversion rate ratio.\n\n    '
    S_site = 0
    N_site = 0
    purine = ('A', 'G')
    pyrimidine = ('T', 'C')
    base_tuple = ('A', 'T', 'C', 'G')
    for codon in codon_lst:
        neighbor_codon = {'transition': [], 'transversion': []}
        codon = codon.replace('U', 'T')
        if codon == '---':
            continue
        for (n, i) in enumerate(codon):
            for j in base_tuple:
                if i == j:
                    pass
                elif i in purine and j in purine:
                    codon_chars = list(codon)
                    codon_chars[n] = j
                    this_codon = ''.join(codon_chars)
                    neighbor_codon['transition'].append(this_codon)
                elif i in pyrimidine and j in pyrimidine:
                    codon_chars = list(codon)
                    codon_chars[n] = j
                    this_codon = ''.join(codon_chars)
                    neighbor_codon['transition'].append(this_codon)
                else:
                    codon_chars = list(codon)
                    codon_chars[n] = j
                    this_codon = ''.join(codon_chars)
                    neighbor_codon['transversion'].append(this_codon)
        aa = codon_table.forward_table[codon]
        this_codon_N_site = this_codon_S_site = 0
        for neighbor in neighbor_codon['transition']:
            if neighbor in codon_table.stop_codons:
                this_codon_N_site += 1
            elif codon_table.forward_table[neighbor] == aa:
                this_codon_S_site += 1
            else:
                this_codon_N_site += 1
        for neighbor in neighbor_codon['transversion']:
            if neighbor in codon_table.stop_codons:
                this_codon_N_site += k
            elif codon_table.forward_table[neighbor] == aa:
                this_codon_S_site += k
            else:
                this_codon_N_site += k
        norm_const = (this_codon_N_site + this_codon_S_site) / 3
        S_site += this_codon_S_site / norm_const
        N_site += this_codon_N_site / norm_const
    return (S_site, N_site)

def _count_diff_NG86(codon1, codon2, codon_table):
    if False:
        i = 10
        return i + 15
    'Count differences between two codons, three-letter string (PRIVATE).\n\n    The function will take multiple pathways from codon1 to codon2\n    into account.\n    '
    if not isinstance(codon1, str) or not isinstance(codon2, str):
        raise TypeError(f'_count_diff_NG86 accepts string object to represent codon ({type(codon1)}, {type(codon2)} detected)')
    if len(codon1) != 3 or len(codon2) != 3:
        raise RuntimeError(f'codon should be three letter string ({len(codon1)}, {len(codon2)} detected)')
    SN = [0, 0]
    if codon1 == '---' or codon2 == '---':
        return SN
    base_tuple = ('A', 'C', 'G', 'T')
    if not all((i in base_tuple for i in codon1)):
        raise RuntimeError(f'Unrecognized character detected in codon1 {codon1} (Codons consist of A, T, C or G)')
    if not all((i in base_tuple for i in codon2)):
        raise RuntimeError(f'Unrecognized character detected in codon2 {codon2} (Codons consist of A, T, C or G)')
    if codon1 == codon2:
        return SN
    else:
        diff_pos = []
        for (i, k) in enumerate(zip(codon1, codon2)):
            if k[0] != k[1]:
                diff_pos.append(i)

        def compare_codon(codon1, codon2, codon_table, weight=1):
            if False:
                for i in range(10):
                    print('nop')
            'Compare two codon accounting for different pathways.'
            sd = nd = 0
            if len(set(map(codon_table.forward_table.get, [codon1, codon2]))) == 1:
                sd += weight
            else:
                nd += weight
            return (sd, nd)
        if len(diff_pos) == 1:
            SN = [i + j for (i, j) in zip(SN, compare_codon(codon1, codon2, codon_table=codon_table))]
        elif len(diff_pos) == 2:
            for i in diff_pos:
                temp_codon = codon1[:i] + codon2[i] + codon1[i + 1:]
                SN = [i + j for (i, j) in zip(SN, compare_codon(codon1, temp_codon, codon_table=codon_table, weight=0.5))]
                SN = [i + j for (i, j) in zip(SN, compare_codon(temp_codon, codon2, codon_table=codon_table, weight=0.5))]
        elif len(diff_pos) == 3:
            paths = list(permutations([0, 1, 2], 3))
            tmp_codon = []
            for p in paths:
                tmp1 = codon1[:p[0]] + codon2[p[0]] + codon1[p[0] + 1:]
                tmp2 = tmp1[:p[1]] + codon2[p[1]] + tmp1[p[1] + 1:]
                tmp_codon.append((tmp1, tmp2))
                SN = [i + j for (i, j) in zip(SN, compare_codon(codon1, tmp1, codon_table, weight=0.5 / 3))]
                SN = [i + j for (i, j) in zip(SN, compare_codon(tmp1, tmp2, codon_table, weight=0.5 / 3))]
                SN = [i + j for (i, j) in zip(SN, compare_codon(tmp2, codon2, codon_table, weight=0.5 / 3))]
    return SN

def _lwl85(seq1, seq2, k, codon_table):
    if False:
        i = 10
        return i + 15
    'LWL85 method main function (PRIVATE).\n\n    Nomenclature is according to Li et al. (1985), PMID 3916709.\n    '
    codon_fold_dict = _get_codon_fold(codon_table)
    fold0 = [0, 0]
    fold2 = [0, 0]
    fold4 = [0, 0]
    for codon in seq1 + seq2:
        fold_num = codon_fold_dict[codon]
        for f in fold_num:
            if f == '0':
                fold0[0] += 1
            elif f == '2':
                fold2[0] += 1
            elif f == '4':
                fold4[0] += 1
    L = [sum(fold0) / 2.0, sum(fold2) / 2.0, sum(fold4) / 2.0]
    PQ = [0] * 6
    for (codon1, codon2) in zip(seq1, seq2):
        if (codon1 == '---' or codon2 == '---') or codon1 == codon2:
            continue
        else:
            PQ = [i + j for (i, j) in zip(PQ, _diff_codon(codon1, codon2, fold_dict=codon_fold_dict))]
    PQ = [i / j for (i, j) in zip(PQ, L * 2)]
    P = PQ[:3]
    Q = PQ[3:]
    A = [1.0 / 2 * log(1.0 / (1 - 2 * i - j)) - 1.0 / 4 * log(1.0 / (1 - 2 * j)) for (i, j) in zip(P, Q)]
    B = [1.0 / 2 * log(1.0 / (1 - 2 * i)) for i in Q]
    dS = 3 * (L[2] * A[1] + L[2] * (A[2] + B[2])) / (L[1] + 3 * L[2])
    dN = 3 * (L[2] * B[1] + L[0] * (A[0] + B[0])) / (2 * L[1] + 3 * L[0])
    return (dN, dS)

def _get_codon_fold(codon_table):
    if False:
        print('Hello World!')
    'Classify different position in a codon into different folds (PRIVATE).'

    def find_fold_class(codon, forward_table):
        if False:
            print('Hello World!')
        base = {'A', 'T', 'C', 'G'}
        fold = ''
        codon_base_lst = list(codon)
        for (i, b) in enumerate(codon_base_lst):
            other_base = base - set(b)
            aa = []
            for j in other_base:
                codon_base_lst[i] = j
                try:
                    aa.append(forward_table[''.join(codon_base_lst)])
                except KeyError:
                    aa.append('stop')
            if aa.count(forward_table[codon]) == 0:
                fold += '0'
            elif aa.count(forward_table[codon]) in (1, 2):
                fold += '2'
            elif aa.count(forward_table[codon]) == 3:
                fold += '4'
            else:
                raise RuntimeError('Unknown Error, cannot assign the position to a fold')
            codon_base_lst[i] = b
        return fold
    fold_table = {}
    for codon in codon_table.forward_table:
        if 'U' not in codon:
            fold_table[codon] = find_fold_class(codon, codon_table.forward_table)
    fold_table['---'] = '---'
    return fold_table

def _diff_codon(codon1, codon2, fold_dict):
    if False:
        while True:
            i = 10
    'Count number of different substitution types between two codons (PRIVATE).\n\n    returns tuple (P0, P2, P4, Q0, Q2, Q4)\n\n    Nomenclature is according to Li et al. (1958), PMID 3916709.\n    '
    P0 = P2 = P4 = Q0 = Q2 = Q4 = 0
    fold_num = fold_dict[codon1]
    purine = ('A', 'G')
    pyrimidine = ('T', 'C')
    for (n, (i, j)) in enumerate(zip(codon1, codon2)):
        if i != j and (i in purine and j in purine):
            if fold_num[n] == '0':
                P0 += 1
            elif fold_num[n] == '2':
                P2 += 1
            elif fold_num[n] == '4':
                P4 += 1
            else:
                raise RuntimeError('Unexpected fold_num %d' % fold_num[n])
        if i != j and (i in pyrimidine and j in pyrimidine):
            if fold_num[n] == '0':
                P0 += 1
            elif fold_num[n] == '2':
                P2 += 1
            elif fold_num[n] == '4':
                P4 += 1
            else:
                raise RuntimeError('Unexpected fold_num %d' % fold_num[n])
        if i != j and (i in purine and j in pyrimidine or (i in pyrimidine and j in purine)):
            if fold_num[n] == '0':
                Q0 += 1
            elif fold_num[n] == '2':
                Q2 += 1
            elif fold_num[n] == '4':
                Q4 += 1
            else:
                raise RuntimeError('Unexpected fold_num %d' % fold_num[n])
    return (P0, P2, P4, Q0, Q2, Q4)

def _yn00(seq1, seq2, k, codon_table):
    if False:
        while True:
            i = 10
    'YN00 method main function (PRIVATE).\n\n    Nomenclature is according to Yang and Nielsen (2000), PMID 10666704.\n    '
    from collections import defaultdict
    from scipy.linalg import expm
    fcodon = [{'A': 0, 'G': 0, 'C': 0, 'T': 0}, {'A': 0, 'G': 0, 'C': 0, 'T': 0}, {'A': 0, 'G': 0, 'C': 0, 'T': 0}]
    codon_fold_dict = _get_codon_fold(codon_table)
    fold0_cnt = defaultdict(int)
    fold4_cnt = defaultdict(int)
    for codon in seq1 + seq2:
        if codon != '---':
            fcodon[0][codon[0]] += 1
            fcodon[1][codon[1]] += 1
            fcodon[2][codon[2]] += 1
        fold_num = codon_fold_dict[codon]
        for (i, f) in enumerate(fold_num):
            if f == '0':
                fold0_cnt[codon[i]] += 1
            elif f == '4':
                fold4_cnt[codon[i]] += 1
    f0_total = sum(fold0_cnt.values())
    f4_total = sum(fold4_cnt.values())
    for (i, j) in zip(fold0_cnt, fold4_cnt):
        fold0_cnt[i] = fold0_cnt[i] / f0_total
        fold4_cnt[i] = fold4_cnt[i] / f4_total
    TV = _get_TV(seq1, seq2, codon_table=codon_table)
    k04 = (_get_kappa_t(fold0_cnt, TV), _get_kappa_t(fold4_cnt, TV))
    kappa = (f0_total * k04[0] + f4_total * k04[1]) / (f0_total + f4_total)
    for i in range(3):
        tot = sum(fcodon[i].values())
        fcodon[i] = {j: k / tot for (j, k) in fcodon[i].items()}
    pi = defaultdict(int)
    for i in list(codon_table.forward_table.keys()) + codon_table.stop_codons:
        if 'U' not in i:
            pi[i] = 0
    for i in seq1 + seq2:
        pi[i] += 1
    (S_sites1, N_sites1, bfreqSN1) = _count_site_YN00(seq1, seq2, pi, k=kappa, codon_table=codon_table)
    (S_sites2, N_sites2, bfreqSN2) = _count_site_YN00(seq2, seq1, pi, k=kappa, codon_table=codon_table)
    N_sites = (N_sites1 + N_sites2) / 2
    S_sites = (S_sites1 + S_sites2) / 2
    bfreqSN = [{'A': 0, 'T': 0, 'C': 0, 'G': 0}, {'A': 0, 'T': 0, 'C': 0, 'G': 0}]
    for i in range(2):
        for b in ('A', 'T', 'C', 'G'):
            bfreqSN[i][b] = (bfreqSN1[i][b] + bfreqSN2[i][b]) / 2
    SN = [0, 0]
    for (i, j) in zip(seq1, seq2):
        SN = [m + n for (m, n) in zip(SN, _count_diff_NG86(i, j, codon_table=codon_table))]
    ps = SN[0] / S_sites
    pn = SN[1] / N_sites
    p = sum(SN) / (S_sites + N_sites)
    w = log(1 - 4.0 / 3 * pn) / log(1 - 4.0 / 3 * ps)
    t = -3 / 4 * log(1 - 4 / 3 * p)
    tolerance = 1e-05
    dSdN_pre = [0, 0]
    for temp in range(20):
        codon_lst = [i for i in list(codon_table.forward_table.keys()) + codon_table.stop_codons if 'U' not in i]
        Q = _get_Q(pi, kappa, w, codon_lst, codon_table)
        P = expm(Q * t)
        TV = [0, 0, 0, 0]
        codon_npath = {}
        for (i, j) in zip(seq1, seq2):
            if i != '---' and j != '---':
                codon_npath.setdefault((i, j), 0)
                codon_npath[i, j] += 1
        for i in codon_npath:
            tv = _count_diff_YN00(i[0], i[1], P, codon_lst, codon_table)
            TV = [m + n * codon_npath[i] for (m, n) in zip(TV, tv)]
        TV = ((TV[0] / S_sites, TV[1] / S_sites), (TV[2] / N_sites, TV[3] / N_sites))
        dSdN = []
        for (f, tv) in zip(bfreqSN, TV):
            dSdN.append(_get_kappa_t(f, tv, t=True))
        t = dSdN[0] * 3 * S_sites / (S_sites + N_sites) + dSdN[1] * 3 * N_sites / (S_sites + N_sites)
        w = dSdN[1] / dSdN[0]
        if all((abs(i - j) < tolerance for (i, j) in zip(dSdN, dSdN_pre))):
            return (dSdN[1], dSdN[0])
        dSdN_pre = dSdN

def _get_TV(codon_lst1, codon_lst2, codon_table):
    if False:
        for i in range(10):
            print('nop')
    'Get TV (PRIVATE).\n\n    Arguments:\n     - T - proportions of transitional differences\n     - V - proportions of transversional differences\n\n    '
    purine = ('A', 'G')
    pyrimidine = ('C', 'T')
    TV = [0, 0]
    sites = 0
    for (codon1, codon2) in zip(codon_lst1, codon_lst2):
        if '---' not in (codon1, codon2):
            for (i, j) in zip(codon1, codon2):
                if i == j:
                    pass
                elif i in purine and j in purine:
                    TV[0] += 1
                elif i in pyrimidine and j in pyrimidine:
                    TV[0] += 1
                else:
                    TV[1] += 1
                sites += 1
    return (TV[0] / sites, TV[1] / sites)

def _get_kappa_t(pi, TV, t=False):
    if False:
        return 10
    'Calculate kappa (PRIVATE).\n\n    The following formula and variable names are according to PMID: 10666704\n    '
    pi['Y'] = pi['T'] + pi['C']
    pi['R'] = pi['A'] + pi['G']
    A = (2 * (pi['T'] * pi['C'] + pi['A'] * pi['G']) + 2 * (pi['T'] * pi['C'] * pi['R'] / pi['Y'] + pi['A'] * pi['G'] * pi['Y'] / pi['R']) * (1 - TV[1] / (2 * pi['Y'] * pi['R'])) - TV[0]) / (2 * (pi['T'] * pi['C'] / pi['Y'] + pi['A'] * pi['G'] / pi['R']))
    B = 1 - TV[1] / (2 * pi['Y'] * pi['R'])
    a = -0.5 * log(A)
    b = -0.5 * log(B)
    kappaF84 = a / b - 1
    if t is False:
        kappaHKY85 = 1 + (pi['T'] * pi['C'] / pi['Y'] + pi['A'] * pi['G'] / pi['R']) * kappaF84 / (pi['T'] * pi['C'] + pi['A'] * pi['G'])
        return kappaHKY85
    else:
        t = (4 * pi['T'] * pi['C'] * (1 + kappaF84 / pi['Y']) + 4 * pi['A'] * pi['G'] * (1 + kappaF84 / pi['R']) + 4 * pi['Y'] * pi['R']) * b
        return t

def _count_site_YN00(codon_lst1, codon_lst2, pi, k, codon_table):
    if False:
        while True:
            i = 10
    'Site counting method from Ina / Yang and Nielsen (PRIVATE).\n\n    Method from `Ina (1995)`_ as modified by `Yang and Nielsen (2000)`_.\n    This will return the total number of synonymous and nonsynonymous sites\n    and base frequencies in each category. The function is equivalent to\n    the ``CountSites()`` function in ``yn00.c`` of PAML.\n\n    .. _`Ina (1995)`: https://doi.org/10.1007/BF00167113\n    .. _`Yang and Nielsen (2000)`: https://doi.org/10.1093/oxfordjournals.molbev.a026236\n\n    '
    if len(codon_lst1) != len(codon_lst2):
        raise RuntimeError('Length of two codon_lst should be the same (%d and %d detected)' % (len(codon_lst1), len(codon_lst2)))
    else:
        length = len(codon_lst1)
    purine = ('A', 'G')
    pyrimidine = ('T', 'C')
    base_tuple = ('A', 'T', 'C', 'G')
    codon_dict = codon_table.forward_table
    stop = codon_table.stop_codons
    codon_npath = {}
    for (i, j) in zip(codon_lst1, codon_lst2):
        if i != '---' and j != '---':
            codon_npath.setdefault((i, j), 0)
            codon_npath[i, j] += 1
    S_sites = N_sites = 0
    freqSN = [{'A': 0, 'T': 0, 'C': 0, 'G': 0}, {'A': 0, 'T': 0, 'C': 0, 'G': 0}]
    for (codon_pair, npath) in codon_npath.items():
        codon = codon_pair[0]
        S = N = 0
        for pos in range(3):
            for base in base_tuple:
                if codon[pos] == base:
                    continue
                neighbor_codon = codon[:pos] + base + codon[pos + 1:]
                if neighbor_codon in stop:
                    continue
                weight = pi[neighbor_codon]
                if codon[pos] in pyrimidine and base in pyrimidine:
                    weight *= k
                elif codon[pos] in purine and base in purine:
                    weight *= k
                if codon_dict[codon] == codon_dict[neighbor_codon]:
                    S += weight
                    freqSN[0][base] += weight * npath
                else:
                    N += weight
                    freqSN[1][base] += weight * npath
        S_sites += S * npath
        N_sites += N * npath
    norm_const = 3 * length / (S_sites + N_sites)
    S_sites *= norm_const
    N_sites *= norm_const
    for i in freqSN:
        norm_const = sum(i.values())
        for b in i:
            i[b] /= norm_const
    return (S_sites, N_sites, freqSN)

def _count_diff_YN00(codon1, codon2, P, codon_lst, codon_table):
    if False:
        for i in range(10):
            print('nop')
    'Count differences between two codons (three-letter string; PRIVATE).\n\n    The function will weighted multiple pathways from codon1 to codon2\n    according to P matrix of codon substitution. The proportion\n    of transition and transversion (TV) will also be calculated in\n    the function.\n    '
    if not isinstance(codon1, str) or not isinstance(codon2, str):
        raise TypeError(f'_count_diff_YN00 accepts string object to represent codon ({type(codon1)}, {type(codon2)} detected)')
    if len(codon1) != 3 or len(codon2) != 3:
        raise RuntimeError(f'codon should be three letter string ({len(codon1)}, {len(codon2)} detected)')
    TV = [0, 0, 0, 0]
    if codon1 == '---' or codon2 == '---':
        return TV
    base_tuple = ('A', 'C', 'G', 'T')
    if not all((i in base_tuple for i in codon1)):
        raise RuntimeError(f'Unrecognized character detected in codon1 {codon1} (Codons consist of A, T, C or G)')
    if not all((i in base_tuple for i in codon2)):
        raise RuntimeError(f'Unrecognized character detected in codon2 {codon2} (Codons consist of A, T, C or G)')
    if codon1 == codon2:
        return TV
    else:
        diff_pos = []
        for (i, k) in enumerate(zip(codon1, codon2)):
            if k[0] != k[1]:
                diff_pos.append(i)

        def count_TV(codon1, codon2, diff, codon_table, weight=1):
            if False:
                print('Hello World!')
            purine = ('A', 'G')
            pyrimidine = ('T', 'C')
            dic = codon_table.forward_table
            stop = codon_table.stop_codons
            if codon1 in stop or codon2 in stop:
                if codon1[diff] in purine and codon2[diff] in purine:
                    return [0, 0, weight, 0]
                elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
                    return [0, 0, weight, 0]
                else:
                    return [0, 0, 0, weight]
            elif dic[codon1] == dic[codon2]:
                if codon1[diff] in purine and codon2[diff] in purine:
                    return [weight, 0, 0, 0]
                elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
                    return [weight, 0, 0, 0]
                else:
                    return [0, weight, 0, 0]
            elif codon1[diff] in purine and codon2[diff] in purine:
                return [0, 0, weight, 0]
            elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
                return [0, 0, weight, 0]
            else:
                return [0, 0, 0, weight]
        if len(diff_pos) == 1:
            TV = [p + q for (p, q) in zip(TV, count_TV(codon1, codon2, diff_pos[0], codon_table))]
        elif len(diff_pos) == 2:
            tmp_codon = [codon1[:i] + codon2[i] + codon1[i + 1:] for i in diff_pos]
            path_prob = []
            for i in tmp_codon:
                codon_idx = list(map(codon_lst.index, [codon1, i, codon2]))
                prob = (P[codon_idx[0], codon_idx[1]], P[codon_idx[1], codon_idx[2]])
                path_prob.append(prob[0] * prob[1])
            path_prob = [2 * i / sum(path_prob) for i in path_prob]
            for (n, i) in enumerate(diff_pos):
                temp_codon = codon1[:i] + codon2[i] + codon1[i + 1:]
                TV = [p + q for (p, q) in zip(TV, count_TV(codon1, temp_codon, i, codon_table, weight=path_prob[n] / 2))]
                TV = [p + q for (p, q) in zip(TV, count_TV(codon1, temp_codon, i, codon_table, weight=path_prob[n] / 2))]
        elif len(diff_pos) == 3:
            paths = list(permutations([0, 1, 2], 3))
            path_prob = []
            tmp_codon = []
            for p in paths:
                tmp1 = codon1[:p[0]] + codon2[p[0]] + codon1[p[0] + 1:]
                tmp2 = tmp1[:p[1]] + codon2[p[1]] + tmp1[p[1] + 1:]
                tmp_codon.append((tmp1, tmp2))
                codon_idx = list(map(codon_lst.index, [codon1, tmp1, tmp2, codon2]))
                prob = (P[codon_idx[0], codon_idx[1]], P[codon_idx[1], codon_idx[2]], P[codon_idx[2], codon_idx[3]])
                path_prob.append(prob[0] * prob[1] * prob[2])
            path_prob = [3 * i / sum(path_prob) for i in path_prob]
            for (i, j, k) in zip(tmp_codon, path_prob, paths):
                TV = [p + q for (p, q) in zip(TV, count_TV(codon1, i[0], k[0], codon_table, weight=j / 3))]
                TV = [p + q for (p, q) in zip(TV, count_TV(i[0], i[1], k[1], codon_table, weight=j / 3))]
                TV = [p + q for (p, q) in zip(TV, count_TV(i[1], codon2, k[1], codon_table, weight=j / 3))]
    return TV

def _ml(seq1, seq2, cmethod, codon_table):
    if False:
        for i in range(10):
            print('nop')
    'ML method main function (PRIVATE).'
    from collections import Counter
    from scipy.optimize import minimize
    codon_cnt = Counter()
    pi = _get_pi(seq1, seq2, cmethod, codon_table=codon_table)
    for (i, j) in zip(seq1, seq2):
        if '---' not in (i, j):
            codon_cnt[i, j] += 1
    codon_lst = [i for i in list(codon_table.forward_table.keys()) + codon_table.stop_codons if 'U' not in i]

    def func(params, pi=pi, codon_cnt=codon_cnt, codon_lst=codon_lst, codon_table=codon_table):
        if False:
            for i in range(10):
                print('nop')
        'Temporary function, params = [t, k, w].'
        return -_likelihood_func(params[0], params[1], params[2], pi, codon_cnt, codon_lst=codon_lst, codon_table=codon_table)
    opt_res = minimize(func, [1, 0.1, 2], method='L-BFGS-B', bounds=((1e-10, 20), (1e-10, 20), (1e-10, 10)), tol=1e-05)
    (t, k, w) = opt_res.x
    Q = _get_Q(pi, k, w, codon_lst, codon_table)
    Sd = Nd = 0
    for (i, c1) in enumerate(codon_lst):
        for (j, c2) in enumerate(codon_lst):
            if i != j:
                try:
                    if codon_table.forward_table[c1] == codon_table.forward_table[c2]:
                        Sd += pi[c1] * Q[i, j]
                    else:
                        Nd += pi[c1] * Q[i, j]
                except KeyError:
                    pass
    Sd *= t
    Nd *= t

    def func_w1(params, pi=pi, codon_cnt=codon_cnt, codon_lst=codon_lst, codon_table=codon_table):
        if False:
            return 10
        'Temporary function, params = [t, k]. w is fixed to 1.'
        return -_likelihood_func(params[0], params[1], 1.0, pi, codon_cnt, codon_lst=codon_lst, codon_table=codon_table)
    opt_res = minimize(func_w1, [1, 0.1], method='L-BFGS-B', bounds=((1e-10, 20), (1e-10, 20)), tol=1e-05)
    (t, k) = opt_res.x
    w = 1.0
    Q = _get_Q(pi, k, w, codon_lst, codon_table)
    rhoS = rhoN = 0
    for (i, c1) in enumerate(codon_lst):
        for (j, c2) in enumerate(codon_lst):
            if i != j:
                try:
                    if codon_table.forward_table[c1] == codon_table.forward_table[c2]:
                        rhoS += pi[c1] * Q[i, j]
                    else:
                        rhoN += pi[c1] * Q[i, j]
                except KeyError:
                    pass
    rhoS *= 3
    rhoN *= 3
    dN = Nd / rhoN
    dS = Sd / rhoS
    return (dN, dS)

def _get_pi(seq1, seq2, cmethod, codon_table):
    if False:
        print('Hello World!')
    'Obtain codon frequency dict (pi) from two codon list (PRIVATE).\n\n    This function is designed for ML method. Available counting methods\n    (cfreq) are F1x4, F3x4 and F64.\n    '
    pi = {}
    if cmethod == 'F1x4':
        fcodon = {'A': 0, 'G': 0, 'C': 0, 'T': 0}
        for i in seq1 + seq2:
            if i != '---':
                for c in i:
                    fcodon[c] += 1
        tot = sum(fcodon.values())
        fcodon = {j: k / tot for (j, k) in fcodon.items()}
        for i in codon_table.forward_table.keys() + codon_table.stop_codons:
            if 'U' not in i:
                pi[i] = fcodon[i[0]] * fcodon[i[1]] * fcodon[i[2]]
    elif cmethod == 'F3x4':
        fcodon = [{'A': 0, 'G': 0, 'C': 0, 'T': 0}, {'A': 0, 'G': 0, 'C': 0, 'T': 0}, {'A': 0, 'G': 0, 'C': 0, 'T': 0}]
        for i in seq1 + seq2:
            if i != '---':
                fcodon[0][i[0]] += 1
                fcodon[1][i[1]] += 1
                fcodon[2][i[2]] += 1
        for i in range(3):
            tot = sum(fcodon[i].values())
            fcodon[i] = {j: k / tot for (j, k) in fcodon[i].items()}
        for i in list(codon_table.forward_table.keys()) + codon_table.stop_codons:
            if 'U' not in i:
                pi[i] = fcodon[0][i[0]] * fcodon[1][i[1]] * fcodon[2][i[2]]
    elif cmethod == 'F61':
        for i in codon_table.forward_table.keys() + codon_table.stop_codons:
            if 'U' not in i:
                pi[i] = 0.1
        for i in seq1 + seq2:
            if i != '---':
                pi[i] += 1
        tot = sum(pi.values())
        pi = {j: k / tot for (j, k) in pi.items()}
    return pi

def _q(i, j, pi, k, w, codon_table):
    if False:
        return 10
    'Q matrix for codon substitution (PRIVATE).\n\n    Arguments:\n     - i, j  : three letter codon string\n     - pi    : expected codon frequency\n     - k     : transition/transversion ratio\n     - w     : nonsynonymous/synonymous rate ratio\n     - codon_table: Bio.Data.CodonTable object\n\n    '
    if i == j:
        return 0
    if i in codon_table.stop_codons or j in codon_table.stop_codons:
        return 0
    if i not in pi or j not in pi:
        return 0
    purine = ('A', 'G')
    pyrimidine = ('T', 'C')
    diff = []
    for (n, (c1, c2)) in enumerate(zip(i, j)):
        if c1 != c2:
            diff.append((n, c1, c2))
    if len(diff) >= 2:
        return 0
    if codon_table.forward_table[i] == codon_table.forward_table[j]:
        if diff[0][1] in purine and diff[0][2] in purine:
            return k * pi[j]
        elif diff[0][1] in pyrimidine and diff[0][2] in pyrimidine:
            return k * pi[j]
        else:
            return pi[j]
    elif diff[0][1] in purine and diff[0][2] in purine:
        return w * k * pi[j]
    elif diff[0][1] in pyrimidine and diff[0][2] in pyrimidine:
        return w * k * pi[j]
    else:
        return w * pi[j]

def _get_Q(pi, k, w, codon_lst, codon_table):
    if False:
        for i in range(10):
            print('nop')
    'Q matrix for codon substitution (PRIVATE).'
    import numpy as np
    codon_num = len(codon_lst)
    Q = np.zeros((codon_num, codon_num))
    for i in range(codon_num):
        for j in range(codon_num):
            if i != j:
                Q[i, j] = _q(codon_lst[i], codon_lst[j], pi, k, w, codon_table=codon_table)
    nucl_substitutions = 0
    for i in range(codon_num):
        Q[i, i] = -sum(Q[i, :])
        try:
            nucl_substitutions += pi[codon_lst[i]] * -Q[i, i]
        except KeyError:
            pass
    Q = Q / nucl_substitutions
    return Q

def _likelihood_func(t, k, w, pi, codon_cnt, codon_lst, codon_table):
    if False:
        while True:
            i = 10
    'Likelihood function for ML method (PRIVATE).'
    from scipy.linalg import expm
    Q = _get_Q(pi, k, w, codon_lst, codon_table)
    P = expm(Q * t)
    likelihood = 0
    for (i, c1) in enumerate(codon_lst):
        for (j, c2) in enumerate(codon_lst):
            if (c1, c2) in codon_cnt:
                if P[i, j] * pi[c1] <= 0:
                    likelihood += codon_cnt[c1, c2] * 0
                else:
                    likelihood += codon_cnt[c1, c2] * log(pi[c1] * P[i, j])
    return likelihood
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()