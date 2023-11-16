"""Code for dealing with Codon Alignment.

CodonAlignment class is inherited from MultipleSeqAlignment class. This is
the core class to deal with codon alignment in biopython.
"""
from math import sqrt, erfc
import warnings
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio import BiopythonWarning
from Bio.codonalign.codonseq import _get_codon_list, CodonSeq, cal_dn_ds

class CodonAlignment(MultipleSeqAlignment):
    """Codon Alignment class that inherits from MultipleSeqAlignment.

    >>> from Bio.SeqRecord import SeqRecord
    >>> a = SeqRecord(CodonSeq("AAAACGTCG"), id="Alpha")
    >>> b = SeqRecord(CodonSeq("AAA---TCG"), id="Beta")
    >>> c = SeqRecord(CodonSeq("AAAAGGTGG"), id="Gamma")
    >>> print(CodonAlignment([a, b, c]))
    CodonAlignment with 3 rows and 9 columns (3 codons)
    AAAACGTCG Alpha
    AAA---TCG Beta
    AAAAGGTGG Gamma

    """

    def __init__(self, records='', name=None):
        if False:
            print('Hello World!')
        'Initialize the class.'
        MultipleSeqAlignment.__init__(self, records)
        for rec in self:
            if not isinstance(rec.seq, CodonSeq):
                raise TypeError('CodonSeq objects are expected in each SeqRecord in CodonAlignment')
        if self.get_alignment_length() % 3 != 0:
            raise ValueError('Alignment length is not a multiple of three (i.e. a whole number of codons)')

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a multi-line string summary of the alignment.\n\n        This output is indicated to be readable, but large alignment\n        is shown truncated. A maximum of 20 rows (sequences) and\n        60 columns (20 codons) are shown, with the record identifiers.\n        This should fit nicely on a single screen. e.g.\n\n        '
        rows = len(self._records)
        lines = ['CodonAlignment with %i rows and %i columns (%i codons)' % (rows, self.get_alignment_length(), self.get_aln_length())]
        if rows <= 60:
            lines.extend([self._str_line(rec, length=60) for rec in self._records])
        else:
            lines.extend([self._str_line(rec, length=60) for rec in self._records[:18]])
            lines.append('...')
            lines.append(self._str_line(self._records[-1], length=60))
        return '\n'.join(lines)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        'Return a CodonAlignment object for single indexing.'
        if isinstance(index, int):
            return self._records[index]
        elif isinstance(index, slice):
            return CodonAlignment(self._records[index])
        elif len(index) != 2:
            raise TypeError('Invalid index type.')
        (row_index, col_index) = index
        if isinstance(row_index, int):
            return self._records[row_index][col_index]
        elif isinstance(col_index, int):
            return ''.join((str(rec[col_index]) for rec in self._records[row_index]))
        else:
            return MultipleSeqAlignment((rec[col_index] for rec in self._records[row_index]))

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        'Combine two codonalignments with the same number of rows by adding them.\n\n        The method also allows to combine a CodonAlignment object with a\n        MultipleSeqAlignment object. The following rules apply:\n\n            * CodonAlignment + CodonAlignment -> CodonAlignment\n            * CodonAlignment + MultipleSeqAlignment -> MultipleSeqAlignment\n        '
        if isinstance(other, CodonAlignment):
            if len(self) != len(other):
                raise ValueError('When adding two alignments they must have the same length (i.e. same number or rows)')
            warnings.warn('Please make sure the two CodonAlignment objects are sharing the same codon table. This is not checked by Biopython.', BiopythonWarning)
            merged = (SeqRecord(seq=CodonSeq(left.seq + right.seq)) for (left, right) in zip(self, other))
            return CodonAlignment(merged)
        elif isinstance(other, MultipleSeqAlignment):
            if len(self) != len(other):
                raise ValueError('When adding two alignments they must have the same length (i.e. same number or rows)')
            return self.toMultipleSeqAlignment() + other
        else:
            raise TypeError(f'Only CodonAlignment or MultipleSeqAlignment object can be added with a CodonAlignment object. {object(other)} detected.')

    def get_aln_length(self):
        if False:
            print('Hello World!')
        'Get alignment length.'
        return self.get_alignment_length() // 3

    def toMultipleSeqAlignment(self):
        if False:
            while True:
                i = 10
        'Convert the CodonAlignment to a MultipleSeqAlignment.\n\n        Return a MultipleSeqAlignment containing all the\n        SeqRecord in the CodonAlignment using Seq to store\n        sequences\n        '
        alignments = [SeqRecord(rec.seq.toSeq(), id=rec.id) for rec in self._records]
        return MultipleSeqAlignment(alignments)

    def get_dn_ds_matrix(self, method='NG86', codon_table=None):
        if False:
            i = 10
            return i + 15
        'Available methods include NG86, LWL85, YN00 and ML.\n\n        Argument:\n         - method       - Available methods include NG86, LWL85, YN00 and ML.\n         - codon_table  - Codon table to use for forward translation.\n\n        '
        from Bio.Phylo.TreeConstruction import DistanceMatrix as DM
        if codon_table is None:
            codon_table = CodonTable.generic_by_id[1]
        names = [i.id for i in self._records]
        size = len(self._records)
        dn_matrix = []
        ds_matrix = []
        for i in range(size):
            dn_matrix.append([])
            ds_matrix.append([])
            for j in range(i + 1):
                if i != j:
                    (dn, ds) = cal_dn_ds(self._records[i], self._records[j], method=method, codon_table=codon_table)
                    dn_matrix[i].append(dn)
                    ds_matrix[i].append(ds)
                else:
                    dn_matrix[i].append(0.0)
                    ds_matrix[i].append(0.0)
        dn_dm = DM(names, matrix=dn_matrix)
        ds_dm = DM(names, matrix=ds_matrix)
        return (dn_dm, ds_dm)

    def get_dn_ds_tree(self, dn_ds_method='NG86', tree_method='UPGMA', codon_table=None):
        if False:
            for i in range(10):
                print('nop')
        'Construct dn tree and ds tree.\n\n        Argument:\n         - dn_ds_method - Available methods include NG86, LWL85, YN00 and ML.\n         - tree_method  - Available methods include UPGMA and NJ.\n\n        '
        from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
        if codon_table is None:
            codon_table = CodonTable.generic_by_id[1]
        (dn_dm, ds_dm) = self.get_dn_ds_matrix(method=dn_ds_method, codon_table=codon_table)
        dn_constructor = DistanceTreeConstructor()
        ds_constructor = DistanceTreeConstructor()
        if tree_method == 'UPGMA':
            dn_tree = dn_constructor.upgma(dn_dm)
            ds_tree = ds_constructor.upgma(ds_dm)
        elif tree_method == 'NJ':
            dn_tree = dn_constructor.nj(dn_dm)
            ds_tree = ds_constructor.nj(ds_dm)
        else:
            raise RuntimeError(f'Unknown tree method ({tree_method}). Only NJ and UPGMA are accepted.')
        return (dn_tree, ds_tree)

    @classmethod
    def from_msa(cls, align):
        if False:
            i = 10
            return i + 15
        "Convert a MultipleSeqAlignment to CodonAlignment.\n\n        Function to convert a MultipleSeqAlignment to CodonAlignment.\n        It is the user's responsibility to ensure all the requirement\n        needed by CodonAlignment is met.\n        "
        rec = [SeqRecord(CodonSeq(str(i.seq)), id=i.id) for i in align._records]
        return cls(rec)

def mktest(codon_alns, codon_table=None, alpha=0.05):
    if False:
        i = 10
        return i + 15
    'McDonald-Kreitman test for neutrality.\n\n    Implement the McDonald-Kreitman test for neutrality (PMID: 1904993)\n    This method counts changes rather than sites\n    (http://mkt.uab.es/mkt/help_mkt.asp).\n\n    Arguments:\n     - codon_alns  - list of CodonAlignment to compare (each\n       CodonAlignment object corresponds to gene sampled from a species)\n\n    Return the p-value of test result.\n    '
    import copy
    if codon_table is None:
        codon_table = CodonTable.generic_by_id[1]
    if not all((isinstance(i, CodonAlignment) for i in codon_alns)):
        raise TypeError('mktest accepts CodonAlignment list.')
    codon_aln_len = [i.get_alignment_length() for i in codon_alns]
    if len(set(codon_aln_len)) != 1:
        raise RuntimeError('CodonAlignment object for mktest should be of equal length.')
    codon_num = codon_aln_len[0] // 3
    codon_dict = copy.deepcopy(codon_table.forward_table)
    for stop in codon_table.stop_codons:
        codon_dict[stop] = 'stop'
    codon_lst = []
    for codon_aln in codon_alns:
        codon_lst.append([])
        for i in codon_aln:
            codon_lst[-1].append(_get_codon_list(i.seq))
    codon_set = []
    for i in range(codon_num):
        uniq_codons = []
        for j in codon_lst:
            uniq_codon = {k[i] for k in j}
            uniq_codons.append(uniq_codon)
        codon_set.append(uniq_codons)
    (syn_fix, nonsyn_fix, syn_poly, nonsyn_poly) = (0, 0, 0, 0)
    (G, nonsyn_G) = _get_codon2codon_matrix(codon_table=codon_table)
    for i in codon_set:
        all_codon = i[0].union(*i[1:])
        if '-' in all_codon or len(all_codon) == 1:
            continue
        fix_or_not = all((len(k) == 1 for k in i))
        if fix_or_not:
            nonsyn_subgraph = _get_subgraph(all_codon, nonsyn_G)
            subgraph = _get_subgraph(all_codon, G)
            this_non = _count_replacement(all_codon, nonsyn_subgraph)
            this_syn = _count_replacement(all_codon, subgraph) - this_non
            nonsyn_fix += this_non
            syn_fix += this_syn
        else:
            nonsyn_subgraph = _get_subgraph(all_codon, nonsyn_G)
            subgraph = _get_subgraph(all_codon, G)
            this_non = _count_replacement(all_codon, nonsyn_subgraph)
            this_syn = _count_replacement(all_codon, subgraph) - this_non
            nonsyn_poly += this_non
            syn_poly += this_syn
    return _G_test([syn_fix, nonsyn_fix, syn_poly, nonsyn_poly])

def _get_codon2codon_matrix(codon_table):
    if False:
        return 10
    'Get codon codon substitution matrix (PRIVATE).\n\n    Elements in the matrix are number of synonymous and nonsynonymous\n    substitutions required for the substitution.\n    '
    import copy
    base_tuple = ('A', 'T', 'C', 'G')
    codons = [i for i in list(codon_table.forward_table.keys()) + codon_table.stop_codons if 'U' not in i]
    codon_dict = copy.deepcopy(codon_table.forward_table)
    for stop in codon_table.stop_codons:
        codon_dict[stop] = 'stop'
    num = len(codons)
    G = {}
    nonsyn_G = {}
    graph = {}
    graph_nonsyn = {}
    for (i, codon) in enumerate(codons):
        graph[codon] = {}
        graph_nonsyn[codon] = {}
        for (p, b) in enumerate(codon):
            for j in base_tuple:
                tmp_codon = codon[0:p] + j + codon[p + 1:]
                if codon_dict[codon] != codon_dict[tmp_codon]:
                    graph_nonsyn[codon][tmp_codon] = 1
                    graph[codon][tmp_codon] = 1
                elif codon != tmp_codon:
                    graph_nonsyn[codon][tmp_codon] = 0.1
                    graph[codon][tmp_codon] = 1
    for codon1 in codons:
        nonsyn_G[codon1] = {}
        G[codon1] = {}
        for codon2 in codons:
            if codon1 == codon2:
                nonsyn_G[codon1][codon2] = 0
                G[codon1][codon2] = 0
            else:
                nonsyn_G[codon1][codon2] = _dijkstra(graph_nonsyn, codon1, codon2)
                G[codon1][codon2] = _dijkstra(graph, codon1, codon2)
    return (G, nonsyn_G)

def _dijkstra(graph, start, end):
    if False:
        print('Hello World!')
    "Dijkstra's algorithm Python implementation (PRIVATE).\n\n    Algorithm adapted from\n    http://thomas.pelletier.im/2010/02/dijkstras-algorithm-python-implementation/.\n    However, an obvious bug in::\n\n        if D[child_node] >(<) D[node] + child_value:\n\n    is fixed.\n    This function will return the distance between start and end.\n\n    Arguments:\n     - graph: Dictionary of dictionary (keys are vertices).\n     - start: Start vertex.\n     - end: End vertex.\n\n    Output:\n       List of vertices from the beginning to the end.\n\n    "
    D = {}
    P = {}
    for node in graph.keys():
        D[node] = 100
        P[node] = ''
    D[start] = 0
    unseen_nodes = list(graph.keys())
    while len(unseen_nodes) > 0:
        shortest = None
        node = ''
        for temp_node in unseen_nodes:
            if shortest is None:
                shortest = D[temp_node]
                node = temp_node
            elif D[temp_node] < shortest:
                shortest = D[temp_node]
                node = temp_node
        unseen_nodes.remove(node)
        for (child_node, child_value) in graph[node].items():
            if D[child_node] > D[node] + child_value:
                D[child_node] = D[node] + child_value
                P[child_node] = node
        if node == end:
            break
    path = []
    node = end
    distance = 0
    while node != start:
        if path.count(node) == 0:
            path.insert(0, node)
            node = P[node]
        else:
            break
    path.insert(0, start)
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i + 1]]
    return distance

def _count_replacement(codon_set, G):
    if False:
        for i in range(10):
            print('nop')
    'Count replacement needed for a given codon_set (PRIVATE).'
    from math import floor
    if len(codon_set) == 1:
        return (0, 0)
    elif len(codon_set) == 2:
        codons = list(codon_set)
        return floor(G[codons[0]][codons[1]])
    else:
        codons = list(codon_set)
        return _prim(G)

def _prim(G):
    if False:
        return 10
    "Prim's algorithm to find minimum spanning tree (PRIVATE).\n\n    Code is adapted from\n    http://programmingpraxis.com/2010/04/09/minimum-spanning-tree-prims-algorithm/\n    "
    from math import floor
    from collections import defaultdict
    from heapq import heapify, heappop, heappush
    nodes = []
    edges = []
    for i in G.keys():
        nodes.append(i)
        for j in G[i]:
            if (i, j, G[i][j]) not in edges and (j, i, G[i][j]) not in edges:
                edges.append((i, j, G[i][j]))
    conn = defaultdict(list)
    for (n1, n2, c) in edges:
        conn[n1].append((c, n1, n2))
        conn[n2].append((c, n2, n1))
    mst = []
    used = set(nodes[0])
    usable_edges = conn[nodes[0]][:]
    heapify(usable_edges)
    while usable_edges:
        (cost, n1, n2) = heappop(usable_edges)
        if n2 not in used:
            used.add(n2)
            mst.append((n1, n2, cost))
            for e in conn[n2]:
                if e[2] not in used:
                    heappush(usable_edges, e)
    length = 0
    for p in mst:
        length += floor(p[2])
    return length

def _get_subgraph(codons, G):
    if False:
        i = 10
        return i + 15
    'Get the subgraph that contains all codons in list (PRIVATE).'
    subgraph = {}
    for i in codons:
        subgraph[i] = {}
        for j in codons:
            if i != j:
                subgraph[i][j] = G[i][j]
    return subgraph

def _G_test(site_counts):
    if False:
        while True:
            i = 10
    'G test for 2x2 contingency table (PRIVATE).\n\n    Arguments:\n     - site_counts - [syn_fix, nonsyn_fix, syn_poly, nonsyn_poly]\n\n    >>> print("%0.6f" % _G_test([17, 7, 42, 2]))\n    0.004924\n    '
    from math import log
    G = 0
    tot = sum(site_counts)
    tot_syn = site_counts[0] + site_counts[2]
    tot_non = site_counts[1] + site_counts[3]
    tot_fix = sum(site_counts[:2])
    tot_poly = sum(site_counts[2:])
    exp = [tot_fix * tot_syn / tot, tot_fix * tot_non / tot, tot_poly * tot_syn / tot, tot_poly * tot_non / tot]
    for (obs, ex) in zip(site_counts, exp):
        G += obs * log(obs / ex)
    return erfc(sqrt(G))
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()