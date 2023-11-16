from __future__ import print_function
from collections import defaultdict

def get_genomes(fname='byronbayseqs.fas.txt'):
    if False:
        i = 10
        return i + 15
    'Return a list of genomes, and a list of their corresponding names.'
    import re
    (names, species, genomes) = ([], [], [])
    for (name, g) in re.findall('>(.*?)\r([^\r]*)\r*', open(fname).read()):
        names.append(name)
        species.append(name.split('|')[-1])
        genomes.append(g)
    return (names, species, genomes)

def get_neighbors(fname='editdistances.txt'):
    if False:
        while True:
            i = 10
    'Return dict: neighbors[i][j] = neighbors[j][i] = d means i,j are d apart.'
    neighbors = dict(((i, {}) for i in range(n)))
    for line in open(fname):
        (i, j, d) = map(int, line.split())
        neighbors[i][j] = neighbors[j][i] = d
    return neighbors

def cluster(neighbors, d, dc):
    if False:
        for i in range(10):
            print('nop')
    'Return a list of clusters, each cluster element is within d of another\n    and within dc of every other cluster element.'
    unclustered = set(neighbors)
    return [closure(g, set(), unclustered, d, dc) for g in neighbors if g in unclustered]

def closure(g, s, unclustered, d, dc):
    if False:
        for i in range(10):
            print('nop')
    "Accumulate in set s the transitive closure of 'near', starting at g"
    if g not in s and g in unclustered and near(g, s, d, dc):
        s.add(g)
        unclustered.remove(g)
        for g2 in neighbors[g]:
            closure(g2, s, unclustered, d, dc)
    return s

def dist(i, j):
    if False:
        return 10
    'Distance between two genomes.'
    if i == j:
        return 0
    return neighbors[min(i, j)].get(max(i, j), max_distance)

def near(g, cluster, d, dc):
    if False:
        return 10
    'Is g within d of some member of c, and within dc of every member of c?'
    distances = [dist(g, g2) for g2 in cluster] or [0]
    return min(distances) <= d and max(distances) <= dc

def diameter(cluster):
    if False:
        while True:
            i = 10
    'The largest distance between two elements of the cluster'
    return max([dist(i, j) for i in cluster for j in cluster] or [0])

def margin(cluster):
    if False:
        for i in range(10):
            print('nop')
    'The distance from a cluster to the nearest g2 outside this cluster.'
    return min([d for g in cluster for (g2, d) in neighbors[g].items() if g2 not in cluster] or [max_distance])

def pct(num, den):
    if False:
        return 10
    'Return a string representing the percentage. '
    if '__len__' in dir(den):
        den = len(den)
    if num == den:
        return ' 100%'
    return '%.1f%%' % (num * 100.0 / den)

def histo(items):
    if False:
        for i in range(10):
            print('nop')
    'Make a histogram from a sequence of items or (item, count) tuples.'
    D = defaultdict(int)
    for item in items:
        if isinstance(item, tuple):
            D[item[0]] += item[1]
        else:
            D[item] += 1
    return D

def showh(d):
    if False:
        while True:
            i = 10
    'Show a histogram'
    if not isinstance(d, dict):
        d = histo(d)
    return ' '.join(('%s:%s' % i for i in sorted(d.items())))

def greport(genomes):
    if False:
        return 10
    print('Number of genomes: %d (%d distinct)' % (len(genomes), len(set(genomes))))
    G = dict(((g, set()) for g in genomes))
    for i in range(n):
        G[genomes[i]].add(species[i])
    print('Multi-named genomes:', len([s for s in G.values() if len(s) > 1]))
    lens = map(len, genomes)
    print('Genome lengths: min=%d, max=%d' % (min(lens), max(lens)))
    print('Character counts: ', showh((c for g in genomes for c in g)))

def nreport(neighbors):
    if False:
        while True:
            i = 10
    (NN, NumN) = (defaultdict(int), defaultdict(int))
    for n in neighbors:
        nn = min(neighbors[n].values() or ['>25'])
        NN[nn] += 1
        for d2 in neighbors[n].values():
            NumN[d2] += 1
    print()
    print('Nearest neighbor counts:', showh(NN))
    print('Number of neighbors at each distance:', showh(NumN))

def nspecies(c):
    if False:
        i = 10
        return i + 15
    return len(set((species[g] for g in c)))

def showc(c):
    if False:
        while True:
            i = 10
    return 'N=%d, D=%d, M=%d: %s %s' % (len(c), diameter(c), margin(c), list(c), showh((species[g] for g in c)))

def creport(drange, dcrange):
    if False:
        print('Hello World!')

    def table(what, fn):
        if False:
            print('Hello World!')
        print('\n' + what)
        print(' ' * 8, ' '.join([' ' + pct(dc, glen) for dc in dcrange]))
        for d in drange:
            print('%s (%2d)' % (pct(d, glen), d), end=' ')
            for dc in dcrange:
                print('%5s' % fn(cluster(neighbors, d, dc)), end=' ')
            print()
    print('\nNearest neighbor must be closer than this percentage (places). ')
    print('Each column: all genomes in cluster within this percentage of each other.')
    table('Number of clusters', len)
    cluster1 = cluster(neighbors, 8, 15)
    print('\nNumber of clusters of different sizes:', showh((len(c) for c in cluster1)))
    (M, T) = (defaultdict(int), defaultdict(int))
    for c in cluster1:
        M[margin(c)] += 1
        T[margin(c)] += len(c)
    for x in M:
        print('%d\t%d\t%d' % (x, M[x], T[x]))
    print('\nMargins', showh(M))
    for c in cluster1:
        if margin(c) <= 16:
            print(showc(c))
    print('\nScatter plot of cluster diameter vs. margin.')
    for c in cluster1:
        if diameter(c) > 0:
            pass
    print('\nDifference from cluster(neighbors, 11, 14):')
    print('\nNumber of clusters witth more than one species name:')

    def pct_near_another(clusters, P=1.25):
        if False:
            print('Hello World!')
        total = 0
        for c in clusters:
            d = diameter(c)
            for g in c:
                for g2 in neighbors[g]:
                    if g2 not in c and dist(g, g2) < P * d:
                        total += 1
        return pct(total, n)

    def f(P):
        if False:
            while True:
                i = 10
        print('\nPercent of individuals within %.2f*diameter of another cluster.' % P)
        table(lambda cl: pct_near_another(cl, P))

def sreport(species):
    if False:
        print('Hello World!')
    SS = defaultdict(int)
    print()
    for s in set(species):
        c = [g for g in range(n) if species[g] == s]
        d = diameter(c)
        if d > 14:
            if d == glen:
                d = '>25'
            print('diameter %s for %s (%d elements)' % (d, s, len(c)))
        SS[d] += 1
    print('Diameters of %d labeled clusters: %s' % (len(set(species)), showh(SS)))

def compare(cl1, cl2):
    if False:
        for i in range(10):
            print('nop')
    'Compare two lists of clusters'
    return sum((c1 == c2 or 0.5 * (abs(len(c1) - len(c2)) == 1 and (c1.issubset(c2) or c2.issubset(c1))) for c1 in cl1 for c2 in cl2))

def unit_tests():
    if False:
        for i in range(10):
            print('nop')
    assert set((len(g) for g in genomes)) == set([glen])
    clusters = cluster(neighbors, 11, 11)
    assert sum((len(c) for c in clusters)) == len(genomes)
    assert len(set((g for c in clusters for g in c))) == len(genomes)
    assert dist(17, 42) == dist(42, 17)
    assert diameter(set()) == 0
    assert diameter([17, 42]) == dist(17, 42)
    assert pct(1, 2) == '50.0%'
    print('\nAll tests pass.\n')
max_distance = 26
(names, species, genomes) = get_genomes()
n = len(genomes)
glen = len(genomes[0])
neighbors = get_neighbors()
greport(genomes)
nreport(neighbors)
creport(range(6, 15), [glen, 16, 15, 14, 13, 12, 11])
unit_tests()