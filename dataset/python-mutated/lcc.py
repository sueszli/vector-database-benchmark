"""Local Composition Complexity."""
import math

def lcc_mult(seq, wsize):
    if False:
        while True:
            i = 10
    'Calculate Local Composition Complexity (LCC) values over sliding window.\n\n    Returns a list of floats, the LCC values for a sliding window over\n    the sequence.\n\n    seq - an unambiguous DNA sequence (a string or Seq object)\n    wsize - window size, integer\n\n    The result is the same as applying lcc_simp multiple times, but this\n    version is optimized for speed. The optimization works by using the\n    value of previous window as a base to compute the next one.\n    '
    l4 = math.log(4)
    seq = seq.upper()
    tamseq = len(seq)
    compone = [0]
    lccsal = []
    for i in range(wsize):
        compone.append((i + 1) / wsize * math.log((i + 1) / wsize) / l4)
    window = seq[0:wsize]
    cant_a = window.count('A')
    cant_c = window.count('C')
    cant_t = window.count('T')
    cant_g = window.count('G')
    term_a = compone[cant_a]
    term_c = compone[cant_c]
    term_t = compone[cant_t]
    term_g = compone[cant_g]
    lccsal.append(-(term_a + term_c + term_t + term_g))
    tail = seq[0]
    for x in range(tamseq - wsize):
        window = seq[x + 1:wsize + x + 1]
        if tail == window[-1]:
            lccsal.append(lccsal[-1])
        elif tail == 'A':
            cant_a -= 1
            if window.endswith('C'):
                cant_c += 1
                term_a = compone[cant_a]
                term_c = compone[cant_c]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('T'):
                cant_t += 1
                term_a = compone[cant_a]
                term_t = compone[cant_t]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('G'):
                cant_g += 1
                term_a = compone[cant_a]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
        elif tail == 'C':
            cant_c -= 1
            if window.endswith('A'):
                cant_a += 1
                term_a = compone[cant_a]
                term_c = compone[cant_c]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('T'):
                cant_t += 1
                term_c = compone[cant_c]
                term_t = compone[cant_t]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('G'):
                cant_g += 1
                term_c = compone[cant_c]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
        elif tail == 'T':
            cant_t -= 1
            if window.endswith('A'):
                cant_a += 1
                term_a = compone[cant_a]
                term_t = compone[cant_t]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('C'):
                cant_c += 1
                term_c = compone[cant_c]
                term_t = compone[cant_t]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('G'):
                cant_g += 1
                term_t = compone[cant_t]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
        elif tail == 'G':
            cant_g -= 1
            if window.endswith('A'):
                cant_a += 1
                term_a = compone[cant_a]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('C'):
                cant_c += 1
                term_c = compone[cant_c]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('T'):
                cant_t += 1
                term_t = compone[cant_t]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
        tail = window[0]
    return lccsal

def lcc_simp(seq):
    if False:
        for i in range(10):
            print('nop')
    'Calculate Local Composition Complexity (LCC) for a sequence.\n\n    seq - an unambiguous DNA sequence (a string or Seq object)\n\n    Returns the Local Composition Complexity (LCC) value for the entire\n    sequence (as a float).\n\n    Reference:\n    Andrzej K Konopka (2005) Sequence Complexity and Composition\n    https://doi.org/10.1038/npg.els.0005260\n    '
    wsize = len(seq)
    seq = seq.upper()
    l4 = math.log(4)
    if 'A' not in seq:
        term_a = 0
    else:
        term_a = seq.count('A') / wsize * math.log(seq.count('A') / wsize) / l4
    if 'C' not in seq:
        term_c = 0
    else:
        term_c = seq.count('C') / wsize * math.log(seq.count('C') / wsize) / l4
    if 'T' not in seq:
        term_t = 0
    else:
        term_t = seq.count('T') / wsize * math.log(seq.count('T') / wsize) / l4
    if 'G' not in seq:
        term_g = 0
    else:
        term_g = seq.count('G') / wsize * math.log(seq.count('G') / wsize) / l4
    return -(term_a + term_c + term_t + term_g)