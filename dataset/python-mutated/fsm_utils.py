import math
import sys
import numpy
try:
    import scipy.linalg
except ImportError:
    print('Error: Program requires scipy (see: www.scipy.org).')
    sys.exit(1)

def dec2base(num, base, l):
    if False:
        print('Hello World!')
    "\n    Decimal to any base conversion.\n    Convert 'num' to a list of 'l' numbers representing 'num'\n    to base 'base' (most significant symbol first).\n    "
    s = list(range(l))
    n = num
    for i in range(l):
        s[l - i - 1] = n % base
        n = int(n / base)
    if n != 0:
        print('Number ', num, ' requires more than ', l, 'digits.')
    return s

def base2dec(s, base):
    if False:
        i = 10
        return i + 15
    "\n    Conversion from any base to decimal.\n    Convert a list 's' of symbols to a decimal number\n    (most significant symbol first)\n    "
    num = 0
    for i in range(len(s)):
        num = num * base + s[i]
    return num

def make_isi_lookup(mod, channel, normalize):
    if False:
        while True:
            i = 10
    "\n    Automatically generate the lookup table that maps the FSM outputs\n    to channel inputs corresponding to a channel 'channel' and a modulation\n    'mod'. Optional normalization of channel to unit energy.\n    This table is used by the 'metrics' block to translate\n    channel outputs to metrics for use with the Viterbi algorithm.\n    Limitations: currently supports only one-dimensional modulations.\n    "
    dim = mod[0]
    constellation = mod[1]
    if normalize:
        p = 0
        for i in range(len(channel)):
            p = p + channel[i] ** 2
        for i in range(len(channel)):
            channel[i] = channel[i] / math.sqrt(p)
    lookup = list(range(len(constellation) ** len(channel)))
    for o in range(len(constellation) ** len(channel)):
        ss = dec2base(o, len(constellation), len(channel))
        ll = 0
        for i in range(len(channel)):
            ll = ll + constellation[ss[i]] * channel[i]
        lookup[o] = ll
    return (1, lookup)

def make_cpm_signals(K, P, M, L, q, frac):
    if False:
        return 10
    '\n    Automatically generate the signals appropriate for CPM\n    decomposition.\n    This decomposition is based on the paper by B. Rimoldi\n    "A decomposition approach to CPM", IEEE Trans. Info Theory, March 1988\n    See also my own notes at http://www.eecs.umich.edu/~anastas/docs/cpm.pdf\n    '
    Q = numpy.size(q) / L
    h = 1.0 * K / P
    f0 = -h * (M - 1) / 2
    dt = 0.0
    t = (dt + numpy.arange(0, Q)) / Q
    qq = numpy.zeros(Q)
    for m in range(L):
        qq = qq + q[m * Q:m * Q + Q]
        w = math.pi * h * (M - 1) * t - 2 * math.pi * h * (M - 1) * qq + math.pi * h * (L - 1) * (M - 1)
    X = M ** L * P
    PSI = numpy.empty((X, Q))
    for x in range(X):
        xv = dec2base(x / P, M, L)
        xv = numpy.append(xv, x % P)
        qq1 = numpy.zeros(Q)
        for m in range(L):
            qq1 = qq1 + xv[m] * q[m * Q:m * Q + Q]
        psi = 2 * math.pi * h * xv[-1] + 4 * math.pi * h * qq1 + w
        PSI[x] = psi
    PSI = numpy.transpose(PSI)
    SS = numpy.exp(1j * PSI)
    F = scipy.linalg.orth(SS)
    S = numpy.dot(numpy.transpose(F.conjugate()), SS)
    E = numpy.sum(numpy.absolute(S) ** 2, axis=1) / Q
    E = E / numpy.sum(E)
    Es = -numpy.sort(-E)
    Esi = numpy.argsort(-E)
    Ecum = numpy.cumsum(Es)
    v0 = numpy.searchsorted(Ecum, frac)
    N = v0 + 1
    Ff = numpy.transpose(numpy.transpose(F)[Esi[0:v0 + 1]])
    Sf = S[Esi[0:v0 + 1]]
    return (f0, SS, S, F, Sf, Ff, N)
pam2 = (1, [-1, 1])
pam4 = (1, [-3, -1, 3, 1])
pam8 = (1, [-7, -5, -3, -1, 1, 3, 5, 7])
psk4 = (2, [1, 0, 0, 1, 0, -1, -1, 0])
psk8 = (2, [math.cos(2 * math.pi * 0 / 8), math.sin(2 * math.pi * 0 / 8), math.cos(2 * math.pi * 1 / 8), math.sin(2 * math.pi * 1 / 8), math.cos(2 * math.pi * 2 / 8), math.sin(2 * math.pi * 2 / 8), math.cos(2 * math.pi * 3 / 8), math.sin(2 * math.pi * 3 / 8), math.cos(2 * math.pi * 4 / 8), math.sin(2 * math.pi * 4 / 8), math.cos(2 * math.pi * 5 / 8), math.sin(2 * math.pi * 5 / 8), math.cos(2 * math.pi * 6 / 8), math.sin(2 * math.pi * 6 / 8), math.cos(2 * math.pi * 7 / 8), math.sin(2 * math.pi * 7 / 8)])
psk2x3 = (3, [-1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1])
psk2x4 = (4, [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1])
orth2 = (2, [1, 0, 0, 1])
orth4 = (4, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
c_channel = [0.227, 0.46, 0.688, 0.46, 0.227]