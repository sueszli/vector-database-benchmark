from gnuradio import gr
from gnuradio import audio
from gnuradio import trellis, digital, filter, blocks
from gnuradio import eng_notation
import math
import sys
import random
import fsm_utils
try:
    from gnuradio import analog
except ImportError:
    sys.stderr.write('Error: Program requires gr-analog.\n')
    sys.exit(1)

def run_test(f, Kb, bitspersymbol, K, channel, modulation, dimensionality, tot_constellation, N0, seed):
    if False:
        for i in range(10):
            print('nop')
    tb = gr.top_block()
    L = len(channel)
    packet = [0] * (K + 2 * L)
    random.seed(seed)
    for i in range(len(packet)):
        packet[i] = random.randint(0, 2 ** bitspersymbol - 1)
    for i in range(L):
        packet[i] = 0
        packet[len(packet) - i - 1] = 0
    src = blocks.vector_source_s(packet, False)
    mod = digital.chunks_to_symbols_sf(modulation[1], modulation[0])
    isi = filter.fir_filter_fff(1, channel)
    add = blocks.add_ff()
    noise = analog.noise_source_f(analog.GR_GAUSSIAN, math.sqrt(N0 / 2), seed)
    skip = blocks.skiphead(gr.sizeof_float, L)
    va = trellis.viterbi_combined_s(f, K + L, 0, 0, dimensionality, tot_constellation, digital.TRELLIS_EUCLIDEAN)
    dst = blocks.vector_sink_s()
    tb.connect(src, mod)
    tb.connect(mod, isi, (add, 0))
    tb.connect(noise, (add, 1))
    tb.connect(add, skip, va, dst)
    tb.run()
    data = dst.data()
    ntotal = len(data) - L
    nright = 0
    for i in range(ntotal):
        if packet[i + L] == data[i]:
            nright = nright + 1
    return (ntotal, ntotal - nright)

def main(args):
    if False:
        for i in range(10):
            print('nop')
    nargs = len(args)
    if nargs == 2:
        esn0_db = float(args[0])
        rep = int(args[1])
    else:
        sys.stderr.write('usage: test_viterbi_equalization1.py Es/No_db  repetitions\n')
        sys.exit(1)
    Kb = 2048
    modulation = fsm_utils.pam4
    channel = fsm_utils.c_channel
    f = trellis.fsm(len(modulation[1]), len(channel))
    bitspersymbol = int(round(math.log(f.I()) / math.log(2)))
    K = Kb / bitspersymbol
    tot_channel = fsm_utils.make_isi_lookup(modulation, channel, True)
    dimensionality = tot_channel[0]
    tot_constellation = tot_channel[1]
    N0 = pow(10.0, -esn0_db / 10.0)
    if len(tot_constellation) / dimensionality != f.O():
        sys.stderr.write('Incompatible FSM output cardinality and lookup table size.\n')
        sys.exit(1)
    tot_s = 0
    terr_s = 0
    terr_p = 0
    for i in range(rep):
        (s, e) = run_test(f, Kb, bitspersymbol, K, channel, modulation, dimensionality, tot_constellation, N0, -int(666 + i))
        tot_s = tot_s + s
        terr_s = terr_s + e
        terr_p = terr_p + (terr_s != 0)
        if (i + 1) % 100 == 0:
            print(i + 1, terr_p, '%.2e' % (1.0 * terr_p / (i + 1)), tot_s, terr_s, '%.2e' % (1.0 * terr_s / tot_s))
    print(rep, terr_p, '%.2e' % (1.0 * terr_p / (i + 1)), tot_s, terr_s, '%.2e' % (1.0 * terr_s / tot_s))
if __name__ == '__main__':
    main(sys.argv[1:])