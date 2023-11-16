from gnuradio import gr
from gnuradio import trellis, digital, blocks
from gnuradio import eng_notation
import math
import sys
import random
from gnuradio.trellis import fsm_utils
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import numpy
try:
    from gnuradio import analog
except ImportError:
    sys.stderr.write('Error: Program requires gr-analog.\n')
    sys.exit(1)

def run_test(f, Kb, bitspersymbol, K, dimensionality, constellation, N0, seed):
    if False:
        for i in range(10):
            print('nop')
    tb = gr.top_block()
    numpy.random.seed(-seed)
    packet = numpy.random.randint(0, 2, Kb)
    packet[Kb - 10:Kb] = 0
    packet[0:Kb] = 0
    src = blocks.vector_source_s(packet.tolist(), False)
    b2s = blocks.unpacked_to_packed_ss(1, gr.GR_MSB_FIRST)
    s2fsmi = blocks.packed_to_unpacked_ss(bitspersymbol, gr.GR_MSB_FIRST)
    enc = trellis.encoder_ss(f, 0)
    mod = digital.chunks_to_symbols_sf(constellation, dimensionality)
    add = blocks.add_ff()
    noise = analog.noise_source_f(analog.GR_GAUSSIAN, math.sqrt(N0 / 2), int(seed))
    va = trellis.viterbi_combined_fs(f, K, 0, 0, dimensionality, constellation, digital.TRELLIS_EUCLIDEAN)
    fsmi2s = blocks.unpacked_to_packed_ss(bitspersymbol, gr.GR_MSB_FIRST)
    s2b = blocks.packed_to_unpacked_ss(1, gr.GR_MSB_FIRST)
    dst = blocks.vector_sink_s()
    tb.connect(src, b2s, s2fsmi, enc, mod)
    tb.connect(mod, (add, 0))
    tb.connect(noise, (add, 1))
    tb.connect(add, va, fsmi2s, s2b, dst)
    tb.run()
    if len(dst.data()) != len(packet):
        print('Error: not enough data:', len(dst.data()), len(packet))
    ntotal = len(packet)
    nwrong = sum(abs(packet - numpy.array(dst.data())))
    return (ntotal, nwrong, abs(packet - numpy.array(dst.data())))

def main():
    if False:
        return 10
    parser = OptionParser(option_class=eng_option)
    parser.add_option('-f', '--fsm_file', type='string', default='fsm_files/awgn1o2_4.fsm', help='Filename containing the fsm specification, e.g. -f fsm_files/awgn1o2_4.fsm (default=fsm_files/awgn1o2_4.fsm)')
    parser.add_option('-e', '--esn0', type='eng_float', default=10.0, help='Symbol energy to noise PSD level ratio in dB, e.g., -e 10.0 (default=10.0)')
    parser.add_option('-r', '--repetitions', type='int', default=100, help='Number of packets to be generated for the simulation, e.g., -r 100 (default=100)')
    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.print_help()
        raise SystemExit(1)
    fname = options.fsm_file
    esn0_db = float(options.esn0)
    rep = int(options.repetitions)
    f = trellis.fsm(fname)
    Kb = 1024 * 16
    bitspersymbol = int(round(math.log(f.I()) / math.log(2)))
    K = Kb / bitspersymbol
    modulation = fsm_utils.psk4
    dimensionality = modulation[0]
    constellation = modulation[1]
    if len(constellation) / dimensionality != f.O():
        sys.stderr.write('Incompatible FSM output cardinality and modulation size.\n')
        sys.exit(1)
    Es = 0
    for i in range(len(constellation)):
        Es = Es + constellation[i] ** 2
    Es = Es / (len(constellation) // dimensionality)
    N0 = Es / pow(10.0, esn0_db / 10.0)
    tot_b = 0
    terr_b = 0
    terr_p = 0
    for i in range(rep):
        (b, e, pattern) = run_test(f, Kb, bitspersymbol, K, dimensionality, constellation, N0, -(666 + i))
        tot_b = tot_b + b
        terr_b = terr_b + e
        terr_p = terr_p + (e != 0)
        if (i + 1) % 100 == 0:
            print(i + 1, terr_p, '%.2e' % (1.0 * terr_p / (i + 1)), tot_b, terr_b, '%.2e' % (1.0 * terr_b / tot_b))
        if e != 0:
            print('rep=', i, e)
            for k in range(Kb):
                if pattern[k] != 0:
                    print(k)
    print(rep, terr_p, '%.2e' % (1.0 * terr_p / (i + 1)), tot_b, terr_b, '%.2e' % (1.0 * terr_b / tot_b))
if __name__ == '__main__':
    main()