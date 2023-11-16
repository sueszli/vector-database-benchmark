import time
import random
from argparse import ArgumentParser
from gnuradio import gr
from gnuradio import blocks, filter
from gnuradio.eng_arg import eng_float, intx

def make_random_complex_tuple(L):
    if False:
        print('Hello World!')
    result = []
    for x in range(L):
        result.append(complex(random.uniform(-1000, 1000), random.uniform(-1000, 1000)))
    return tuple(result)

def benchmark(name, creator, dec, ntaps, total_test_size, block_size):
    if False:
        print('Hello World!')
    block_size = 32768
    tb = gr.top_block()
    taps = make_random_complex_tuple(ntaps)
    src = blocks.vector_source_c(make_random_complex_tuple(block_size), True)
    head = blocks.head(gr.sizeof_gr_complex, int(total_test_size))
    op = creator(dec, taps)
    dst = blocks.null_sink(gr.sizeof_gr_complex)
    tb.connect(src, head, op, dst)
    start = time.time()
    tb.run()
    stop = time.time()
    delta = stop - start
    print('%16s: taps: %4d  input: %4g, time: %6.3f  taps/sec: %10.4g' % (name, ntaps, total_test_size, delta, ntaps * total_test_size / delta))

def main():
    if False:
        print('Hello World!')
    parser = ArgumentParser()
    parser.add_argument('-n', '--ntaps', type=int, default=256)
    parser.add_argument('-t', '--total-input-size', type=eng_float, default=40000000.0)
    parser.add_argument('-b', '--block-size', type=intx, default=50000)
    parser.add_argument('-d', '--decimation', type=int, default=1)
    args = parser.parse_args()
    benchmark('filter.fir_filter_ccc', filter.fir_filter_ccc, args.decimation, args.ntaps, args.total_input_size, args.block_size)
    benchmark('filter.fft_filter_ccc', filter.fft_filter_ccc, args.decimation, args.ntaps, args.total_input_size, args.block_size)
if __name__ == '__main__':
    main()