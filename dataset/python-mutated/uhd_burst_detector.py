from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import filter, analog, blocks
from gnuradio import uhd
from gnuradio.fft import window
from gnuradio.eng_arg import eng_float
from gnuradio.filter import firdes
from argparse import ArgumentParser

class uhd_burst_detector(gr.top_block):

    def __init__(self, uhd_address, options):
        if False:
            print('Hello World!')
        gr.top_block.__init__(self)
        self.uhd_addr = uhd_address
        self.freq = options.freq
        self.samp_rate = options.samp_rate
        self.gain = options.gain
        self.threshold = options.threshold
        self.trigger = options.trigger
        self.uhd_src = uhd.single_usrp_source(device_addr=self.uhd_addr, stream_args=uhd.stream_args('fc32'))
        self.uhd_src.set_samp_rate(self.samp_rate)
        self.uhd_src.set_center_freq(self.freq, 0)
        self.uhd_src.set_gain(self.gain, 0)
        taps = firdes.low_pass_2(1, 1, 0.4, 0.1, 60)
        self.chanfilt = filter.fir_filter_ccc(10, taps)
        self.tagger = blocks.burst_tagger(gr.sizeof_gr_complex)
        data = 1000 * [0] + 1000 * [1]
        self.signal = blocks.vector_source_s(data, True)
        self.det = analog.simple_squelch_cc(self.threshold, 0.01)
        self.c2m = blocks.complex_to_mag_squared()
        self.avg = filter.single_pole_iir_filter_ff(0.01)
        self.scale = blocks.multiply_const_ff(2 ** 16)
        self.f2s = blocks.float_to_short()
        self.fsnk = blocks.tagged_file_sink(gr.sizeof_gr_complex, self.samp_rate)
        self.connect((self.uhd_src, 0), (self.tagger, 0))
        self.connect((self.tagger, 0), (self.fsnk, 0))
        if self.trigger:
            self.connect((self.signal, 0), (self.tagger, 1))
        else:
            self.connect(self.uhd_src, self.det)
            self.connect(self.det, self.c2m, self.avg, self.scale, self.f2s)
            self.connect(self.f2s, (self.tagger, 1))

    def set_samp_rate(self, samp_rate):
        if False:
            return 10
        self.samp_rate = samp_rate
        self.uhd_src_0.set_samp_rate(self.samp_rate)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-a', '--address', default='addr=192.168.10.2', help='select address of the device [default=%(default)r]')
    parser.add_argument('-f', '--freq', type=eng_float, default=450000000.0, help='set frequency to FREQ', metavar='FREQ')
    parser.add_argument('-g', '--gain', type=eng_float, default=0, help='set gain in dB [default=%(default)r]')
    parser.add_argument('-R', '--samp-rate', type=eng_float, default=200000, help='set USRP sample rate [default=%(default)r]')
    parser.add_argument('-t', '--threshold', type=float, default=-60, help='Set the detection power threshold (dBm) [default=%(default)r')
    parser.add_argument('-T', '--trigger', action='store_true', default=False, help='Use internal trigger instead of detector [default=%(default)r]')
    args = parser.parse_args()
    uhd_addr = args.address
    tb = uhd_burst_detector(uhd_addr, args)
    tb.run()