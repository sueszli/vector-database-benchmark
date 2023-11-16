from gnuradio import gr, eng_notation, filter
from gnuradio import audio
from gnuradio import analog
from gnuradio import blocks
from gnuradio.eng_arg import eng_float, intx
from argparse import ArgumentParser
import sys
import math

class wfm_rx_block(gr.top_block):

    def __init__(self):
        if False:
            while True:
                i = 10
        gr.top_block.__init__(self)
        parser = ArgumentParser(description='Decode WFM signal into WAV file.')
        parser.add_argument('-V', '--volume', type=eng_float, help='Volume (dB) <%r, %r> (default is midpoint)' % self.volume_range()[:2])
        parser.add_argument('input_file', help='Input file (complex samples)')
        parser.add_argument('output_file', help='Output WAV file')
        args = parser.parse_args()
        self.vol = 0
        self.src = blocks.file_source(gr.sizeof_gr_complex, args.input_file, False)
        adc_rate = 64000000.0
        usrp_decim = 200
        usrp_rate = adc_rate / usrp_decim
        chanfilt_decim = 1
        demod_rate = usrp_rate / chanfilt_decim
        audio_decimation = 10
        audio_rate = demod_rate / audio_decimation
        chan_filt_coeffs = filter.optfir.low_pass(1, usrp_rate, 80000.0, 115000.0, 0.1, 60)
        chan_filt = filter.fir_filter_ccf(chanfilt_decim, chan_filt_coeffs)
        self.guts = analog.wfm_rcv_pll(demod_rate, audio_decimation)
        self.volume_control_l = blocks.multiply_const_ff(self.vol)
        self.volume_control_r = blocks.multiply_const_ff(self.vol)
        if 1:
            sink = blocks.wavfile_sink(args.output_file, 2, int(audio_rate), blocks.FORMAT_WAV, blocks.FORMAT_PCM_16)
        else:
            sink = audio.sink(int(audio_rate), args.audio_output, False)
        self.connect(self.src, chan_filt, self.guts)
        self.connect((self.guts, 0), self.volume_control_l, (sink, 0))
        self.connect((self.guts, 1), self.volume_control_r, (sink, 1))
        if args.volume is None:
            g = self.volume_range()
            args.volume = float(g[0] + g[1]) / 2
        self.set_vol(args.volume)

    def set_vol(self, vol):
        if False:
            for i in range(10):
                print('nop')
        g = self.volume_range()
        self.vol = max(g[0], min(g[1], vol))
        self.volume_control_l.set_k(10 ** (self.vol / 10))
        self.volume_control_r.set_k(10 ** (self.vol / 10))

    def volume_range(self):
        if False:
            while True:
                i = 10
        return (-20.0, 0.0, 0.5)
if __name__ == '__main__':
    tb = wfm_rx_block()
    try:
        tb.run()
    except KeyboardInterrupt:
        pass