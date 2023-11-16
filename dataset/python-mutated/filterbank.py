from gnuradio import gr
from gnuradio import fft
from gnuradio import blocks
from .filter_python import fft_filter_ccc

def _generate_synthesis_taps(mpoints):
    if False:
        i = 10
        return i + 15
    return []

def _split_taps(taps, mpoints):
    if False:
        while True:
            i = 10
    assert len(taps) % mpoints == 0
    result = [list() for x in range(mpoints)]
    for i in range(len(taps)):
        result[i % mpoints].append(taps[i])
    return [tuple(x) for x in result]

class synthesis_filterbank(gr.hier_block2):
    """
    Uniformly modulated polyphase DFT filter bank: synthesis

    See http://cnx.org/content/m10424/latest
    """

    def __init__(self, mpoints, taps=None):
        if False:
            print('Hello World!')
        '\n        Takes M complex streams in, produces single complex stream out\n        that runs at M times the input sample rate\n\n        Args:\n            mpoints: number of freq bins/interpolation factor/subbands\n            taps: filter taps for subband filter\n\n        The channel spacing is equal to the input sample rate.\n        The total bandwidth and output sample rate are equal the input\n        sample rate * nchannels.\n\n        Output stream to frequency mapping:\n\n          channel zero is at zero frequency.\n\n          if mpoints is odd:\n\n            Channels with increasing positive frequencies come from\n            channels 1 through (N-1)/2.\n\n            Channel (N+1)/2 is the maximum negative frequency, and\n            frequency increases through N-1 which is one channel lower\n            than the zero frequency.\n\n          if mpoints is even:\n\n            Channels with increasing positive frequencies come from\n            channels 1 through (N/2)-1.\n\n            Channel (N/2) is evenly split between the max positive and\n            negative bins.\n\n            Channel (N/2)+1 is the maximum negative frequency, and\n            frequency increases through N-1 which is one channel lower\n            than the zero frequency.\n\n            Channels near the frequency extremes end up getting cut\n            off by subsequent filters and therefore have diminished\n            utility.\n        '
        item_size = gr.sizeof_gr_complex
        gr.hier_block2.__init__(self, 'synthesis_filterbank', gr.io_signature(mpoints, mpoints, item_size), gr.io_signature(1, 1, item_size))
        if taps is None:
            taps = _generate_synthesis_taps(mpoints)
        r = len(taps) % mpoints
        if r != 0:
            taps = taps + (mpoints - r) * (0,)
        sub_taps = _split_taps(taps, mpoints)
        self.ss2v = blocks.streams_to_vector(item_size, mpoints)
        self.ifft = fft.fft_vcc(mpoints, False, [])
        self.v2ss = blocks.vector_to_streams(item_size, mpoints)
        self.ss2s = blocks.streams_to_stream(item_size, mpoints)
        for i in range(mpoints):
            self.connect((self, i), (self.ss2v, i))
        self.connect(self.ss2v, self.ifft, self.v2ss)
        for i in range(mpoints):
            f = fft_filter_ccc(1, sub_taps[i])
            self.connect((self.v2ss, i), f)
            self.connect(f, (self.ss2s, i))
            self.connect(self.ss2s, self)

class analysis_filterbank(gr.hier_block2):
    """
    Uniformly modulated polyphase DFT filter bank: analysis

    See http://cnx.org/content/m10424/latest
    """

    def __init__(self, mpoints, taps=None):
        if False:
            i = 10
            return i + 15
        '\n        Takes 1 complex stream in, produces M complex streams out\n        that runs at 1/M times the input sample rate\n\n        Args:\n            mpoints: number of freq bins/interpolation factor/subbands\n            taps: filter taps for subband filter\n\n        Same channel to frequency mapping as described above.\n        '
        item_size = gr.sizeof_gr_complex
        gr.hier_block2.__init__(self, 'analysis_filterbank', gr.io_signature(1, 1, item_size), gr.io_signature(mpoints, mpoints, item_size))
        if taps is None:
            taps = _generate_synthesis_taps(mpoints)
        r = len(taps) % mpoints
        if r != 0:
            taps = taps + (mpoints - r) * (0,)
        sub_taps = _split_taps(taps, mpoints)
        self.s2ss = blocks.stream_to_streams(item_size, mpoints)
        self.ss2v = blocks.streams_to_vector(item_size, mpoints)
        self.fft = fft.fft_vcc(mpoints, True, [])
        self.v2ss = blocks.vector_to_streams(item_size, mpoints)
        self.connect(self, self.s2ss)
        for i in range(mpoints):
            f = fft_filter_ccc(1, sub_taps[mpoints - i - 1])
            self.connect((self.s2ss, i), f)
            self.connect(f, (self.ss2v, i))
            self.connect((self.v2ss, i), (self, i))
        self.connect(self.ss2v, self.fft, self.v2ss)