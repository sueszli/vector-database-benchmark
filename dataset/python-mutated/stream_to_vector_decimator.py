from gnuradio import gr
from . import blocks_python as blocks

class stream_to_vector_decimator(gr.hier_block2):
    """
    Convert the stream to a vector, decimate the vector stream to achieve the vector rate.
    """

    def __init__(self, item_size, sample_rate, vec_rate, vec_len):
        if False:
            print('Hello World!')
        '\n        Create the block chain.\n\n        Args:\n            item_size: the number of bytes per sample\n            sample_rate: the rate of incoming samples\n            vec_rate: the rate of outgoing vectors (same units as sample_rate)\n            vec_len: the length of the outgoing vectors in items\n        '
        self._vec_rate = vec_rate
        self._vec_len = vec_len
        self._sample_rate = sample_rate
        gr.hier_block2.__init__(self, 'stream_to_vector_decimator', gr.io_signature(1, 1, item_size), gr.io_signature(1, 1, item_size * vec_len))
        s2v = blocks.stream_to_vector(item_size, vec_len)
        self.one_in_n = blocks.keep_one_in_n(item_size * vec_len, 1)
        self._update_decimator()
        self.connect(self, s2v, self.one_in_n, self)

    def set_sample_rate(self, sample_rate):
        if False:
            print('Hello World!')
        '\n        Set the new sampling rate and update the decimator.\n\n        Args:\n            sample_rate: the new rate\n        '
        self._sample_rate = sample_rate
        self._update_decimator()

    def set_vec_rate(self, vec_rate):
        if False:
            return 10
        '\n        Set the new vector rate and update the decimator.\n\n        Args:\n            vec_rate: the new rate\n        '
        self._vec_rate = vec_rate
        self._update_decimator()

    def set_decimation(self, decim):
        if False:
            return 10
        '\n        Set the decimation parameter directly.\n\n        Args:\n            decim: the new decimation\n        '
        self._decim = max(1, int(round(decim)))
        self.one_in_n.set_n(self._decim)

    def _update_decimator(self):
        if False:
            while True:
                i = 10
        self.set_decimation(self._sample_rate / self._vec_len / self._vec_rate)

    def decimation(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the actual decimation.\n        '
        return self._decim

    def sample_rate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns configured sample rate.\n        '
        return self._sample_rate

    def frame_rate(self):
        if False:
            return 10
        '\n        Returns actual frame rate\n        '
        return self._sample_rate / self._vec_len / self._decim