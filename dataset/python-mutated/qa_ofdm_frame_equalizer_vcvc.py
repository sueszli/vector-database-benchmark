import numpy
from gnuradio import gr, gr_unittest, digital, blocks
import pmt

class qa_ofdm_frame_equalizer_vcvc(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()
        self.tsb_key = 'tsb_key'

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_001_simple(self):
        if False:
            i = 10
            return i + 15
        " Very simple functionality testing:\n        - static equalizer\n        - init channel state with all ones\n        - transmit all ones\n        - make sure we rx all ones\n        - Tag check: put in frame length tag and one other random tag,\n                     make sure they're propagated\n        "
        fft_len = 8
        equalizer = digital.ofdm_equalizer_static(fft_len)
        n_syms = 3
        tx_data = [1] * fft_len * n_syms
        chan_tag = gr.tag_t()
        chan_tag.offset = 0
        chan_tag.key = pmt.string_to_symbol('ofdm_sync_chan_taps')
        chan_tag.value = pmt.init_c32vector(fft_len, (1,) * fft_len)
        random_tag = gr.tag_t()
        random_tag.offset = 1
        random_tag.key = pmt.string_to_symbol('foo')
        random_tag.value = pmt.from_long(42)
        src = blocks.vector_source_c(tx_data, False, fft_len, (chan_tag, random_tag))
        eq = digital.ofdm_frame_equalizer_vcvc(equalizer.base(), 0, self.tsb_key)
        sink = blocks.tsb_vector_sink_c(fft_len, tsb_key=self.tsb_key)
        self.tb.connect(src, blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, fft_len, n_syms, self.tsb_key), eq, sink)
        self.tb.run()
        self.assertEqual(tx_data, sink.data()[0])
        tag_dict = dict()
        for tag in sink.tags():
            ptag = gr.tag_to_python(tag)
            tag_dict[ptag.key] = ptag.value
        expected_dict = {'foo': 42}
        self.assertEqual(tag_dict, expected_dict)

    def test_001b_simple_skip_nothing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Same as before, but put a skip-header in there\n        '
        fft_len = 8
        equalizer = digital.ofdm_equalizer_static(fft_len, symbols_skipped=1)
        n_syms = 3
        tx_data = [1] * fft_len * n_syms
        chan_tag = gr.tag_t()
        chan_tag.offset = 0
        chan_tag.key = pmt.string_to_symbol('ofdm_sync_chan_taps')
        chan_tag.value = pmt.init_c32vector(fft_len, (1,) * fft_len)
        src = blocks.vector_source_c(tx_data, False, fft_len, (chan_tag,))
        eq = digital.ofdm_frame_equalizer_vcvc(equalizer.base(), 0, self.tsb_key)
        sink = blocks.tsb_vector_sink_c(fft_len, tsb_key=self.tsb_key)
        self.tb.connect(src, blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, fft_len, n_syms, self.tsb_key), eq, sink)
        self.tb.run()
        self.assertEqual(tx_data, sink.data()[0])

    def test_001c_carrier_offset_no_cp(self):
        if False:
            return 10
        '\n        Same as before, but put a carrier offset in there\n        '
        fft_len = 8
        cp_len = 0
        n_syms = 1
        carr_offset = 1
        occupied_carriers = ((-2, -1, 1, 2),)
        tx_data = (0, 0, 0, -1j, -1j, 0, -1j, -1j)
        rx_expected = (0, 0, 1, 1, 0, 1, 1, 0) * n_syms
        equalizer = digital.ofdm_equalizer_static(fft_len, occupied_carriers)
        chan_tag = gr.tag_t()
        chan_tag.offset = 0
        chan_tag.key = pmt.string_to_symbol('ofdm_sync_chan_taps')
        chan_tag.value = pmt.init_c32vector(fft_len, (0, 0, -1j, -1j, 0, -1j, -1j, 0))
        offset_tag = gr.tag_t()
        offset_tag.offset = 0
        offset_tag.key = pmt.string_to_symbol('ofdm_sync_carr_offset')
        offset_tag.value = pmt.from_long(carr_offset)
        src = blocks.vector_source_c(tx_data, False, fft_len, (chan_tag, offset_tag))
        eq = digital.ofdm_frame_equalizer_vcvc(equalizer.base(), cp_len, self.tsb_key)
        sink = blocks.tsb_vector_sink_c(fft_len, tsb_key=self.tsb_key)
        self.tb.connect(src, blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, fft_len, n_syms, self.tsb_key), eq, sink)
        self.tb.run()
        self.assertComplexTuplesAlmostEqual(rx_expected, sink.data()[0], places=4)

    def test_001c_carrier_offset_cp(self):
        if False:
            return 10
        '\n        Same as before, but put a carrier offset in there and a CP\n        '
        fft_len = 8
        cp_len = 2
        n_syms = 3
        occupied_carriers = ((-2, -1, 1, 2),)
        carr_offset = -1
        tx_data = (0, -1j, -1j, 0, -1j, -1j, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0, 0)
        rx_expected = (0, 0, 1, 1, 0, 1, 1, 0) * n_syms
        equalizer = digital.ofdm_equalizer_static(fft_len, occupied_carriers)
        chan_tag = gr.tag_t()
        chan_tag.offset = 0
        chan_tag.key = pmt.string_to_symbol('ofdm_sync_chan_taps')
        chan_tag.value = pmt.init_c32vector(fft_len, (0, 0, 1, 1, 0, 1, 1, 0))
        offset_tag = gr.tag_t()
        offset_tag.offset = 0
        offset_tag.key = pmt.string_to_symbol('ofdm_sync_carr_offset')
        offset_tag.value = pmt.from_long(carr_offset)
        src = blocks.vector_source_c(tx_data, False, fft_len, (chan_tag, offset_tag))
        eq = digital.ofdm_frame_equalizer_vcvc(equalizer.base(), cp_len, self.tsb_key)
        sink = blocks.tsb_vector_sink_c(fft_len, tsb_key=self.tsb_key)
        self.tb.connect(src, blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, fft_len, n_syms, self.tsb_key), eq, sink)
        self.tb.run()
        self.assertComplexTuplesAlmostEqual(rx_expected, sink.data()[0], places=4)

    def test_002_static(self):
        if False:
            print('Hello World!')
        '\n        - Add a simple channel\n        - Make symbols QPSK\n        '
        fft_len = 8
        tx_data = [-1, -1, 1, 2, -1, 3, 0, -1, -1, -1, 0, 2, -1, 2, 0, -1, -1, -1, 3, 0, -1, 1, 0, -1, -1, -1, 1, 1, -1, 0, 2, -1]
        cnst = digital.constellation_qpsk()
        tx_signal = [cnst.map_to_points_v(x)[0] if x != -1 else 0 for x in tx_data]
        occupied_carriers = ((1, 2, 6, 7),)
        pilot_carriers = ((), (), (1, 2, 6, 7), ())
        pilot_symbols = ([], [], [cnst.map_to_points_v(x)[0] for x in (1, 0, 3, 0)], [])
        equalizer = digital.ofdm_equalizer_static(fft_len, occupied_carriers, pilot_carriers, pilot_symbols)
        channel = [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0]
        for idx in range(fft_len, 2 * fft_len):
            channel[idx] = channel[idx - fft_len] * numpy.exp(1j * 0.1 * numpy.pi * (numpy.random.rand() - 0.5))
        chan_tag = gr.tag_t()
        chan_tag.offset = 0
        chan_tag.key = pmt.string_to_symbol('ofdm_sync_chan_taps')
        chan_tag.value = pmt.init_c32vector(fft_len, channel[:fft_len])
        src = blocks.vector_source_c(numpy.multiply(tx_signal, channel), False, fft_len, (chan_tag,))
        sink = blocks.tsb_vector_sink_c(vlen=fft_len, tsb_key=self.tsb_key)
        eq = digital.ofdm_frame_equalizer_vcvc(equalizer.base(), 0, self.tsb_key, True)
        self.tb.connect(src, blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, fft_len, len(tx_data) // fft_len, self.tsb_key), eq, sink)
        self.tb.run()
        rx_data = [cnst.decision_maker_v((x,)) if x != 0 else -1 for x in sink.data()[0]]
        self.assertEqual(tx_data, rx_data)
        tag_dict = dict()
        for tag in sink.tags():
            ptag = gr.tag_to_python(tag)
            tag_dict[ptag.key] = ptag.value
            if ptag.key == 'ofdm_sync_chan_taps':
                tag_dict[ptag.key] = list(pmt.c32vector_elements(tag.value))
            else:
                tag_dict[ptag.key] = pmt.to_python(tag.value)
        expected_dict = {'ofdm_sync_chan_taps': channel[-fft_len:]}
        self.assertTrue(numpy.allclose(tag_dict['ofdm_sync_chan_taps'], expected_dict['ofdm_sync_chan_taps']))
        expected_dict['ofdm_sync_chan_taps'] = tag_dict['ofdm_sync_chan_taps']
        self.assertEqual(tag_dict, expected_dict)

    def test_002_static_wo_tags(self):
        if False:
            print('Hello World!')
        ' Same as before, but the input stream has no tag.\n        We specify the frame size in the constructor.\n        We also specify a tag key, so the output stream *should* have\n        a TSB tag.\n        '
        fft_len = 8
        n_syms = 4
        tx_data = [-1, -1, 1, 2, -1, 3, 0, -1, -1, -1, 0, 2, -1, 2, 0, -1, -1, -1, 3, 0, -1, 1, 0, -1, -1, -1, 1, 1, -1, 0, 2, -1]
        cnst = digital.constellation_qpsk()
        tx_signal = [cnst.map_to_points_v(x)[0] if x != -1 else 0 for x in tx_data]
        occupied_carriers = ((1, 2, 6, 7),)
        pilot_carriers = ((), (), (1, 2, 6, 7), ())
        pilot_symbols = ([], [], [cnst.map_to_points_v(x)[0] for x in (1, 0, 3, 0)], [])
        equalizer = digital.ofdm_equalizer_static(fft_len, occupied_carriers, pilot_carriers, pilot_symbols)
        channel = [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0]
        for idx in range(fft_len, 2 * fft_len):
            channel[idx] = channel[idx - fft_len] * numpy.exp(1j * 0.1 * numpy.pi * (numpy.random.rand() - 0.5))
            idx2 = idx + 2 * fft_len
            channel[idx2] = channel[idx2] * numpy.exp(1j * 0 * numpy.pi * (numpy.random.rand() - 0.5))
        src = blocks.vector_source_c(numpy.multiply(tx_signal, channel), False, fft_len)
        eq = digital.ofdm_frame_equalizer_vcvc(equalizer.base(), 0, self.tsb_key, False, n_syms)
        sink = blocks.tsb_vector_sink_c(vlen=fft_len, tsb_key=self.tsb_key)
        self.tb.connect(src, blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, fft_len, len(tx_data) // fft_len, self.tsb_key), eq, sink)
        self.tb.run()
        rx_data = [cnst.decision_maker_v((x,)) if x != 0 else -1 for x in sink.data()[0]]
        self.assertEqual(tx_data, rx_data)
        packets = sink.data()
        self.assertEqual(len(packets), 1)
        self.assertEqual(len(packets[0]), len(tx_data))

    def test_002_static_wo_tags_2(self):
        if False:
            print('Hello World!')
        fft_len = 8
        tx_data = [-1, -1, 1, 2, -1, 3, 0, -1, -1, -1, 0, 2, -1, 2, 0, -1, -1, -1, 3, 0, -1, 1, 0, -1, -1, -1, 1, 1, -1, 0, 2, -1]
        cnst = digital.constellation_qpsk()
        tx_signal = [cnst.map_to_points_v(x)[0] if x != -1 else 0 for x in tx_data]
        occupied_carriers = ((1, 2, 6, 7),)
        pilot_carriers = ((), (), (1, 2, 6, 7), ())
        pilot_symbols = ([], [], [cnst.map_to_points_v(x)[0] for x in (1, 0, 3, 0)], [])
        equalizer = digital.ofdm_equalizer_static(fft_len, occupied_carriers, pilot_carriers, pilot_symbols)
        channel = [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0]
        for idx in range(fft_len, 2 * fft_len):
            channel[idx] = channel[idx - fft_len] * numpy.exp(1j * 0.1 * numpy.pi * (numpy.random.rand() - 0.5))
            idx2 = idx + 2 * fft_len
            channel[idx2] = channel[idx2] * numpy.exp(1j * 0 * numpy.pi * (numpy.random.rand() - 0.5))
        src = blocks.vector_source_c(numpy.multiply(tx_signal, channel), False, fft_len)
        sink = blocks.vector_sink_c(fft_len)
        eq = digital.ofdm_frame_equalizer_vcvc(equalizer.base(), 0, '', False, 4)
        self.tb.connect(src, eq, sink)
        self.tb.run()
        rx_data = [cnst.decision_maker_v((x,)) if x != 0 else -1 for x in sink.data()]
        self.assertEqual(tx_data, rx_data)

    def test_002_simpledfe(self):
        if False:
            i = 10
            return i + 15
        ' Use the simple DFE equalizer. '
        fft_len = 8
        tx_data = [-1, -1, 1, 2, -1, 3, 0, -1, -1, -1, 0, 2, -1, 2, 0, -1, -1, -1, 3, 0, -1, 1, 0, -1, -1, -1, 1, 1, -1, 0, 2, -1]
        cnst = digital.constellation_qpsk()
        tx_signal = [cnst.map_to_points_v(x)[0] if x != -1 else 0 for x in tx_data]
        occupied_carriers = ((1, 2, 6, 7),)
        pilot_carriers = ((), (), (1, 2, 6, 7), ())
        pilot_symbols = ([], [], [cnst.map_to_points_v(x)[0] for x in (1, 0, 3, 0)], [])
        equalizer = digital.ofdm_equalizer_simpledfe(fft_len, cnst.base(), occupied_carriers, pilot_carriers, pilot_symbols, 0, 0.01)
        equalizer_soft = digital.ofdm_equalizer_simpledfe(fft_len, cnst.base(), occupied_carriers, pilot_carriers, pilot_symbols, 0, 0.01, enable_soft_output=True)
        channel = [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0, 0, 0, 1j, 1j, 0, 1j, 1j, 0]
        for idx in range(fft_len, 2 * fft_len):
            channel[idx] = channel[idx - fft_len] * numpy.exp(1j * 0.1 * numpy.pi * (numpy.random.rand() - 0.5))
            idx2 = idx + 2 * fft_len
            channel[idx2] = channel[idx2] * numpy.exp(1j * 0 * numpy.pi * (numpy.random.rand() - 0.5))
        chan_tag = gr.tag_t()
        chan_tag.offset = 0
        chan_tag.key = pmt.string_to_symbol('ofdm_sync_chan_taps')
        chan_tag.value = pmt.init_c32vector(fft_len, channel[:fft_len])
        src = blocks.vector_source_c(numpy.multiply(tx_signal, channel), False, fft_len, (chan_tag,))
        eq = digital.ofdm_frame_equalizer_vcvc(equalizer.base(), 0, self.tsb_key, True)
        eq_soft = digital.ofdm_frame_equalizer_vcvc(equalizer_soft.base(), 0, self.tsb_key, True)
        sink = blocks.tsb_vector_sink_c(fft_len, tsb_key=self.tsb_key)
        sink_soft = blocks.tsb_vector_sink_c(fft_len, tsb_key=self.tsb_key)
        stream_to_tagged = blocks.stream_to_tagged_stream(gr.sizeof_gr_complex, fft_len, len(tx_data) // fft_len, self.tsb_key)
        self.tb.connect(src, stream_to_tagged, eq, sink)
        self.tb.connect(stream_to_tagged, eq_soft, sink_soft)
        self.tb.run()
        out_syms = numpy.array(sink.data()[0])
        out_syms_soft = numpy.array(sink_soft.data()[0])

        def demod(syms):
            if False:
                i = 10
                return i + 15
            return [cnst.decision_maker_v((x,)) if x != 0 else -1 for x in syms]
        rx_data = demod(out_syms)
        rx_data_soft = demod(out_syms_soft)
        self.assertEqual(tx_data, rx_data)
        self.assertEqual(rx_data, rx_data_soft)
        self.assertFalse(numpy.allclose(out_syms, out_syms_soft))
        self.assertEqual(len(sink.tags()), 1)
        tag = sink.tags()[0]
        self.assertEqual(pmt.symbol_to_string(tag.key), 'ofdm_sync_chan_taps')
        self.assertComplexTuplesAlmostEqual(list(pmt.c32vector_elements(tag.value)), channel[-fft_len:], places=1)
if __name__ == '__main__':
    gr_unittest.run(qa_ofdm_frame_equalizer_vcvc)