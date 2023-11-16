import time
import random
import numpy
from gnuradio import gr
from gnuradio import gr_unittest
from gnuradio import digital
from gnuradio import blocks
import pmt

def make_tag(key, value, offset):
    if False:
        while True:
            i = 10
    'Create a gr.tag_t() from key, value, offset.'
    tag = gr.tag_t()
    tag.offset = offset
    tag.key = pmt.string_to_symbol(key)
    tag.value = pmt.to_pmt(value)
    return tag

class HeaderToMessageBlock(gr.sync_block):
    """
    Helps with testing the HPD. Receives a header, stores it, posts
    a predetermined message. forecast() is not currently working in
    Python, so use a local buffer to accumulate header data.
    """

    def __init__(self, itemsize, header_len, messages):
        if False:
            i = 10
            return i + 15
        gr.sync_block.__init__(self, name='HeaderToMessageBlock', in_sig=[itemsize], out_sig=[itemsize])
        self.header_len = header_len
        self.message_port_register_out(pmt.intern('header_data'))
        self.messages = messages
        self.msg_count = 0
        self.buf = []

    def work(self, input_items, output_items):
        if False:
            for i in range(10):
                print('nop')
        'Where the magic happens.'
        self.buf.extend(input_items[0])
        for _ in range(len(self.buf) // self.header_len):
            msg = self.messages[self.msg_count] or False
            self.message_port_pub(pmt.intern('header_data'), pmt.to_pmt(msg))
            self.msg_count += 1
            del self.buf[:self.header_len]
        output_items[0][:] = input_items[0][:]
        return len(input_items[0])

class qa_header_payload_demux(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        'Runs before every test.'
        self.tb = gr.top_block()
        random.seed(0)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Runs after every test.'
        self.tb = None

    def connect_all_blocks(self, data_src, trigger_src, hpd, mock_header_demod, payload_sink, header_sink):
        if False:
            print('Hello World!')
        '\n        Connect the standard HPD test flowgraph\n        '
        self.tb.connect(data_src, (hpd, 0))
        if trigger_src is not None:
            self.tb.connect(trigger_src, (hpd, 1))
        self.tb.connect((hpd, 0), mock_header_demod)
        self.tb.connect(mock_header_demod, header_sink)
        self.tb.msg_connect(mock_header_demod, 'header_data', hpd, 'header_data')
        self.tb.connect((hpd, 1), payload_sink)

    def run_tb(self, payload_sink, payload_len, header_sink, header_len, timeout=30):
        if False:
            i = 10
            return i + 15
        'Execute self.tb'
        stop_time = time.time() + timeout
        self.tb.start()
        while (len(payload_sink.data()) < payload_len or len(header_sink.data()) < header_len) and time.time() < stop_time:
            time.sleep(0.2)
        self.tb.stop()
        self.tb.wait()

    def test_001_t(self):
        if False:
            print('Hello World!')
        ' Simplest possible test: put in zeros, then header,\n        then payload, trigger signal, try to demux.\n        The return signal from the header parser is faked via _post()\n        Add in some tags for fun.\n        '
        n_zeros = 1
        header = (1, 2, 3)
        payload = tuple(range(5, 20))
        data_signal = (0,) * n_zeros + header + payload
        trigger_signal = [0] * len(data_signal)
        trigger_signal[n_zeros] = 1
        testtag1 = make_tag('tag1', 0, 0)
        testtag2 = make_tag('tag2', 23, n_zeros)
        testtag3 = make_tag('tag3', 42, n_zeros + len(header) - 1)
        testtag4 = make_tag('tag4', 314, n_zeros + len(header) + 3)
        data_src = blocks.vector_source_f(data_signal, False, tags=(testtag1, testtag2, testtag3, testtag4))
        trigger_src = blocks.vector_source_b(trigger_signal, False)
        hpd = digital.header_payload_demux(len(header), 1, 0, 'frame_len', 'detect', False, gr.sizeof_float)
        mock_header_demod = HeaderToMessageBlock(numpy.float32, len(header), [len(payload)])
        self.assertEqual(pmt.length(hpd.message_ports_in()), 2)
        payload_sink = blocks.vector_sink_f()
        header_sink = blocks.vector_sink_f()
        self.connect_all_blocks(data_src, trigger_src, hpd, mock_header_demod, payload_sink, header_sink)
        self.run_tb(payload_sink, len(payload), header_sink, len(header))
        self.assertEqual(header_sink.data(), list(header))
        self.assertEqual(payload_sink.data(), list(payload))
        ptags_header = []
        for tag in header_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_header.append({'key': ptag.key, 'offset': ptag.offset})
        expected_tags_header = [{'key': 'tag2', 'offset': 0}, {'key': 'tag3', 'offset': 2}]
        self.assertEqual(expected_tags_header, ptags_header)
        ptags_payload = []
        for tag in payload_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_payload.append({'key': ptag.key, 'offset': ptag.offset})
        expected_tags_payload = [{'key': 'frame_len', 'offset': 0}, {'key': 'tag4', 'offset': 3}]
        self.assertEqual(expected_tags_payload, ptags_payload)

    def test_001_t_tags(self):
        if False:
            for i in range(10):
                print('nop')
        ' Like the previous test, but use a trigger tag instead of\n        a trigger signal.\n        '
        n_zeros = 1
        header = (1, 2, 3)
        payload = tuple(range(5, 20))
        data_signal = (0,) * n_zeros + header + payload
        trigger_tag = make_tag('detect', True, n_zeros)
        testtag1 = make_tag('tag1', 0, 0)
        testtag2 = make_tag('tag2', 23, n_zeros)
        testtag3 = make_tag('tag3', 42, n_zeros + len(header) - 1)
        testtag4 = make_tag('tag4', 314, n_zeros + len(header) + 3)
        data_src = blocks.vector_source_f(data_signal, False, tags=(trigger_tag, testtag1, testtag2, testtag3, testtag4))
        hpd = digital.header_payload_demux(len(header), 1, 0, 'frame_len', 'detect', False, gr.sizeof_float)
        self.assertEqual(pmt.length(hpd.message_ports_in()), 2)
        header_sink = blocks.vector_sink_f()
        payload_sink = blocks.vector_sink_f()
        mock_header_demod = HeaderToMessageBlock(numpy.float32, len(header), [len(payload)])
        self.connect_all_blocks(data_src, None, hpd, mock_header_demod, payload_sink, header_sink)
        self.run_tb(payload_sink, len(payload), header_sink, len(header))
        self.assertEqual(header_sink.data(), list(header))
        self.assertEqual(payload_sink.data(), list(payload))
        ptags_header = []
        for tag in header_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_header.append({'key': ptag.key, 'offset': ptag.offset})
        expected_tags_header = [{'key': 'tag2', 'offset': 0}, {'key': 'tag3', 'offset': 2}]
        self.assertEqual(expected_tags_header, ptags_header)
        ptags_payload = []
        for tag in payload_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_payload.append({'key': ptag.key, 'offset': ptag.offset})
        expected_tags_payload = [{'key': 'frame_len', 'offset': 0}, {'key': 'tag4', 'offset': 3}]
        self.assertEqual(expected_tags_payload, ptags_payload)

    def test_001_headerpadding(self):
        if False:
            while True:
                i = 10
        ' Like test 1, but with header padding. '
        n_zeros = 3
        header = [1, 2, 3]
        header_padding = 1
        payload = list(range(5, 20))
        data_signal = [0] * n_zeros + header + payload
        trigger_signal = [0] * len(data_signal)
        trigger_signal[n_zeros] = 1
        testtag1 = make_tag('tag1', 0, 0)
        testtag2 = make_tag('tag2', 23, n_zeros)
        testtag3 = make_tag('tag3', 42, n_zeros + len(header) - 1)
        testtag4 = make_tag('tag4', 314, n_zeros + len(header) + 3)
        data_src = blocks.vector_source_f(data_signal, False, tags=(testtag1, testtag2, testtag3, testtag4))
        trigger_src = blocks.vector_source_b(trigger_signal, False)
        hpd = digital.header_payload_demux(len(header), 1, 0, 'frame_len', 'detect', False, gr.sizeof_float, '', 1.0, (), header_padding)
        mock_header_demod = HeaderToMessageBlock(numpy.float32, len(header), [len(payload)])
        header_sink = blocks.vector_sink_f()
        payload_sink = blocks.vector_sink_f()
        self.connect_all_blocks(data_src, trigger_src, hpd, mock_header_demod, payload_sink, header_sink)
        self.run_tb(payload_sink, len(payload), header_sink, len(header) + 2)
        self.assertEqual(header_sink.data(), [0] + header + [payload[0]])
        self.assertEqual(payload_sink.data(), payload)
        ptags_header = []
        for tag in header_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_header.append({'key': ptag.key, 'offset': ptag.offset})
        expected_tags_header = [{'key': 'tag2', 'offset': 1}, {'key': 'tag3', 'offset': 3}]
        self.assertEqual(expected_tags_header, ptags_header)
        ptags_payload = []
        for tag in payload_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_payload.append({'key': ptag.key, 'offset': ptag.offset})
        expected_tags_payload = [{'key': 'frame_len', 'offset': 0}, {'key': 'tag4', 'offset': 3}]
        self.assertEqual(expected_tags_payload, ptags_payload)

    def test_001_headerpadding_payload_offset(self):
        if False:
            while True:
                i = 10
        ' Like test 1, but with header padding + payload offset. '
        n_zeros = 3
        header = [1, 2, 3]
        header_padding = 1
        payload_offset = -1
        payload = list(range(5, 20))
        data_signal = [0] * n_zeros + header + payload + [0] * 100
        trigger_signal = [0] * len(data_signal)
        trigger_signal[n_zeros] = 1
        testtag4 = make_tag('tag4', 314, n_zeros + len(header) + 3)
        data_src = blocks.vector_source_f(data_signal, False, tags=(testtag4,))
        trigger_src = blocks.vector_source_b(trigger_signal, False)
        hpd = digital.header_payload_demux(len(header), 1, 0, 'frame_len', 'detect', False, gr.sizeof_float, '', 1.0, (), header_padding)
        self.assertEqual(pmt.length(hpd.message_ports_in()), 2)
        header_sink = blocks.vector_sink_f()
        payload_sink = blocks.vector_sink_f()
        self.tb.connect(data_src, (hpd, 0))
        self.tb.connect(trigger_src, (hpd, 1))
        self.tb.connect((hpd, 0), header_sink)
        self.tb.connect((hpd, 1), payload_sink)
        self.tb.start()
        time.sleep(0.2)
        hpd.to_basic_block()._post(pmt.intern('header_data'), pmt.to_pmt({'frame_len': len(payload), 'payload_offset': payload_offset}))
        while len(payload_sink.data()) < len(payload):
            time.sleep(0.2)
        self.tb.stop()
        self.tb.wait()
        self.assertEqual(header_sink.data(), [0] + header + [payload[0]])
        self.assertEqual(payload_sink.data(), data_signal[n_zeros + len(header) + payload_offset:n_zeros + len(header) + payload_offset + len(payload)])
        ptags_payload = {}
        for tag in payload_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_payload[ptag.key] = ptag.offset
        expected_tags_payload = {'frame_len': 0, 'payload_offset': 0, 'tag4': 3 - payload_offset}
        self.assertEqual(expected_tags_payload, ptags_payload)

    def test_002_symbols(self):
        if False:
            i = 10
            return i + 15
        '\n        Same as before, but operate on symbols\n        '
        n_zeros = 1
        items_per_symbol = 3
        gi = 1
        n_symbols = 4
        header = (1, 2, 3)
        payload = (1, 2, 3)
        data_signal = (0,) * n_zeros + (0,) + header + ((0,) + payload) * n_symbols
        trigger_signal = [0] * len(data_signal)
        trigger_signal[n_zeros] = 1
        testtag1 = make_tag('tag1', 0, 0)
        testtag2 = make_tag('tag2', 23, n_zeros)
        testtag3 = make_tag('tag3', 42, n_zeros + gi + 1)
        testtag4 = make_tag('tag4', 314, n_zeros + (gi + items_per_symbol) * 2 + 1)
        data_src = blocks.vector_source_f(data_signal, False, tags=(testtag1, testtag2, testtag3, testtag4))
        trigger_src = blocks.vector_source_b(trigger_signal, False)
        hpd = digital.header_payload_demux(len(header) // items_per_symbol, items_per_symbol, gi, 'frame_len', 'detect', True, gr.sizeof_float)
        self.assertEqual(pmt.length(hpd.message_ports_in()), 2)
        header_sink = blocks.vector_sink_f(items_per_symbol)
        payload_sink = blocks.vector_sink_f(items_per_symbol)
        self.tb.connect(data_src, (hpd, 0))
        self.tb.connect(trigger_src, (hpd, 1))
        self.tb.connect((hpd, 0), header_sink)
        self.tb.connect((hpd, 1), payload_sink)
        self.tb.start()
        time.sleep(0.2)
        hpd.to_basic_block()._post(pmt.intern('header_data'), pmt.from_long(n_symbols))
        while len(payload_sink.data()) < len(payload) * n_symbols:
            time.sleep(0.2)
        self.tb.stop()
        self.tb.wait()
        self.assertEqual(header_sink.data(), list(header))
        self.assertEqual(payload_sink.data(), list(payload * n_symbols))
        ptags_header = []
        for tag in header_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_header.append({'key': ptag.key, 'offset': ptag.offset})
        expected_tags_header = [{'key': 'tag2', 'offset': 0}, {'key': 'tag3', 'offset': 0}]
        self.assertEqual(expected_tags_header, ptags_header)
        ptags_payload = []
        for tag in payload_sink.tags():
            ptag = gr.tag_to_python(tag)
            ptags_payload.append({'key': ptag.key, 'offset': ptag.offset})
        expected_tags_payload = [{'key': 'frame_len', 'offset': 0}, {'key': 'tag4', 'offset': 1}]
        self.assertEqual(expected_tags_payload, ptags_payload)

    def test_003_t(self):
        if False:
            return 10
        '\n        Like test 1, but twice, plus one fail\n        '
        n_zeros = 5
        header = [1, 2, 3]
        header_fail = [-1, -2, -4]
        payload1 = list(range(5, 20))
        payload2 = [42]
        sampling_rate = 2
        data_signal = [0] * n_zeros + header + payload1
        trigger_signal = [0] * len(data_signal) * 2
        trigger_signal[n_zeros] = 1
        trigger_signal[len(data_signal)] = 1
        trigger_signal[len(data_signal) + len(header_fail) + n_zeros] = 1
        print('Triggers at: {0} {1} {2}'.format(n_zeros, len(data_signal), len(data_signal) + len(header_fail) + n_zeros))
        tx_signal = data_signal + header_fail + [0] * n_zeros + header + payload2 + [0] * 1000
        timing_tag = make_tag('rx_time', (0, 0), 0)
        rx_freq_tag1 = make_tag('rx_freq', 1.0, 0)
        rx_freq_tag2 = make_tag('rx_freq', 1.5, 29)
        rx_freq_tag3 = make_tag('rx_freq', 2.0, 30)
        data_src = blocks.vector_source_f(tx_signal, False, tags=(timing_tag, rx_freq_tag1, rx_freq_tag2, rx_freq_tag3))
        trigger_src = blocks.vector_source_b(trigger_signal, False)
        hpd = digital.header_payload_demux(header_len=len(header), items_per_symbol=1, guard_interval=0, length_tag_key='frame_len', trigger_tag_key='detect', output_symbols=False, itemsize=gr.sizeof_float, timing_tag_key='rx_time', samp_rate=sampling_rate, special_tags=('rx_freq',))
        self.assertEqual(pmt.length(hpd.message_ports_in()), 2)
        header_sink = blocks.vector_sink_f()
        payload_sink = blocks.vector_sink_f()
        self.tb.connect(data_src, (hpd, 0))
        self.tb.connect(trigger_src, (hpd, 1))
        self.tb.connect((hpd, 0), header_sink)
        self.tb.connect((hpd, 1), payload_sink)
        self.tb.start()
        time.sleep(0.2)
        hpd.to_basic_block()._post(pmt.intern('header_data'), pmt.from_long(len(payload1)))
        while len(payload_sink.data()) < len(payload1):
            time.sleep(0.2)
        hpd.to_basic_block()._post(pmt.intern('header_data'), pmt.PMT_F)
        time.sleep(0.7)
        hpd.to_basic_block()._post(pmt.intern('header_data'), pmt.from_long(len(payload2)))
        while len(payload_sink.data()) < len(payload1) + len(payload2):
            time.sleep(0.2)
        self.tb.stop()
        self.tb.wait()
        self.assertEqual(header_sink.data(), list(header + header_fail + header))
        self.assertEqual(payload_sink.data(), payload1 + payload2)
        tags_payload = [gr.tag_to_python(x) for x in payload_sink.tags()]
        tags_payload = sorted([(x.offset, x.key, x.value) for x in tags_payload])
        tags_expected_payload = [(0, 'frame_len', len(payload1)), (len(payload1), 'frame_len', len(payload2))]
        tags_header = [gr.tag_to_python(x) for x in header_sink.tags()]
        tags_header = sorted([(x.offset, x.key, x.value) for x in tags_header])
        tags_expected_header = [(0, 'rx_freq', 1.0), (0, 'rx_time', (2, 0.5)), (len(header), 'rx_freq', 1.0), (len(header), 'rx_time', (11, 0.5)), (2 * len(header), 'rx_freq', 2.0), (2 * len(header), 'rx_time', (15, 0.5))]
        self.assertEqual(tags_header, tags_expected_header)
        self.assertEqual(tags_payload, tags_expected_payload)

    def test_004_fuzz(self):
        if False:
            print('Hello World!')
        '\n        Long random test\n        '

        def create_signal(n_bursts, header_len, max_gap, max_burstsize, fail_rate):
            if False:
                while True:
                    i = 10
            signal = []
            indexes = []
            burst_sizes = []
            total_payload_len = 0
            for _ in range(n_bursts):
                gap_size = random.randint(0, max_gap)
                signal += [0] * gap_size
                is_failure = random.random() < fail_rate
                if not is_failure:
                    burst_size = random.randint(0, max_burstsize)
                else:
                    burst_size = 0
                total_payload_len += burst_size
                indexes += [len(signal)]
                signal += [1] * header_len
                signal += [2] * burst_size
                burst_sizes += [burst_size]
            return (signal, indexes, total_payload_len, burst_sizes)

        def indexes_to_triggers(indexes, signal_len):
            if False:
                return 10
            '\n            Convert indexes to a mix of trigger signals and tags\n            '
            trigger_signal = [0] * signal_len
            trigger_tags = []
            for index in indexes:
                if random.random() > 0.5:
                    trigger_signal[index] = 1
                else:
                    trigger_tags += [make_tag('detect', True, index)]
            return (trigger_signal, trigger_tags)
        n_bursts = 400
        header_len = 5
        max_gap = 50
        max_burstsize = 100
        fail_rate = 0.05
        (signal, indexes, total_payload_len, burst_sizes) = create_signal(n_bursts, header_len, max_gap, max_burstsize, fail_rate)
        (trigger_signal, trigger_tags) = indexes_to_triggers(indexes, len(signal))
        data_src = blocks.vector_source_f(signal, False, tags=trigger_tags)
        trigger_src = blocks.vector_source_b(trigger_signal, False)
        hpd = digital.header_payload_demux(header_len=header_len, items_per_symbol=1, guard_interval=0, length_tag_key='frame_len', trigger_tag_key='detect', output_symbols=False, itemsize=gr.sizeof_float, timing_tag_key='rx_time', samp_rate=1.0, special_tags=('rx_freq',))
        mock_header_demod = HeaderToMessageBlock(numpy.float32, header_len, burst_sizes)
        header_sink = blocks.vector_sink_f()
        payload_sink = blocks.vector_sink_f()
        self.connect_all_blocks(data_src, trigger_src, hpd, mock_header_demod, payload_sink, header_sink)
        self.run_tb(payload_sink, total_payload_len, header_sink, header_len * n_bursts)
        self.assertEqual(header_sink.data(), list([1] * header_len * n_bursts))
        self.assertEqual(payload_sink.data(), list([2] * total_payload_len))
if __name__ == '__main__':
    gr_unittest.run(qa_header_payload_demux)