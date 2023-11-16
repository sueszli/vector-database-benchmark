from gnuradio import gr, gr_unittest, digital, blocks
from gnuradio.digital.utils import lfsr_args
import numpy as np
import pmt

def additive_scramble_lfsr(mask, seed, reglen, bpb, data):
    if False:
        return 10
    l = digital.lfsr(mask, seed, reglen)
    out = []
    for d in data:
        scramble_word = 0
        for i in range(0, bpb):
            scramble_word ^= l.next_bit() << i
        out.append(d ^ scramble_word)
    return out

class test_scrambler(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_lfsr_002(self):
        if False:
            for i in range(10):
                print('nop')
        _a = lfsr_args(1, 51, 3, 0)
        l = digital.lfsr(*_a)
        seq = [l.next_bit() for _ in range(2 ** 10)]
        reg = np.zeros(52, np.int8)
        reg[::-1][(51, 3, 0),] = 1
        res = np.convolve(seq, reg) % 2
        self.assertEqual(sum(res[52:-52]), 0, msg='LRS not generated properly')

    def test_scrambler_descrambler_001(self):
        if False:
            print('Hello World!')
        src_data = np.random.randint(0, 2, 500, dtype=np.int8)
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.scrambler_bb(*lfsr_args(1, 7, 2, 0))
        descrambler = digital.descrambler_bb(*lfsr_args(7, 7, 2, 0))
        m_tap = blocks.vector_sink_b()
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, descrambler, dst)
        self.tb.connect(scrambler, m_tap)
        self.tb.run()
        self.assertEqual(src_data[:-7].tolist(), dst.data()[7:])
        self.assertEqual(tuple(np.convolve(m_tap.data(), [1, 0, 0, 0, 0, 1, 0, 1]) % 2)[7:-10], tuple(src_data[:-10]))

    def test_scrambler_descrambler_002(self):
        if False:
            while True:
                i = 10
        _a = lfsr_args(1, 51, 6, 0)
        src_data = np.random.randint(0, 2, 1000, dtype=np.int8)
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.scrambler_bb(*_a)
        m_tap = blocks.vector_sink_b()
        descrambler = digital.descrambler_bb(*_a)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, descrambler, dst)
        self.tb.connect(scrambler, m_tap)
        self.tb.run()
        self.assertTrue(np.all(src_data[:-51] == dst.data()[51:]))
        reg = np.zeros(52, np.int8)
        reg[::-1][(51, 6, 0),] = 1
        self.assertTrue(np.all(np.convolve(m_tap.data(), reg)[51:-60] % 2 == src_data[:-60]))

    def test_scrambler_descrambler_003(self):
        if False:
            while True:
                i = 10
        src_data = np.random.randint(0, 2, 1000, dtype=np.int8)
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.scrambler_bb(*lfsr_args(1, 12, 10, 3, 2, 0))
        descrambler1 = digital.descrambler_bb(*lfsr_args(1, 5, 3, 0))
        descrambler2 = digital.descrambler_bb(*lfsr_args(1, 7, 2, 0))
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, descrambler1, descrambler2, dst)
        self.tb.run()
        self.assertTrue(np.all(src_data[:-12] == dst.data()[12:]))

    def test_additive_scrambler_001(self):
        if False:
            print('Hello World!')
        _a = lfsr_args(1, 51, 3, 0)
        src_data = np.random.randint(0, 2, 1000, dtype=np.int8).tolist()
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.additive_scrambler_bb(*_a)
        descrambler = digital.additive_scrambler_bb(*_a)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, descrambler, dst)
        self.tb.run()
        self.assertEqual(tuple(src_data), tuple(dst.data()))

    def test_additive_scrambler_002(self):
        if False:
            print('Hello World!')
        _a = lfsr_args(1, 51, 3, 0)
        src_data = [1] * 1000
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.additive_scrambler_bb(*_a)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, dst)
        self.tb.run()
        reg = np.zeros(52, np.int8)
        reg[::-1][(51, 3, 0),] = 1
        res = (np.convolve(dst.data(), reg) % 2)[52:-52]
        self.assertEqual(len(res), sum(res))

    def test_scrambler_descrambler(self):
        if False:
            i = 10
            return i + 15
        src_data = [1] * 1000
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.scrambler_bb(138, 127, 7)
        descrambler = digital.descrambler_bb(138, 127, 7)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, descrambler, dst)
        self.tb.run()
        self.assertEqual(src_data[:-8], dst.data()[8:])

    def test_additive_scrambler(self):
        if False:
            i = 10
            return i + 15
        src_data = [1] * 1000
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.additive_scrambler_bb(138, 127, 7)
        descrambler = digital.additive_scrambler_bb(138, 127, 7)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, descrambler, dst)
        self.tb.run()
        self.assertEqual(src_data, dst.data())

    def test_additive_scrambler_reset(self):
        if False:
            while True:
                i = 10
        src_data = [1] * 200
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.additive_scrambler_bb(138, 127, 7, 50)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, dst)
        self.tb.run()
        output = dst.data()
        self.assertEqual(output[:50] * 4, output)

    def test_additive_scrambler_reset_3bpb(self):
        if False:
            i = 10
            return i + 15
        src_data = [5] * 200
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.additive_scrambler_bb(138, 127, 7, 50, 3)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, dst)
        self.tb.run()
        output = dst.data()
        self.assertEqual(output[:50] * 4, output)

    def test_additive_scrambler_tags(self):
        if False:
            print('Hello World!')
        src_data = [1] * 1000
        src = blocks.vector_source_b(src_data, False)
        scrambler = digital.additive_scrambler_bb(138, 127, 7, 100)
        descrambler = digital.additive_scrambler_bb(138, 127, 7, 100)
        reset_tag_key = 'reset_lfsr'
        reset_tag1 = gr.tag_t()
        reset_tag1.key = pmt.string_to_symbol(reset_tag_key)
        reset_tag1.offset = 17
        reset_tag2 = gr.tag_t()
        reset_tag2.key = pmt.string_to_symbol(reset_tag_key)
        reset_tag2.offset = 110
        reset_tag3 = gr.tag_t()
        reset_tag3.key = pmt.string_to_symbol(reset_tag_key)
        reset_tag3.offset = 523
        src = blocks.vector_source_b(src_data, False, 1, (reset_tag1, reset_tag2, reset_tag3))
        scrambler = digital.additive_scrambler_bb(138, 127, 7, 100, 1, reset_tag_key)
        descrambler = digital.additive_scrambler_bb(138, 127, 7, 100, 1, reset_tag_key)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, descrambler, dst)
        self.tb.run()
        self.assertEqual(src_data, dst.data())

    def test_additive_scrambler_tags_oneway(self):
        if False:
            while True:
                i = 10
        src_data = [x for x in range(0, 10)]
        reset_tag_key = 'reset_lfsr'
        reset_tag1 = gr.tag_t()
        reset_tag1.key = pmt.string_to_symbol(reset_tag_key)
        reset_tag1.offset = 0
        reset_tag2 = gr.tag_t()
        reset_tag2.key = pmt.string_to_symbol(reset_tag_key)
        reset_tag2.offset = 10
        reset_tag3 = gr.tag_t()
        reset_tag3.key = pmt.string_to_symbol(reset_tag_key)
        reset_tag3.offset = 20
        src = blocks.vector_source_b(src_data * 3, False, 1, (reset_tag1, reset_tag2, reset_tag3))
        scrambler = digital.additive_scrambler_bb(138, 127, 7, 0, 8, reset_tag_key)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, scrambler, dst)
        self.tb.run()
        expected_data = additive_scramble_lfsr(138, 127, 7, 8, src_data)
        self.assertEqual(expected_data * 3, dst.data())
if __name__ == '__main__':
    gr_unittest.run(test_scrambler)