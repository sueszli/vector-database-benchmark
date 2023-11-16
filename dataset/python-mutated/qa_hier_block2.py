import time
from gnuradio import gr_unittest, blocks, gr
from gnuradio.gr.hier_block2 import _multiple_endpoints, _optional_endpoints
import pmt

class test_hblk(gr.hier_block2):

    def __init__(self, io_sig=1 * [gr.sizeof_gr_complex], ndebug=2):
        if False:
            while True:
                i = 10
        gr.hier_block2.__init__(self, 'test_hblk', gr.io_signature(len(io_sig), len(io_sig), io_sig[0]), gr.io_signature(0, 0, 0))
        self.message_port_register_hier_in('msg_in')
        self.vsnk = blocks.vector_sink_c()
        self.blks = []
        for i in range(0, ndebug):
            self.blks.append(blocks.message_debug())
        self.connect(self, self.vsnk)
        for blk in self.blks:
            self.msg_connect(self, 'msg_in', blk, 'print')

class test_hier_block2(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.call_log = []
        self.Block = type('Block', (), {'to_basic_block': lambda bl: bl})

    def test_f(self, *args):
        if False:
            while True:
                i = 10
        'test doc'
        self.call_log.append(args)
    multi = _multiple_endpoints(test_f)
    opt = _optional_endpoints(test_f)

    def test_000(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.multi.__doc__, 'test doc')
        self.assertEqual(self.multi.__name__, 'test_f')

    def test_001(self):
        if False:
            print('Hello World!')
        b = self.Block()
        self.multi(b)
        self.assertEqual((b,), self.call_log[0])

    def test_002(self):
        if False:
            return 10
        (b1, b2) = (self.Block(), self.Block())
        self.multi(b1, b2)
        self.assertEqual([(b1, 0, b2, 0)], self.call_log)

    def test_003(self):
        if False:
            for i in range(10):
                print('nop')
        (b1, b2) = (self.Block(), self.Block())
        self.multi((b1, 1), (b2, 2))
        self.assertEqual([(b1, 1, b2, 2)], self.call_log)

    def test_004(self):
        if False:
            return 10
        (b1, b2, b3, b4) = [self.Block()] * 4
        self.multi(b1, (b2, 5), b3, (b4, 0))
        expected = [(b1, 0, b2, 5), (b2, 5, b3, 0), (b3, 0, b4, 0)]
        self.assertEqual(expected, self.call_log)

    def test_005(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            self.multi((self.Block(), 5))

    def test_006(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            self.multi(self.Block(), (5, 5))

    def test_007(self):
        if False:
            print('Hello World!')
        (b1, b2) = (self.Block(), self.Block())
        self.opt(b1, 'in', b2, 'out')
        self.assertEqual([(b1, 'in', b2, 'out')], self.call_log)

    def test_008(self):
        if False:
            while True:
                i = 10
        (f, b1, b2) = (self.multi, self.Block(), self.Block())
        self.opt((b1, 'in'), (b2, 'out'))
        self.assertEqual([(b1, 'in', b2, 'out')], self.call_log)

    def test_009(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            self.multi(self.Block(), 5)

    def test_010_end_with_head(self):
        if False:
            i = 10
            return i + 15
        import math
        exp = 1j * 440 / 44100
        src = blocks.vector_source_c([math.e ** (exp * n) for n in range(10 ** 6)])
        head = blocks.head(gr.sizeof_gr_complex, 1000)
        test = test_hblk([gr.sizeof_gr_complex], 0)
        tb = gr.top_block()
        tb.connect(src, head, test)
        tb.run()

    def test_011_test_message_connect(self):
        if False:
            while True:
                i = 10
        import math
        exp = 1j * 440 / 44100
        src = blocks.vector_source_c([math.e ** (exp * n) for n in range(10 ** 6)])
        strobe = blocks.message_strobe(pmt.PMT_NIL, 100)
        head = blocks.head(gr.sizeof_gr_complex, 1000)
        test = test_hblk([gr.sizeof_gr_complex], 1)
        tb = gr.top_block()
        tb.connect(src, head, test)
        tb.msg_connect(strobe, 'strobe', test, 'msg_in')
        tb.start()
        time.sleep(0.5)
        tb.stop()
        tb.wait()

    def test_012(self):
        if False:
            i = 10
            return i + 15
        import math
        exp = 1j * 440 / 44100
        src = blocks.vector_source_c([math.e ** (exp * n) for n in range(10 ** 6)])
        strobe = blocks.message_strobe(pmt.PMT_NIL, 100)
        head = blocks.head(gr.sizeof_gr_complex, 1000)
        test = test_hblk([gr.sizeof_gr_complex], 16)
        tb = gr.top_block()
        tb.connect(src, head, test)
        tb.msg_connect(strobe, 'strobe', test, 'msg_in')
        tb.start()
        time.sleep(0.5)
        tb.stop()
        tb.wait()
if __name__ == '__main__':
    gr_unittest.run(test_hier_block2)