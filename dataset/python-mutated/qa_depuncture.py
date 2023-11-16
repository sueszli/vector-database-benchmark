from collections import deque
from gnuradio import gr, gr_unittest, blocks
from gnuradio import fec

class test_depuncture(gr_unittest.TestCase):

    def depuncture_setup(self):
        if False:
            return 10
        p = []
        for i in range(self.puncsize):
            p.append(self.puncpat >> self.puncsize - 1 - i & 1)
        d = deque(p)
        d.rotate(self.delay)
        _puncpat = list(d)
        k = 0
        self.expected = []
        for n in range(len(self.src_data) // (self.puncsize - self.puncholes)):
            for i in range(self.puncsize):
                if _puncpat[i] == 1:
                    self.expected.append(self.src_data[k])
                    k += 1
                else:
                    self.expected.append(self.sym)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.src_data = 2000 * list(range(64))
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_000(self):
        if False:
            print('Hello World!')
        self.puncsize = 8
        self.puncpat = 239
        self.delay = 0
        self.sym = 0
        self.puncholes = 1
        self.depuncture_setup()
        src = blocks.vector_source_b(self.src_data)
        op = fec.depuncture_bb(self.puncsize, self.puncpat, self.delay, self.sym)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        dst_data = list(dst.data())
        for i in range(len(dst_data)):
            dst_data[i] = int(dst_data[i])
        self.assertSequenceEqualGR(self.expected, dst_data)

    def test_001(self):
        if False:
            while True:
                i = 10
        self.puncsize = 8
        self.puncpat = 239
        self.delay = 1
        self.sym = 0
        self.puncholes = 1
        self.depuncture_setup()
        src = blocks.vector_source_b(self.src_data)
        op = fec.depuncture_bb(self.puncsize, self.puncpat, self.delay, self.sym)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        dst_data = list(dst.data())
        for i in range(len(dst_data)):
            dst_data[i] = int(dst_data[i])
        self.assertSequenceEqualGR(self.expected, dst_data)

    def test_002(self):
        if False:
            return 10
        self.puncsize = 4
        self.puncpat = 21845
        self.delay = 0
        self.sym = 0
        self.puncholes = 2
        self.depuncture_setup()
        src = blocks.vector_source_b(self.src_data)
        op = fec.depuncture_bb(self.puncsize, self.puncpat, self.delay, self.sym)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        dst_data = list(dst.data())
        for i in range(len(dst_data)):
            dst_data[i] = int(dst_data[i])
        self.assertSequenceEqualGR(self.expected, dst_data)

    def test_003(self):
        if False:
            print('Hello World!')
        self.puncsize = 4
        self.puncpat0 = 21845
        self.puncpat1 = 85
        self.delay = 1
        self.sym = 0
        src = blocks.vector_source_b(self.src_data)
        op0 = fec.depuncture_bb(self.puncsize, self.puncpat0, self.delay, self.sym)
        op1 = fec.depuncture_bb(self.puncsize, self.puncpat1, self.delay, self.sym)
        dst0 = blocks.vector_sink_b()
        dst1 = blocks.vector_sink_b()
        self.tb.connect(src, op0, dst0)
        self.tb.connect(src, op1, dst1)
        self.tb.run()
        dst_data0 = list(dst0.data())
        for i in range(len(dst_data0)):
            dst_data0[i] = int(dst_data0[i])
        dst_data1 = list(dst1.data())
        for i in range(len(dst_data1)):
            dst_data1[i] = int(dst_data1[i])
        self.assertSequenceEqualGR(dst_data1, dst_data0)

    def test_004(self):
        if False:
            for i in range(10):
                print('nop')
        self.puncsize = 8
        self.puncpat = 239
        self.delay = 0
        self.sym = 127
        self.puncholes = 1
        self.depuncture_setup()
        src = blocks.vector_source_b(self.src_data)
        op = fec.depuncture_bb(self.puncsize, self.puncpat, self.delay)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        dst_data = list(dst.data())
        for i in range(len(dst_data)):
            dst_data[i] = int(dst_data[i])
        self.assertSequenceEqualGR(self.expected, dst_data)
if __name__ == '__main__':
    gr_unittest.run(test_depuncture)