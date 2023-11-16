import numpy
from gnuradio import gr, gr_unittest, wavelet, analog, blocks
import copy
import math

def sqr(x):
    if False:
        return 10
    return x * x

def np2(k):
    if False:
        while True:
            i = 10
    m = 0
    n = k - 1
    while n > 0:
        m += 1
    return m

class test_classify(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_001_(self):
        if False:
            while True:
                i = 10
        src_data = numpy.array([-1.0, 1.0, -1.0, 1.0])
        trg_data = src_data * 0.5
        src = blocks.vector_source_f(src_data)
        dst = blocks.vector_sink_f()
        rail = analog.rail_ff(-0.5, 0.5)
        self.tb.connect(src, rail)
        self.tb.connect(rail, dst)
        self.tb.run()
        rsl_data = dst.data()
        sum = 0
        for (u, v) in zip(trg_data, rsl_data):
            w = u - v
            sum += w * w
        sum /= float(len(trg_data))
        assert sum < 1e-06

    def test_002_(self):
        if False:
            return 10
        src_data = numpy.array([-1.0, -1.0 / 2.0, -1.0 / 3.0, -1.0 / 4.0, -1.0 / 5.0])
        trg_data = copy.deepcopy(src_data)
        src = blocks.vector_source_f(src_data, False, len(src_data))
        st = blocks.stretch_ff(-1.0 / 5.0, len(src_data))
        dst = blocks.vector_sink_f(len(src_data))
        self.tb.connect(src, st)
        self.tb.connect(st, dst)
        self.tb.run()
        rsl_data = dst.data()
        sum = 0
        for (u, v) in zip(trg_data, rsl_data):
            w = u - v
            sum += w * w
        sum /= float(len(trg_data))
        assert sum < 1e-06

    def test_003_(self):
        if False:
            print('Hello World!')
        src_grid = (0.0, 1.0, 2.0, 3.0, 4.0)
        trg_grid = copy.deepcopy(src_grid)
        src_data = (0.0, 1.0, 0.0, 1.0, 0.0)
        src = blocks.vector_source_f(src_data, False, len(src_grid))
        sq = wavelet.squash_ff(src_grid, trg_grid)
        dst = blocks.vector_sink_f(len(trg_grid))
        self.tb.connect(src, sq)
        self.tb.connect(sq, dst)
        self.tb.run()
        rsl_data = dst.data()
        sum = 0
        for (u, v) in zip(src_data, rsl_data):
            w = u - v
            sum += w * w
        sum /= float(len(src_data))
        assert sum < 1e-06

    def test_005_(self):
        if False:
            print('Hello World!')
        src_data = (1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        dwav = numpy.array(src_data)
        wvps = numpy.zeros(3)
        scl = 1.0 / sqr(dwav[0])
        k = 1
        for e in range(len(wvps)):
            wvps[e] = scl * sqr(dwav[k:k + (1 << e)]).sum()
            k += 1 << e
        src = blocks.vector_source_f(src_data, False, len(src_data))
        kon = wavelet.wvps_ff(len(src_data))
        dst = blocks.vector_sink_f(int(math.ceil(math.log(len(src_data), 2))))
        self.tb.connect(src, kon)
        self.tb.connect(kon, dst)
        self.tb.run()
        snk_data = dst.data()
        sum = 0
        for (u, v) in zip(snk_data, wvps):
            w = u - v
            sum += w * w
        sum /= float(len(snk_data))
        assert sum < 1e-06
if __name__ == '__main__':
    gr_unittest.run(test_classify)