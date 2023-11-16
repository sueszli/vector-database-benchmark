import random
import cmath
from gnuradio import gr, gr_unittest, digital, blocks
from gnuradio.digital import psk

class test_costas_loop_cc(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test01(self):
        if False:
            while True:
                i = 10
        natfreq = 0.0
        order = 2
        self.test = digital.costas_loop_cc(natfreq, order)
        data = 100 * [complex(1, 0)]
        self.src = blocks.vector_source_c(data, False)
        self.snk = blocks.vector_sink_c()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        expected_result = data
        dst_data = self.snk.data()
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 5)

    def test02(self):
        if False:
            while True:
                i = 10
        natfreq = 0.25
        order = 2
        self.test = digital.costas_loop_cc(natfreq, order)
        data = [complex(2 * random.randint(0, 1) - 1, 0) for i in range(100)]
        self.src = blocks.vector_source_c(data, False)
        self.snk = blocks.vector_sink_c()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        expected_result = data
        dst_data = self.snk.data()
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 5)

    def test03(self):
        if False:
            while True:
                i = 10
        natfreq = 0.25
        order = 2
        self.test = digital.costas_loop_cc(natfreq, order)
        rot = cmath.exp(0.2j)
        data = [complex(2 * random.randint(0, 1) - 1, 0) for i in range(100)]
        N = 40
        expected_result = data[N:]
        data = [rot * d for d in data]
        self.src = blocks.vector_source_c(data, False)
        self.snk = blocks.vector_sink_c()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        dst_data = self.snk.data()[N:]
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 2)

    def test04(self):
        if False:
            print('Hello World!')
        natfreq = 0.25
        order = 4
        self.test = digital.costas_loop_cc(natfreq, order)
        rot = cmath.exp(0.2j)
        data = [complex(2 * random.randint(0, 1) - 1, 2 * random.randint(0, 1) - 1) for i in range(100)]
        N = 40
        expected_result = data[N:]
        data = [rot * d for d in data]
        self.src = blocks.vector_source_c(data, False)
        self.snk = blocks.vector_sink_c()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        dst_data = self.snk.data()[N:]
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 2)

    def test05(self):
        if False:
            for i in range(10):
                print('nop')
        natfreq = 0.25
        order = 8
        self.test = digital.costas_loop_cc(natfreq, order)
        rot = cmath.exp(-cmath.pi / 8j)
        const = psk.psk_constellation(order)
        data = [random.randint(0, 7) for i in range(100)]
        data = [2 * rot * const.points()[d] for d in data]
        N = 40
        expected_result = data[N:]
        rot = cmath.exp(0.1j)
        data = [rot * d for d in data]
        self.src = blocks.vector_source_c(data, False)
        self.snk = blocks.vector_sink_c()
        self.tb.connect(self.src, self.test, self.snk)
        self.tb.run()
        dst_data = self.snk.data()[N:]
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 2)
if __name__ == '__main__':
    gr_unittest.run(test_costas_loop_cc)