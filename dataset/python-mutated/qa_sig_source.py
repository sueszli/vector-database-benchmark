import math
import pmt
from gnuradio import gr, gr_unittest, analog, blocks

class test_sig_source(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_const_f(self):
        if False:
            i = 10
            return i + 15
        tb = self.tb
        expected_result = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        src1 = analog.sig_source_f(1000000.0, analog.GR_CONST_WAVE, 0, 1.5)
        op = blocks.head(gr.sizeof_float, 10)
        dst1 = blocks.vector_sink_f()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_const_i(self):
        if False:
            return 10
        tb = self.tb
        expected_result = [1, 1, 1, 1]
        src1 = analog.sig_source_i(1000000.0, analog.GR_CONST_WAVE, 0, 1)
        op = blocks.head(gr.sizeof_int, 4)
        dst1 = blocks.vector_sink_i()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_const_b(self):
        if False:
            while True:
                i = 10
        tb = self.tb
        expected_result = [1, 1, 1, 1]
        src1 = analog.sig_source_b(1000000.0, analog.GR_CONST_WAVE, 0, 1)
        op = blocks.head(gr.sizeof_char, 4)
        dst1 = blocks.vector_sink_b()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_sine_f(self):
        if False:
            for i in range(10):
                print('nop')
        tb = self.tb
        sqrt2 = math.sqrt(2) / 2
        expected_result = [0, sqrt2, 1, sqrt2, 0, -sqrt2, -1, -sqrt2, 0]
        src1 = analog.sig_source_f(8, analog.GR_SIN_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_float, 9)
        dst1 = blocks.vector_sink_f()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_sine_b(self):
        if False:
            while True:
                i = 10
        tb = self.tb
        sqrt2 = math.sqrt(2) / 2
        temp_result = [0, sqrt2, 1, sqrt2, 0, -sqrt2, -1, -sqrt2, 0]
        amp = 8
        expected_result = tuple([int(z * amp) for z in temp_result])
        src1 = analog.sig_source_b(8, analog.GR_SIN_WAVE, 1.0, amp)
        op = blocks.head(gr.sizeof_char, 9)
        dst1 = blocks.vector_sink_b()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        dst_data_signed = [b if b < 127 else (256 - b) * -1 for b in dst_data]
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data_signed)

    def test_cosine_f(self):
        if False:
            while True:
                i = 10
        tb = self.tb
        sqrt2 = math.sqrt(2) / 2
        expected_result = [1, sqrt2, 0, -sqrt2, -1, -sqrt2, 0, sqrt2, 1]
        src1 = analog.sig_source_f(8, analog.GR_COS_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_float, 9)
        dst1 = blocks.vector_sink_f()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_cosine_c(self):
        if False:
            i = 10
            return i + 15
        tb = self.tb
        sqrt2 = math.sqrt(2) / 2
        sqrt2j = 1j * math.sqrt(2) / 2
        expected_result = [1, sqrt2 + sqrt2j, 1j, -sqrt2 + sqrt2j, -1, -sqrt2 - sqrt2j, -1j, sqrt2 - sqrt2j, 1]
        src1 = analog.sig_source_c(8, analog.GR_COS_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_gr_complex, 9)
        dst1 = blocks.vector_sink_c()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_sqr_c(self):
        if False:
            i = 10
            return i + 15
        tb = self.tb
        expected_result = [1j, 1j, 0, 0, 1, 1, 1 + 0j, 1 + 1j, 1j]
        src1 = analog.sig_source_c(8, analog.GR_SQR_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_gr_complex, 9)
        dst1 = blocks.vector_sink_c()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_tri_c(self):
        if False:
            return 10
        tb = self.tb
        expected_result = [1 + 0.5j, 0.75 + 0.75j, 0.5 + 1j, 0.25 + 0.75j, 0 + 0.5j, 0.25 + 0.25j, 0.5 + 0j, 0.75 + 0.25j, 1 + 0.5j]
        src1 = analog.sig_source_c(8, analog.GR_TRI_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_gr_complex, 9)
        dst1 = blocks.vector_sink_c()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_saw_c(self):
        if False:
            return 10
        tb = self.tb
        expected_result = [0.5 + 0.25j, 0.625 + 0.375j, 0.75 + 0.5j, 0.875 + 0.625j, 0 + 0.75j, 0.125 + 0.875j, 0.25 + 1j, 0.375 + 0.125j, 0.5 + 0.25j]
        src1 = analog.sig_source_c(8, analog.GR_SAW_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_gr_complex, 9)
        dst1 = blocks.vector_sink_c()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertComplexTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_sqr_f(self):
        if False:
            i = 10
            return i + 15
        tb = self.tb
        expected_result = [0, 0, 0, 0, 1, 1, 1, 1, 0]
        src1 = analog.sig_source_f(8, analog.GR_SQR_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_float, 9)
        dst1 = blocks.vector_sink_f()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_tri_f(self):
        if False:
            print('Hello World!')
        tb = self.tb
        expected_result = [1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1]
        src1 = analog.sig_source_f(8, analog.GR_TRI_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_float, 9)
        dst1 = blocks.vector_sink_f()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_saw_f(self):
        if False:
            print('Hello World!')
        tb = self.tb
        expected_result = [0.5, 0.625, 0.75, 0.875, 0, 0.125, 0.25, 0.375, 0.5]
        src1 = analog.sig_source_f(8, analog.GR_SAW_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_float, 9)
        dst1 = blocks.vector_sink_f()
        tb.connect(src1, op)
        tb.connect(op, dst1)
        tb.run()
        dst_data = dst1.data()
        self.assertFloatTuplesAlmostEqual(expected_result, dst_data, 5)

    def test_freq_msg(self):
        if False:
            i = 10
            return i + 15
        src = analog.sig_source_c(8, analog.GR_SIN_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_gr_complex, 9)
        snk = blocks.vector_sink_c()
        self.tb.connect(src, op, snk)
        self.assertAlmostEqual(src.frequency(), 1.0)
        frequency = 3.0
        src._post(pmt.to_pmt('cmd'), pmt.to_pmt({'freq': frequency}))
        self.tb.run()
        self.assertAlmostEqual(src.frequency(), frequency)

    def test_cmd_msg(self):
        if False:
            for i in range(10):
                print('nop')
        src = analog.sig_source_c(8, analog.GR_SIN_WAVE, 1.0, 1.0)
        op = blocks.head(gr.sizeof_gr_complex, 9)
        snk = blocks.vector_sink_c()
        self.tb.connect(src, op, snk)
        self.assertAlmostEqual(src.frequency(), 1.0)
        frequency = 3.0
        amplitude = 10
        offset = -1.0
        src._post(pmt.to_pmt('cmd'), pmt.to_pmt({'freq': frequency, 'ampl': amplitude, 'offset': offset}))
        self.tb.run()
        self.assertAlmostEqual(src.frequency(), frequency)
        self.assertAlmostEqual(src.amplitude(), amplitude)
        self.assertAlmostEqual(src.offset(), offset)
if __name__ == '__main__':
    gr_unittest.run(test_sig_source)