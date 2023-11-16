from gnuradio import gr, gr_unittest, blocks

class test_boolean_operators(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def help_ss(self, src_data, exp_data, op):
        if False:
            print('Hello World!')
        for s in zip(list(range(len(src_data))), src_data):
            src = blocks.vector_source_s(s[1])
            self.tb.connect(src, (op, s[0]))
        dst = blocks.vector_sink_s()
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_bb(self, src_data, exp_data, op):
        if False:
            return 10
        for s in zip(list(range(len(src_data))), src_data):
            src = blocks.vector_source_b(s[1])
            self.tb.connect(src, (op, s[0]))
        dst = blocks.vector_sink_b()
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_ii(self, src_data, exp_data, op):
        if False:
            print('Hello World!')
        for s in zip(list(range(len(src_data))), src_data):
            src = blocks.vector_source_i(s[1])
            self.tb.connect(src, (op, s[0]))
        dst = blocks.vector_sink_i()
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def test_xor_ss(self):
        if False:
            i = 10
            return i + 15
        src1_data = [1, 2, 3, 20484, 4432]
        src2_data = [8, 2, 1, 1288, 4357]
        expected_result = [9, 0, 2, 21772, 85]
        op = blocks.xor_ss()
        self.help_ss((src1_data, src2_data), expected_result, op)

    def test_xor_bb(self):
        if False:
            i = 10
            return i + 15
        src1_data = [1, 2, 3, 4, 80]
        src2_data = [8, 2, 1, 8, 5]
        expected_result = [9, 0, 2, 12, 85]
        op = blocks.xor_bb()
        self.help_bb((src1_data, src2_data), expected_result, op)

    def test_xor_ii(self):
        if False:
            print('Hello World!')
        src1_data = [1, 2, 3, 83886084, 285212752]
        src2_data = [8, 2, 1, 5242888, 285212677]
        expected_result = [9, 0, 2, 89128972, 85]
        op = blocks.xor_ii()
        self.help_ii((src1_data, src2_data), expected_result, op)

    def test_and_ss(self):
        if False:
            return 10
        src1_data = [1, 2, 3, 20484, 4432]
        src2_data = [8, 2, 1, 1288, 4357]
        expected_result = [0, 2, 1, 0, 4352]
        op = blocks.and_ss()
        self.help_ss((src1_data, src2_data), expected_result, op)

    def test_and_bb(self):
        if False:
            print('Hello World!')
        src1_data = [1, 2, 2, 3, 4, 80]
        src2_data = [8, 2, 2, 1, 8, 5]
        src3_data = [8, 2, 1, 1, 8, 5]
        expected_result = [0, 2, 0, 1, 0, 0]
        op = blocks.and_bb()
        self.help_bb((src1_data, src2_data, src3_data), expected_result, op)

    def test_and_ii(self):
        if False:
            return 10
        src1_data = [1, 2, 3, 1342197764, 285217104]
        src2_data = [8, 2, 1, 83887368, 285217029]
        expected_result = [0, 2, 1, 0, 285217024]
        op = blocks.and_ii()
        self.help_ii((src1_data, src2_data), expected_result, op)

    def test_and_const_ss(self):
        if False:
            return 10
        src_data = [1, 2, 3, 20484, 4432]
        expected_result = [0, 2, 2, 20480, 4352]
        src = blocks.vector_source_s(src_data)
        op = blocks.and_const_ss(21930)
        dst = blocks.vector_sink_s()
        self.tb.connect(src, op, dst)
        self.tb.run()
        self.assertEqual(dst.data(), expected_result)

    def test_and_const_bb(self):
        if False:
            while True:
                i = 10
        src_data = [1, 2, 3, 80, 17]
        expected_result = [0, 2, 2, 0, 0]
        src = blocks.vector_source_b(src_data)
        op = blocks.and_const_bb(170)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        self.assertEqual(dst.data(), expected_result)

    def test_and_const_ii(self):
        if False:
            print('Hello World!')
        src_data = [1, 2, 3, 20484, 4432]
        expected_result = [0, 2, 2, 20480, 4352]
        src = blocks.vector_source_i(src_data)
        op = blocks.and_const_ii(21930)
        dst = blocks.vector_sink_i()
        self.tb.connect(src, op, dst)
        self.tb.run()
        self.assertEqual(dst.data(), expected_result)

    def test_or_ss(self):
        if False:
            return 10
        src1_data = [1, 2, 3, 20484, 4432]
        src2_data = [8, 2, 1, 1288, 4357]
        expected_result = [9, 2, 3, 21772, 4437]
        op = blocks.or_ss()
        self.help_ss((src1_data, src2_data), expected_result, op)

    def test_or_bb(self):
        if False:
            print('Hello World!')
        src1_data = [1, 2, 2, 3, 4, 80]
        src2_data = [8, 2, 2, 1, 8, 5]
        src3_data = [8, 2, 1, 1, 8, 5]
        expected_result = [9, 2, 3, 3, 12, 85]
        op = blocks.or_bb()
        self.help_bb((src1_data, src2_data, src3_data), expected_result, op)

    def test_or_ii(self):
        if False:
            print('Hello World!')
        src1_data = [1, 2, 3, 1342197764, 285217104]
        src2_data = [8, 2, 1, 83887368, 285217029]
        expected_result = [9, 2, 3, 1426085132, 285217109]
        op = blocks.or_ii()
        self.help_ii((src1_data, src2_data), expected_result, op)

    def test_not_ss(self):
        if False:
            for i in range(10):
                print('nop')
        src1_data = [1, 2, 3, 20484, 4432]
        expected_result = [~1, ~2, ~3, ~20484, ~4432]
        op = blocks.not_ss()
        self.help_ss((src1_data,), expected_result, op)

    def test_not_bb(self):
        if False:
            print('Hello World!')
        src1_data = [1, 2, 2, 3, 4, 80]
        expected_result = [254, 253, 253, 252, 251, 175]
        op = blocks.not_bb()
        self.help_bb((src1_data,), expected_result, op)

    def test_not_ii(self):
        if False:
            for i in range(10):
                print('nop')
        src1_data = [1, 2, 3, 1342197764, 285217104]
        expected_result = [~1, ~2, ~3, ~1342197764, ~285217104]
        op = blocks.not_ii()
        self.help_ii((src1_data,), expected_result, op)
if __name__ == '__main__':
    gr_unittest.run(test_boolean_operators)