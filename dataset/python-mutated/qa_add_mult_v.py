from gnuradio import gr, gr_unittest, blocks

class test_add_mult_v(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def help_ss(self, size, src_data, exp_data, op):
        if False:
            print('Hello World!')
        for s in zip(list(range(len(src_data))), src_data):
            src = blocks.vector_source_s(s[1])
            srcv = blocks.stream_to_vector(gr.sizeof_short, size)
            self.tb.connect(src, srcv)
            self.tb.connect(srcv, (op, s[0]))
        rhs = blocks.vector_to_stream(gr.sizeof_short, size)
        dst = blocks.vector_sink_s()
        self.tb.connect(op, rhs, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_ii(self, size, src_data, exp_data, op):
        if False:
            while True:
                i = 10
        for s in zip(list(range(len(src_data))), src_data):
            src = blocks.vector_source_i(s[1])
            srcv = blocks.stream_to_vector(gr.sizeof_int, size)
            self.tb.connect(src, srcv)
            self.tb.connect(srcv, (op, s[0]))
        rhs = blocks.vector_to_stream(gr.sizeof_int, size)
        dst = blocks.vector_sink_i()
        self.tb.connect(op, rhs, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_ff(self, size, src_data, exp_data, op):
        if False:
            print('Hello World!')
        for s in zip(list(range(len(src_data))), src_data):
            src = blocks.vector_source_f(s[1])
            srcv = blocks.stream_to_vector(gr.sizeof_float, size)
            self.tb.connect(src, srcv)
            self.tb.connect(srcv, (op, s[0]))
        rhs = blocks.vector_to_stream(gr.sizeof_float, size)
        dst = blocks.vector_sink_f()
        self.tb.connect(op, rhs, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_cc(self, size, src_data, exp_data, op):
        if False:
            return 10
        for s in zip(list(range(len(src_data))), src_data):
            src = blocks.vector_source_c(s[1])
            srcv = blocks.stream_to_vector(gr.sizeof_gr_complex, size)
            self.tb.connect(src, srcv)
            self.tb.connect(srcv, (op, s[0]))
        rhs = blocks.vector_to_stream(gr.sizeof_gr_complex, size)
        dst = blocks.vector_sink_c()
        self.tb.connect(op, rhs, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_const_ss(self, src_data, exp_data, op):
        if False:
            i = 10
            return i + 15
        src = blocks.vector_source_s(src_data)
        srcv = blocks.stream_to_vector(gr.sizeof_short, len(src_data))
        rhs = blocks.vector_to_stream(gr.sizeof_short, len(src_data))
        dst = blocks.vector_sink_s()
        self.tb.connect(src, srcv, op, rhs, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_const_ii(self, src_data, exp_data, op):
        if False:
            while True:
                i = 10
        src = blocks.vector_source_i(src_data)
        srcv = blocks.stream_to_vector(gr.sizeof_int, len(src_data))
        rhs = blocks.vector_to_stream(gr.sizeof_int, len(src_data))
        dst = blocks.vector_sink_i()
        self.tb.connect(src, srcv, op, rhs, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_const_ff(self, src_data, exp_data, op):
        if False:
            print('Hello World!')
        src = blocks.vector_source_f(src_data)
        srcv = blocks.stream_to_vector(gr.sizeof_float, len(src_data))
        rhs = blocks.vector_to_stream(gr.sizeof_float, len(src_data))
        dst = blocks.vector_sink_f()
        self.tb.connect(src, srcv, op, rhs, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def help_const_cc(self, src_data, exp_data, op):
        if False:
            i = 10
            return i + 15
        src = blocks.vector_source_c(src_data)
        srcv = blocks.stream_to_vector(gr.sizeof_gr_complex, len(src_data))
        rhs = blocks.vector_to_stream(gr.sizeof_gr_complex, len(src_data))
        dst = blocks.vector_sink_c()
        self.tb.connect(src, srcv, op, rhs, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(exp_data, result_data)

    def test_add_vss_one(self):
        if False:
            i = 10
            return i + 15
        src1_data = [1]
        src2_data = [2]
        src3_data = [3]
        expected_result = [6]
        op = blocks.add_ss(1)
        self.help_ss(1, (src1_data, src2_data, src3_data), expected_result, op)

    def test_add_vss_five(self):
        if False:
            while True:
                i = 10
        src1_data = [1, 2, 3, 4, 5]
        src2_data = [6, 7, 8, 9, 10]
        src3_data = [11, 12, 13, 14, 15]
        expected_result = [18, 21, 24, 27, 30]
        op = blocks.add_ss(5)
        self.help_ss(5, (src1_data, src2_data, src3_data), expected_result, op)

    def test_add_vii_one(self):
        if False:
            print('Hello World!')
        src1_data = [1]
        src2_data = [2]
        src3_data = [3]
        expected_result = [6]
        op = blocks.add_ii(1)
        self.help_ii(1, (src1_data, src2_data, src3_data), expected_result, op)

    def test_add_vii_five(self):
        if False:
            print('Hello World!')
        src1_data = [1, 2, 3, 4, 5]
        src2_data = [6, 7, 8, 9, 10]
        src3_data = [11, 12, 13, 14, 15]
        expected_result = [18, 21, 24, 27, 30]
        op = blocks.add_ii(5)
        self.help_ii(5, (src1_data, src2_data, src3_data), expected_result, op)

    def test_add_vff_one(self):
        if False:
            print('Hello World!')
        src1_data = [1.0]
        src2_data = [2.0]
        src3_data = [3.0]
        expected_result = [6.0]
        op = blocks.add_ff(1)
        self.help_ff(1, (src1_data, src2_data, src3_data), expected_result, op)

    def test_add_vff_five(self):
        if False:
            print('Hello World!')
        src1_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        src2_data = [6.0, 7.0, 8.0, 9.0, 10.0]
        src3_data = [11.0, 12.0, 13.0, 14.0, 15.0]
        expected_result = [18.0, 21.0, 24.0, 27.0, 30.0]
        op = blocks.add_ff(5)
        self.help_ff(5, (src1_data, src2_data, src3_data), expected_result, op)

    def test_add_vcc_one(self):
        if False:
            i = 10
            return i + 15
        src1_data = [1.0 + 2j]
        src2_data = [3.0 + 4j]
        src3_data = [5.0 + 6j]
        expected_result = [9.0 + 12j]
        op = blocks.add_cc(1)
        self.help_cc(1, (src1_data, src2_data, src3_data), expected_result, op)

    def test_add_vcc_five(self):
        if False:
            for i in range(10):
                print('nop')
        src1_data = [1.0 + 2j, 3.0 + 4j, 5.0 + 6j, 7.0 + 8j, 9.0 + 10j]
        src2_data = [11.0 + 12j, 13.0 + 14j, 15.0 + 16j, 17.0 + 18j, 19.0 + 20j]
        src3_data = [21.0 + 22j, 23.0 + 24j, 25.0 + 26j, 27.0 + 28j, 29.0 + 30j]
        expected_result = [33.0 + 36j, 39.0 + 42j, 45.0 + 48j, 51.0 + 54j, 57.0 + 60j]
        op = blocks.add_cc(5)
        self.help_cc(5, (src1_data, src2_data, src3_data), expected_result, op)

    def test_add_const_vss_one(self):
        if False:
            while True:
                i = 10
        src_data = [1]
        op = blocks.add_const_vss((2,))
        exp_data = [3]
        self.help_const_ss(src_data, exp_data, op)

    def test_add_const_vss_five(self):
        if False:
            while True:
                i = 10
        src_data = [1, 2, 3, 4, 5]
        op = blocks.add_const_vss((6, 7, 8, 9, 10))
        exp_data = [7, 9, 11, 13, 15]
        self.help_const_ss(src_data, exp_data, op)

    def test_add_const_vii_one(self):
        if False:
            return 10
        src_data = [1]
        op = blocks.add_const_vii((2,))
        exp_data = [3]
        self.help_const_ii(src_data, exp_data, op)

    def test_add_const_vii_five(self):
        if False:
            while True:
                i = 10
        src_data = [1, 2, 3, 4, 5]
        op = blocks.add_const_vii((6, 7, 8, 9, 10))
        exp_data = [7, 9, 11, 13, 15]
        self.help_const_ii(src_data, exp_data, op)

    def test_add_const_vff_one(self):
        if False:
            while True:
                i = 10
        src_data = [1.0]
        op = blocks.add_const_vff((2.0,))
        exp_data = [3.0]
        self.help_const_ff(src_data, exp_data, op)

    def test_add_const_vff_five(self):
        if False:
            for i in range(10):
                print('nop')
        src_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        op = blocks.add_const_vff((6.0, 7.0, 8.0, 9.0, 10.0))
        exp_data = [7.0, 9.0, 11.0, 13.0, 15.0]
        self.help_const_ff(src_data, exp_data, op)

    def test_add_const_vcc_one(self):
        if False:
            return 10
        src_data = [1.0 + 2j]
        op = blocks.add_const_vcc((2.0 + 3j,))
        exp_data = [3.0 + 5j]
        self.help_const_cc(src_data, exp_data, op)

    def test_add_const_vcc_five(self):
        if False:
            for i in range(10):
                print('nop')
        src_data = [1.0 + 2j, 3.0 + 4j, 5.0 + 6j, 7.0 + 8j, 9.0 + 10j]
        op = blocks.add_const_vcc((11.0 + 12j, 13.0 + 14j, 15.0 + 16j, 17.0 + 18j, 19.0 + 20j))
        exp_data = [12.0 + 14j, 16.0 + 18j, 20.0 + 22j, 24.0 + 26j, 28.0 + 30j]
        self.help_const_cc(src_data, exp_data, op)

    def test_multiply_vss_one(self):
        if False:
            i = 10
            return i + 15
        src1_data = [1]
        src2_data = [2]
        src3_data = [3]
        expected_result = [6]
        op = blocks.multiply_ss(1)
        self.help_ss(1, (src1_data, src2_data, src3_data), expected_result, op)

    def test_multiply_vss_five(self):
        if False:
            while True:
                i = 10
        src1_data = [1, 2, 3, 4, 5]
        src2_data = [6, 7, 8, 9, 10]
        src3_data = [11, 12, 13, 14, 15]
        expected_result = [66, 168, 312, 504, 750]
        op = blocks.multiply_ss(5)
        self.help_ss(5, (src1_data, src2_data, src3_data), expected_result, op)

    def test_multiply_vii_one(self):
        if False:
            return 10
        src1_data = [1]
        src2_data = [2]
        src3_data = [3]
        expected_result = [6]
        op = blocks.multiply_ii(1)
        self.help_ii(1, (src1_data, src2_data, src3_data), expected_result, op)

    def test_multiply_vii_five(self):
        if False:
            for i in range(10):
                print('nop')
        src1_data = [1, 2, 3, 4, 5]
        src2_data = [6, 7, 8, 9, 10]
        src3_data = [11, 12, 13, 14, 15]
        expected_result = [66, 168, 312, 504, 750]
        op = blocks.multiply_ii(5)
        self.help_ii(5, (src1_data, src2_data, src3_data), expected_result, op)

    def test_multiply_vff_one(self):
        if False:
            print('Hello World!')
        src1_data = [1.0]
        src2_data = [2.0]
        src3_data = [3.0]
        expected_result = [6.0]
        op = blocks.multiply_ff(1)
        self.help_ff(1, (src1_data, src2_data, src3_data), expected_result, op)

    def test_multiply_vff_five(self):
        if False:
            return 10
        src1_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        src2_data = [6.0, 7.0, 8.0, 9.0, 10.0]
        src3_data = [11.0, 12.0, 13.0, 14.0, 15.0]
        expected_result = [66.0, 168.0, 312.0, 504.0, 750.0]
        op = blocks.multiply_ff(5)
        self.help_ff(5, (src1_data, src2_data, src3_data), expected_result, op)

    def test_multiply_vcc_one(self):
        if False:
            while True:
                i = 10
        src1_data = [1.0 + 2j]
        src2_data = [3.0 + 4j]
        src3_data = [5.0 + 6j]
        expected_result = [-85 + 20j]
        op = blocks.multiply_cc(1)
        self.help_cc(1, (src1_data, src2_data, src3_data), expected_result, op)

    def test_multiply_vcc_five(self):
        if False:
            for i in range(10):
                print('nop')
        src1_data = [1.0 + 2j, 3.0 + 4j, 5.0 + 6j, 7.0 + 8j, 9.0 + 10j]
        src2_data = [11.0 + 12j, 13.0 + 14j, 15.0 + 16j, 17.0 + 18j, 19.0 + 20j]
        src3_data = [21.0 + 22j, 23.0 + 24j, 25.0 + 26j, 27.0 + 28j, 29.0 + 30j]
        expected_result = [-1021.0 + 428j, -2647.0 + 1754j, -4945.0 + 3704j, -8011.0 + 6374j, -11941.0 + 9860j]
        op = blocks.multiply_cc(5)
        self.help_cc(5, (src1_data, src2_data, src3_data), expected_result, op)

    def test_multiply_const_vss_one(self):
        if False:
            print('Hello World!')
        src_data = [2]
        op = blocks.multiply_const_vss((3,))
        exp_data = [6]
        self.help_const_ss(src_data, exp_data, op)

    def test_multiply_const_vss_five(self):
        if False:
            i = 10
            return i + 15
        src_data = [1, 2, 3, 4, 5]
        op = blocks.multiply_const_vss([6, 7, 8, 9, 10])
        exp_data = [6, 14, 24, 36, 50]
        self.help_const_ss(src_data, exp_data, op)

    def test_multiply_const_vii_one(self):
        if False:
            while True:
                i = 10
        src_data = [2]
        op = blocks.multiply_const_vii((3,))
        exp_data = [6]
        self.help_const_ii(src_data, exp_data, op)

    def test_multiply_const_vii_five(self):
        if False:
            for i in range(10):
                print('nop')
        src_data = [1, 2, 3, 4, 5]
        op = blocks.multiply_const_vii((6, 7, 8, 9, 10))
        exp_data = [6, 14, 24, 36, 50]
        self.help_const_ii(src_data, exp_data, op)

    def test_multiply_const_vff_one(self):
        if False:
            while True:
                i = 10
        src_data = [2.0]
        op = blocks.multiply_const_vff((3.0,))
        exp_data = [6.0]
        self.help_const_ff(src_data, exp_data, op)

    def test_multiply_const_vff_five(self):
        if False:
            print('Hello World!')
        src_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        op = blocks.multiply_const_vff((6.0, 7.0, 8.0, 9.0, 10.0))
        exp_data = [6.0, 14.0, 24.0, 36.0, 50.0]
        self.help_const_ff(src_data, exp_data, op)

    def test_multiply_const_vcc_one(self):
        if False:
            i = 10
            return i + 15
        src_data = [1.0 + 2j]
        op = blocks.multiply_const_vcc((2.0 + 3j,))
        exp_data = [-4.0 + 7j]
        self.help_const_cc(src_data, exp_data, op)

    def test_multiply_const_vcc_five(self):
        if False:
            for i in range(10):
                print('nop')
        src_data = [1.0 + 2j, 3.0 + 4j, 5.0 + 6j, 7.0 + 8j, 9.0 + 10j]
        op = blocks.multiply_const_vcc((11.0 + 12j, 13.0 + 14j, 15.0 + 16j, 17.0 + 18j, 19.0 + 20j))
        exp_data = [-13.0 + 34j, -17.0 + 94j, -21.0 + 170j, -25.0 + 262j, -29.0 + 370j]
        self.help_const_cc(src_data, exp_data, op)
if __name__ == '__main__':
    gr_unittest.run(test_add_mult_v)