from gnuradio import gr, gr_unittest, digital, blocks
import numpy as np

class test_constellation_encoder(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_constellation_encoder_bc_bpsk(self):
        if False:
            i = 10
            return i + 15
        cnst = digital.constellation_bpsk()
        src_data = (1, 1, 0, 0, 1, 0, 1)
        const_map = [-1.0, 1.0]
        expected_result = [const_map[x] for x in src_data]
        src = blocks.vector_source_b(src_data)
        op = digital.constellation_encoder_bc(cnst.base())
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result)

    def test_constellation_encoder_bc_qpsk(self):
        if False:
            i = 10
            return i + 15
        cnst = digital.constellation_qpsk()
        src_data = (3, 1, 0, 2, 3, 2, 1)
        expected_result = [cnst.points()[x] for x in src_data]
        src = blocks.vector_source_b(src_data)
        op = digital.constellation_encoder_bc(cnst.base())
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result)

    def test_constellation_encoder_bc_qpsk_random(self):
        if False:
            for i in range(10):
                print('nop')
        cnst = digital.constellation_qpsk()
        src_data = np.random.randint(0, 4, size=20000)
        expected_result = [cnst.points()[x] for x in src_data]
        src = blocks.vector_source_b(src_data)
        op = digital.constellation_encoder_bc(cnst.base())
        dst = blocks.vector_sink_c()
        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        actual_result = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, actual_result)
if __name__ == '__main__':
    gr_unittest.run(test_constellation_encoder)