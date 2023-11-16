import random
from gnuradio import gr, gr_unittest, digital, blocks

def make_random_int_list(L, min, max):
    if False:
        print('Hello World!')
    result = []
    for x in range(L):
        result.append(random.randint(min, max))
    return list(result)

class test_diff_encoder(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_diff_encdec_000(self):
        if False:
            i = 10
            return i + 15
        for modulus in (2, 4, 8):
            with self.subTest(modulus=modulus):
                src_data = make_random_int_list(40000, 0, modulus - 1)
                expected_result = src_data
                src = blocks.vector_source_b(src_data)
                enc = digital.diff_encoder_bb(modulus)
                dec = digital.diff_decoder_bb(modulus)
                dst = blocks.vector_sink_b()
                self.tb.connect(src, enc, dec, dst)
                self.tb.run()
                actual_result = dst.data()
                self.assertEqual(expected_result, actual_result)
if __name__ == '__main__':
    gr_unittest.run(test_diff_encoder)