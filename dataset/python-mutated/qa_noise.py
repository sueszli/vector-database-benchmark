from gnuradio import gr, gr_unittest, analog

class test_noise_source(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_001_instantiate(self):
        if False:
            return 10
        analog.noise_source_f(analog.GR_GAUSSIAN, 10, 10)
        analog.noise_source_f(analog.GR_GAUSSIAN, 10, -10)
        analog.noise_source_f(analog.GR_GAUSSIAN, 10, -2 ** 63)
        analog.noise_source_f(analog.GR_GAUSSIAN, 10, 2 ** 64 - 1)

    def test_002_getters(self):
        if False:
            i = 10
            return i + 15
        set_type = analog.GR_GAUSSIAN
        set_ampl = 10
        op = analog.noise_source_f(set_type, set_ampl, 10)
        get_type = op.type()
        get_ampl = op.amplitude()
        self.assertEqual(get_type, set_type)
        self.assertEqual(get_ampl, set_ampl)
if __name__ == '__main__':
    gr_unittest.run(test_noise_source)