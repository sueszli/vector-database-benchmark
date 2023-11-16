from gnuradio import gr, gr_unittest, analog, blocks

class test_fm_emph(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_001(self):
        if False:
            print('Hello World!')
        tb = self.tb
        src = analog.sig_source_f(48000, analog.GR_COS_WAVE, 5000.0, 1.0)
        op = analog.fm_preemph(fs=48000, tau=7.5e-05, fh=-1.0)
        head = blocks.head(gr.sizeof_float, 100)
        dst = blocks.vector_sink_f()
        tb.connect(src, op)
        tb.connect(op, head)
        tb.connect(head, dst)
        tb.run()

    def test_002(self):
        if False:
            for i in range(10):
                print('nop')
        tb = self.tb
        src = analog.sig_source_f(48000, analog.GR_COS_WAVE, 5000.0, 1.0)
        op = analog.fm_deemph(fs=48000, tau=7.5e-05)
        head = blocks.head(gr.sizeof_float, 100)
        dst = blocks.vector_sink_f()
        tb.connect(src, op)
        tb.connect(op, head)
        tb.connect(head, dst)
        tb.run()
if __name__ == '__main__':
    gr_unittest.run(test_fm_emph)