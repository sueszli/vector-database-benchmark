from gnuradio import gr, gr_unittest

class test_prefs(gr_unittest.TestCase):

    def test_001(self):
        if False:
            i = 10
            return i + 15
        p = gr.prefs()
        self.assertFalse(p.has_option('doesnt', 'exist'))
if __name__ == '__main__':
    gr_unittest.run(test_prefs)