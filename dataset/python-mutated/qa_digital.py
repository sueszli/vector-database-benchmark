from gnuradio import gr, gr_unittest, digital

class test_digital(gr_unittest.TestCase):

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
if __name__ == '__main__':
    gr_unittest.run(test_digital)