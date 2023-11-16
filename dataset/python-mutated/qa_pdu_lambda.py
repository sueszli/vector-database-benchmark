from gnuradio import gr, gr_unittest
from gnuradio import pdu

class qa_pdu_lambda(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_smoketest(self):
        if False:
            print('Hello World!')
        instance = pdu.pdu_lambda(lambda uvec: uvec * 10, False)

    def test_001_descriptive_test_name(self):
        if False:
            return 10
        self.tb.run()
if __name__ == '__main__':
    gr_unittest.run(qa_pdu_lambda)