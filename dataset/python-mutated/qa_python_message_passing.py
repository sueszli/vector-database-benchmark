from gnuradio import gr, gr_unittest, blocks
import pmt
import numpy
import time

class message_generator(gr.sync_block):

    def __init__(self, msg_list, msg_interval):
        if False:
            return 10
        gr.sync_block.__init__(self, name='message generator', in_sig=[numpy.float32], out_sig=None)
        self.msg_list = msg_list
        self.msg_interval = msg_interval
        self.msg_ctr = 0
        self.message_port_register_out(pmt.intern('out_port'))

    def work(self, input_items, output_items):
        if False:
            i = 10
            return i + 15
        inLen = len(input_items[0])
        while self.msg_ctr < len(self.msg_list) and self.msg_ctr * self.msg_interval < self.nitems_read(0) + inLen:
            self.message_port_pub(pmt.intern('out_port'), self.msg_list[self.msg_ctr])
            self.msg_ctr += 1
        return inLen

class message_consumer(gr.sync_block):

    def __init__(self):
        if False:
            while True:
                i = 10
        gr.sync_block.__init__(self, name='message consumer', in_sig=None, out_sig=None)
        self.msg_list = []
        self.message_port_register_in(pmt.intern('in_port'))
        self.set_msg_handler(pmt.intern('in_port'), self.handle_msg)

    def handle_msg(self, msg):
        if False:
            i = 10
            return i + 15
        self.msg_list.append(pmt.from_long(pmt.to_long(msg)))

class test_python_message_passing(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_000(self):
        if False:
            return 10
        num_msgs = 10
        msg_interval = 1000
        msg_list = []
        for i in range(num_msgs):
            msg_list.append(pmt.from_long(i))
        src_data = []
        for i in range(num_msgs * msg_interval):
            src_data.append(float(i))
        src = blocks.vector_source_f(src_data, False)
        msg_gen = message_generator(msg_list, msg_interval)
        msg_cons = message_consumer()
        self.tb.connect(src, msg_gen)
        self.tb.msg_connect(msg_gen, 'out_port', msg_cons, 'in_port')
        self.assertEqual(pmt.to_python(msg_gen.message_ports_out())[0], 'out_port')
        self.assertEqual('in_port' in pmt.to_python(msg_cons.message_ports_in()), True)
        self.tb.run()
        self.assertEqual(num_msgs, len(msg_cons.msg_list))
        for i in range(num_msgs):
            self.assertTrue(pmt.equal(msg_list[i], msg_cons.msg_list[i]))
if __name__ == '__main__':
    gr_unittest.run(test_python_message_passing)