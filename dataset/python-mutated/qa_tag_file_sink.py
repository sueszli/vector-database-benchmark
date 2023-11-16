from gnuradio import gr, gr_unittest, blocks
import os
import struct

class test_tag_file_sink(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_001(self):
        if False:
            for i in range(10):
                print('nop')
        src_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        trg_data = [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
        src = blocks.vector_source_i(src_data)
        trg = blocks.vector_source_s(trg_data)
        op = blocks.burst_tagger(gr.sizeof_int)
        snk = blocks.tagged_file_sink(gr.sizeof_int, 1)
        self.tb.connect(src, (op, 0))
        self.tb.connect(trg, (op, 1))
        self.tb.connect(op, snk)
        self.tb.run()
        file0 = 'file{0}_0_2.00000000.dat'.format(snk.unique_id())
        file1 = 'file{0}_1_6.00000000.dat'.format(snk.unique_id())
        outfile0 = open(file0, 'rb')
        outfile1 = open(file1, 'rb')
        data0 = outfile0.read(8)
        data1 = outfile1.read(8)
        outfile0.close()
        outfile1.close()
        os.remove(file0)
        os.remove(file1)
        idata0 = struct.unpack('ii', data0)
        idata1 = struct.unpack('ii', data1)
        self.assertEqual(idata0, (3, 4))
        self.assertEqual(idata1, (7, 8))
if __name__ == '__main__':
    gr_unittest.run(test_tag_file_sink)