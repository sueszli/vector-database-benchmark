import os
import tempfile
import array
from gnuradio import gr, gr_unittest, blocks

class test_file_sink(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        os.environ['GR_CONF_CONTROLPORT_ON'] = 'False'
        self.tb = gr.top_block()
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.close()
        self._datafilename = temp.name

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None
        os.unlink(self._datafilename)

    def test_file_sink(self):
        if False:
            i = 10
            return i + 15
        data = range(1000)
        expected_result = data
        src = blocks.vector_source_f(data)
        snk = blocks.file_sink(gr.sizeof_float, self._datafilename)
        snk.set_unbuffered(True)
        self.tb.connect(src, snk)
        self.tb.run()
        snk.close()
        file_size = os.stat(self._datafilename).st_size
        self.assertEqual(file_size, 4 * len(data))
        result_data = array.array('f')
        with open(self._datafilename, 'rb') as datafile:
            result_data.fromfile(datafile, len(data))
        self.assertFloatTuplesAlmostEqual(expected_result, result_data)
if __name__ == '__main__':
    gr_unittest.run(test_file_sink)