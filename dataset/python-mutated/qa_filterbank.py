import time
import random
import math
from gnuradio import gr, gr_unittest, filter, blocks

def convolution(A, B):
    if False:
        print('Hello World!')
    '\n    Returns a convolution of the A and B vectors of length\n    len(A)-len(B).\n    '
    rs = []
    for i in range(len(B) - 1, len(A)):
        r = 0
        for (j, b) in enumerate(B):
            r += A[i - j] * b
        rs.append(r)
    return rs

class test_filterbank_vcvcf(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_000(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates nfilts sets of random complex data.\n        Generates two sets of random taps for each filter.\n        Applies one set of the random taps, gets some output,\n        applies the second set of random taps, gets some more output,\n        The output is then compared with a python-implemented\n        convolution.\n        '
        myrand = random.Random(123).random
        nfilts = 10
        ntaps = 5
        zero_filts1 = (3, 7)
        zero_filts2 = (1, 6, 9)
        ndatapoints = 100
        data_sets = []
        for i in range(0, nfilts):
            data_sets.append([myrand() - 0.5 + (myrand() - 0.5) * (0 + 1j) for k in range(0, ndatapoints)])
        data = []
        for dp in zip(*data_sets):
            data += dp
        taps1 = []
        taps2 = []
        for i in range(0, nfilts):
            if i in zero_filts1:
                taps1.append([0] * ntaps)
            else:
                taps1.append([myrand() - 0.5 for k in range(0, ntaps)])
            if i in zero_filts2:
                taps2.append([0] * ntaps)
            else:
                taps2.append([myrand() - 0.5 for k in range(0, ntaps)])
        results = []
        results2 = []
        for (ds, ts, ts2) in zip(data_sets, taps1, taps2):
            results.append(convolution(ds[-(len(ts) - 1):] + ds, ts))
            results2.append(convolution(ds[-(len(ts) - 1):] + ds, ts2))
        comb_results = []
        for rs in zip(*results):
            comb_results += rs
        comb_results2 = []
        for rs in zip(*results2):
            comb_results2 += rs
        src = blocks.vector_source_c(data, True, nfilts)
        fb = filter.filterbank_vcvcf(taps1)
        v2s = blocks.vector_to_stream(gr.sizeof_gr_complex, nfilts)
        s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, nfilts * ndatapoints)
        snk = blocks.probe_signal_vc(nfilts * ndatapoints)
        self.tb.connect(src, fb, v2s, s2v, snk)
        self.tb.start()
        all_zero = True
        outdata = None
        waittime = 0.001
        while not outdata or outdata[0] == 0:
            time.sleep(waittime)
            outdata = snk.level()
        fb.set_taps(taps2)
        outdata2 = None
        while not outdata2 or abs(outdata2[0] - outdata[0]) < 1e-06:
            time.sleep(waittime)
            outdata2 = snk.level()
        self.tb.stop()
        self.assertComplexTuplesAlmostEqual(comb_results, outdata, 6)
        self.assertComplexTuplesAlmostEqual(comb_results2, outdata2, 6)
if __name__ == '__main__':
    gr_unittest.run(test_filterbank_vcvcf)