from gnuradio import gr, gr_unittest, fft, blocks
primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311)
primes_transformed = (4377 + 4516j, -1706.1268310546875 + 1638.4256591796875j, -915.2083740234375 + 660.6942749023438j, -660.370361328125 + 381.59600830078125j, -499.96044921875 + 238.4163055419922j, -462.2674865722656 + 152.8894805908203j, -377.9844055175781 + 77.5928955078125j, -346.85821533203125 + 47.15200424194336j, -295 + 20j, -286.3360900878906 - 22.257017135620117j, -271.5299987792969 - 33.08182144165039j, -224.6358642578125 - 67.01953887939453j, -244.24473571777344 - 91.52482604980469j, -203.09068298339844 - 108.54627227783203j, -198.4519500732422 - 115.90768432617188j, -182.97744750976562 - 128.12318420410156j, -167 - 180j, -130.33688354492188 - 173.83778381347656j, -141.19784545898438 - 190.28807067871094j, -111.09677124023438 - 214.4889678955078j, -70.03954315185547 - 242.4163055419922j, -68.96054077148438 - 228.30015563964844j, -53.04920196533203 - 291.4709777832031j, -28.695289611816406 - 317.6455383300781j, 57 - 300j, 45.301143646240234 - 335.6950988769531j, 91.93619537353516 - 373.3243713378906j, 172.0946502685547 - 439.275146484375j, 242.24473571777344 - 504.47515869140625j, 387.81732177734375 - 666.6788330078125j, 689.4855346679688 - 918.2142333984375j, 1646.539306640625 - 1694.1956787109375j)

class test_fft(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()
        self.fft_size = 32

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def assert_fft_ok2(self, expected_result, result_data):
        if False:
            print('Hello World!')
        expected_result = expected_result[:len(result_data)]
        self.assertComplexTuplesAlmostEqual2(expected_result, result_data, abs_eps=1e-09, rel_eps=0.0004)

    def test_forward(self):
        if False:
            return 10
        src_data = tuple([complex(primes[2 * i], primes[2 * i + 1]) for i in range(self.fft_size)])
        expected_result = primes_transformed
        src = blocks.vector_source_c(src_data)
        s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)
        op = fft.fft_vcc(self.fft_size, True, [], False)
        v2s = blocks.vector_to_stream(gr.sizeof_gr_complex, self.fft_size)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, s2v, op, v2s, dst)
        self.tb.run()
        result_data = dst.data()
        self.assert_fft_ok2(expected_result, result_data)

    def test_reverse(self):
        if False:
            print('Hello World!')
        src_data = tuple([x / self.fft_size for x in primes_transformed])
        expected_result = tuple([complex(primes[2 * i], primes[2 * i + 1]) for i in range(self.fft_size)])
        src = blocks.vector_source_c(src_data)
        s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)
        op = fft.fft_vcc(self.fft_size, False, [], False)
        v2s = blocks.vector_to_stream(gr.sizeof_gr_complex, self.fft_size)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, s2v, op, v2s, dst)
        self.tb.run()
        result_data = dst.data()
        self.assert_fft_ok2(expected_result, result_data)

    def test_multithreaded(self):
        if False:
            while True:
                i = 10
        src_data = tuple([x / self.fft_size for x in primes_transformed])
        expected_result = tuple([complex(primes[2 * i], primes[2 * i + 1]) for i in range(self.fft_size)])
        nthreads = 2
        src = blocks.vector_source_c(src_data)
        s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)
        op = fft.fft_vcc(self.fft_size, False, [], False, nthreads)
        v2s = blocks.vector_to_stream(gr.sizeof_gr_complex, self.fft_size)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, s2v, op, v2s, dst)
        self.tb.run()
        result_data = dst.data()
        self.assert_fft_ok2(expected_result, result_data)

    def test_window(self):
        if False:
            print('Hello World!')
        src_data = tuple([complex(primes[2 * i], primes[2 * i + 1]) for i in range(self.fft_size)])
        expected_result = (2238.9174 + 2310.475j, -1603.7416 - 466.742j, 116.7449 - 70.8553j, -13.9157 + 19.0855j, -4.8283 + 16.7025j, -43.7425 + 16.9871j, -16.1904 + 1.7494j, -32.3797 + 6.9964j, -13.5283 + 7.7721j, -24.3276 - 7.5378j, -29.2711 + 4.5709j, -2.7124 - 6.6307j, -33.5486 - 8.3485j, -8.3016 - 9.9534j, -18.859 - 8.3501j, -13.9092 - 1.1396j, -17.7626 - 26.9281j, 0.0182 - 8.9j, -19.9143 - 14.132j, -10.3073 - 15.5759j, 3.58 - 29.1835j, -7.5263 - 1.59j, -3.0392 - 31.7445j, -15.1355 - 33.6158j, 28.2345 - 11.4373j, -6.0055 - 27.0418j, 5.2074 - 21.2431j, 23.1617 - 31.861j, 13.6494 - 11.1982j, 14.7145 - 14.4113j, -60.0053 + 114.7418j, -440.1561 - 1632.9807j)
        window = fft.window.hamming(ntaps=self.fft_size)
        src = blocks.vector_source_c(src_data)
        s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)
        op = fft.fft_vcc(self.fft_size, True, window, False)
        v2s = blocks.vector_to_stream(gr.sizeof_gr_complex, self.fft_size)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, s2v, op, v2s, dst)
        self.tb.run()
        result_data = dst.data()
        self.assert_fft_ok2(expected_result, result_data)

    def test_reverse_window_shift(self):
        if False:
            i = 10
            return i + 15
        src_data = tuple([x / self.fft_size for x in primes_transformed])
        expected_result = (-74.8629 - 63.2502j, -3.5446 - 2.0365j, 2.9231 + 1.6827j, -2.7852 - 0.8613j, 2.4763 + 2.7881j, -2.7457 - 3.2602j, 4.7748 + 2.4145j, -2.8807 - 4.5313j, 5.9949 + 4.1976j, -6.1095 - 6.0681j, 5.2248 + 5.7743j, -6.0436 - 6.3773j, 9.7184 + 9.2482j, -8.2791 - 8.6507j, 6.3273 + 6.156j, -12.2841 - 12.4692j, 10.5816 + 10.0241j, -13.0312 - 11.9451j, 12.2983 + 13.3644j, -13.0372 - 14.0795j, 14.4682 + 13.3079j, -16.7673 - 16.7287j, 14.3946 + 11.5916j, -16.8368 - 21.3156j, 20.4528 + 16.8499j, -18.4075 - 18.2446j, 17.7507 + 19.2109j, -21.5207 - 20.7159j, 22.2183 + 19.8012j, -22.2144 - 20.0343j, 17.0359 + 17.691j, -91.8955 - 103.1093j)
        window = fft.window.hamming(ntaps=self.fft_size)
        src = blocks.vector_source_c(src_data)
        s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.fft_size)
        op = fft.fft_vcc(self.fft_size, False, window, True)
        v2s = blocks.vector_to_stream(gr.sizeof_gr_complex, self.fft_size)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, s2v, op, v2s, dst)
        self.tb.run()
        result_data = dst.data()
        self.assert_fft_ok2(expected_result, result_data)
if __name__ == '__main__':
    gr_unittest.run(test_fft)