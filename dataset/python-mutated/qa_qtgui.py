from gnuradio import gr, gr_unittest, qtgui

class test_qtgui(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test01(self):
        if False:
            i = 10
            return i + 15
        self.qtsnk = qtgui.sink_c(1024, 5, 0.0, 1.0, 'Test', True, True, True, True, None)

    def test02(self):
        if False:
            print('Hello World!')
        self.qtsnk = qtgui.sink_f(1024, 5, 0, 1, 'Test', True, True, True, True, None)

    def test03(self):
        if False:
            i = 10
            return i + 15
        self.qtsnk = qtgui.time_sink_c(1024, 1, 'Test', 1, None)

    def test04(self):
        if False:
            i = 10
            return i + 15
        self.qtsnk = qtgui.time_sink_f(1024, 1, 'Test', 1, None)

    def test05(self):
        if False:
            print('Hello World!')
        self.qtsnk = qtgui.freq_sink_c(1024, 5, 0, 1, 'Test', 1, None)

    def test06(self):
        if False:
            while True:
                i = 10
        self.qtsnk = qtgui.freq_sink_f(1024, 5, 0, 1, 'Test', 1, None)

    def test07(self):
        if False:
            i = 10
            return i + 15
        self.qtsnk = qtgui.waterfall_sink_c(1024, 5, 0, 1, 'Test', 1, None)

    def test08(self):
        if False:
            while True:
                i = 10
        self.qtsnk = qtgui.waterfall_sink_f(1024, 5, 0, 1, 'Test', 1, None)

    def test09(self):
        if False:
            for i in range(10):
                print('nop')
        self.qtsnk = qtgui.const_sink_c(1024, 'Test', 1, None)

    def test10(self):
        if False:
            while True:
                i = 10
        self.qtsnk = qtgui.time_raster_sink_b(1024, 100, 100.5, [], [], 'Test', 1, None)

    def test11(self):
        if False:
            i = 10
            return i + 15
        self.qtsnk = qtgui.time_raster_sink_f(1024, 100, 100.5, [], [], 'Test', 1, None)

    def test12(self):
        if False:
            for i in range(10):
                print('nop')
        self.qtsnk = qtgui.histogram_sink_f(1024, 100, -1, 1, 'Test', 1, None)

    def test13(self):
        if False:
            print('Hello World!')
        self.qtsnk = qtgui.eye_sink_f(1024, 1, 1, None)

    def test14(self):
        if False:
            for i in range(10):
                print('nop')
        self.qtsnk = qtgui.eye_sink_c(1024, 1, 1, None)

    def test15(self):
        if False:
            while True:
                i = 10
        self.qtsnk = qtgui.matrix_sink('Doppler', 2, 4, False, 'rgb', 'BilinearInterpolation', None)
if __name__ == '__main__':
    gr_unittest.run(test_qtgui)