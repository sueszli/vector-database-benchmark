from gnuradio import gr, gr_unittest, video_sdl

class test_video_sdl(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_000_nop(self):
        if False:
            print('Hello World!')
        "Just see if we can import the module...\n        They may not have video drivers, etc.  Don't try to run anything"
        pass
if __name__ == '__main__':
    gr_unittest.run(test_video_sdl)