from gnuradio import gr
from gnuradio import blocks
from argparse import ArgumentParser
import sys

class my_graph(gr.top_block):

    def __init__(self):
        if False:
            return 10
        gr.top_block.__init__(self)
        parser = ArgumentParser()
        args = parser.parse_args()
        src = blocks.lfsr_32k_source_s()
        head = blocks.head(gr.sizeof_short, 2048)
        self.dst = blocks.vector_sink_s()
        self.connect(src, head, self.dst)
if __name__ == '__main__':
    try:
        tb = my_graph()
        tb.run()
        f = sys.stdout
        i = 0
        for s in tb.dst.data():
            f.write('%3d, ' % (s & 255,))
            f.write('%3d, ' % (s >> 8 & 255,))
            i = i + 2
            if i % 16 == 0:
                f.write('\n')
    except KeyboardInterrupt:
        pass