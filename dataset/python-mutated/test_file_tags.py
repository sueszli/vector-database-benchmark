from gnuradio import gr
from gnuradio import blocks
import sys
import numpy

def main():
    if False:
        return 10
    data = numpy.arange(0, 32000, 1).tolist()
    trig = 100 * [0] + 100 * [1]
    src = blocks.vector_source_s(data, True)
    trigger = blocks.vector_source_s(trig, True)
    thr = blocks.throttle(gr.sizeof_short, 10000.0)
    ann = blocks.annotator_alltoall(1000000, gr.sizeof_short)
    tagger = blocks.burst_tagger(gr.sizeof_short)
    fsnk = blocks.tagged_file_sink(gr.sizeof_short, 1)
    tb = gr.top_block()
    tb.connect(src, thr, (tagger, 0))
    tb.connect(trigger, (tagger, 1))
    tb.connect(tagger, fsnk)
    tb.run()
if __name__ == '__main__':
    main()