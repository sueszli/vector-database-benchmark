from gnuradio import gr
from gnuradio import audio
from gnuradio.eng_arg import eng_float
from argparse import ArgumentParser

class my_top_block(gr.top_block):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        gr.top_block.__init__(self)
        parser = ArgumentParser()
        parser.add_argument('-I', '--audio-input', default='', help='pcm input device name.  E.g., hw:0,0 or /dev/dsp')
        parser.add_argument('-O', '--audio-output', default='', help='pcm output device name.  E.g., hw:0,0 or /dev/dsp')
        parser.add_argument('-r', '--sample-rate', type=eng_float, default=48000, help='set sample rate, default=%(default)s')
        args = parser.parse_args()
        sample_rate = int(args.sample_rate)
        src = audio.source(sample_rate, args.audio_input)
        dst = audio.sink(sample_rate, args.audio_output)
        nchan = min(src.output_signature().max_streams(), dst.input_signature().max_streams())
        for i in range(nchan):
            self.connect((src, i), (dst, i))
if __name__ == '__main__':
    try:
        my_top_block().run()
    except KeyboardInterrupt:
        pass