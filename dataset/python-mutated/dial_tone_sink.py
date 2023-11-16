from gnuradio import gr, audio
from gnuradio import blocks
from argparse import ArgumentParser

class dial_tone_sink(gr.top_block):

    def __init__(self, host, port, pkt_size, sample_rate, eof):
        if False:
            return 10
        gr.top_block.__init__(self, 'dial_tone_sink')
        udp = blocks.udp_source(gr.sizeof_float, host, port, pkt_size, eof=eof)
        sink = audio.sink(sample_rate)
        self.connect(udp, sink)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='local host name (domain name or IP address)')
    parser.add_argument('--port', type=int, default=65500, help='port value to listen to for connection')
    parser.add_argument('--packet-size', type=int, default=1472, help='packet size.')
    parser.add_argument('-r', '--sample-rate', type=int, default=8000, help='audio signal sample rate [default=%(default)r]')
    parser.add_argument('--no-eof', action='store_true', default=False, help="don't send EOF on disconnect")
    args = parser.parse_args()
    top_block = dial_tone_sink(args.host, args.port, args.packet_size, args.sample_rate, not args.no_eof)
    try:
        top_block.run()
    except KeyboardInterrupt:
        pass