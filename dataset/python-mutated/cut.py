""" this file was written by Paul Brossier
  it is released under the GNU/GPL license.
"""
import sys
from aubio.cmd import AubioArgumentParser, _cut_slice

def aubio_cut_parser():
    if False:
        return 10
    parser = AubioArgumentParser()
    parser.add_input()
    parser.add_argument('-O', '--onset-method', action='store', dest='onset_method', default='default', metavar='<onset_method>', help='onset detection method [default=default]                     complexdomain|hfc|phase|specdiff|energy|kl|mkl')
    parser.add_argument('-b', '--beat', action='store_true', dest='beat', default=False, help='slice at beat locations')
    '\n    parser.add_argument("-S", "--silencecut",\n            action="store_true", dest="silencecut", default=False,\n            help="use silence locations")\n    parser.add_argument("-s", "--silence",\n            metavar = "<value>",\n            action="store", dest="silence", default=-70,\n            help="silence threshold [default=-70]")\n            '
    parser.add_buf_hop_size()
    parser.add_argument('-t', '--threshold', '--onset-threshold', metavar='<threshold>', type=float, action='store', dest='threshold', default=0.3, help='onset peak picking threshold [default=0.3]')
    parser.add_argument('-c', '--cut', action='store_true', dest='cut', default=False, help='cut input sound file at detected labels')
    parser.add_minioi()
    '\n    parser.add_argument("-D", "--delay",\n            action = "store", dest = "delay", type = float,\n            metavar = "<seconds>", default=0,\n            help="number of seconds to take back [default=system]                    default system delay is 3*hopsize/samplerate")\n    parser.add_argument("-C", "--dcthreshold",\n            metavar = "<value>",\n            action="store", dest="dcthreshold", default=1.,\n            help="onset peak picking DC component [default=1.]")\n    parser.add_argument("-L", "--localmin",\n            action="store_true", dest="localmin", default=False,\n            help="use local minima after peak detection")\n    parser.add_argument("-d", "--derivate",\n            action="store_true", dest="derivate", default=False,\n            help="derivate onset detection function")\n    parser.add_argument("-z", "--zerocross",\n            metavar = "<value>",\n            action="store", dest="zerothres", default=0.008,\n            help="zero-crossing threshold for slicing [default=0.00008]")\n    # plotting functions\n    parser.add_argument("-p", "--plot",\n            action="store_true", dest="plot", default=False,\n            help="draw plot")\n    parser.add_argument("-x", "--xsize",\n            metavar = "<size>",\n            action="store", dest="xsize", default=1.,\n            type=float, help="define xsize for plot")\n    parser.add_argument("-y", "--ysize",\n            metavar = "<size>",\n            action="store", dest="ysize", default=1.,\n            type=float, help="define ysize for plot")\n    parser.add_argument("-f", "--function",\n            action="store_true", dest="func", default=False,\n            help="print detection function")\n    parser.add_argument("-n", "--no-onsets",\n            action="store_true", dest="nplot", default=False,\n            help="do not plot detected onsets")\n    parser.add_argument("-O", "--outplot",\n            metavar = "<output_image>",\n            action="store", dest="outplot", default=None,\n            help="save plot to output.{ps,png}")\n    parser.add_argument("-F", "--spectrogram",\n            action="store_true", dest="spectro", default=False,\n            help="add spectrogram to the plot")\n    '
    parser.add_slicer_options()
    parser.add_verbose_help()
    return parser

def _cut_analyze(options):
    if False:
        print('Hello World!')
    hopsize = options.hop_size
    bufsize = options.buf_size
    samplerate = options.samplerate
    source_uri = options.source_uri
    from aubio import onset, tempo, source
    s = source(source_uri, samplerate, hopsize)
    if samplerate == 0:
        samplerate = s.samplerate
        options.samplerate = samplerate
    if options.beat:
        o = tempo(options.onset_method, bufsize, hopsize, samplerate=samplerate)
    else:
        o = onset(options.onset_method, bufsize, hopsize, samplerate=samplerate)
        if options.minioi:
            if options.minioi.endswith('ms'):
                o.set_minioi_ms(int(options.minioi[:-2]))
            elif options.minioi.endswith('s'):
                o.set_minioi_s(int(options.minioi[:-1]))
            else:
                o.set_minioi(int(options.minioi))
    o.set_threshold(options.threshold)
    timestamps = []
    total_frames = 0
    while True:
        (samples, read) = s()
        if o(samples):
            timestamps.append(o.get_last())
            if options.verbose:
                print('%.4f' % o.get_last_s())
        total_frames += read
        if read < hopsize:
            break
    del s
    return (timestamps, total_frames)

def main():
    if False:
        return 10
    parser = aubio_cut_parser()
    options = parser.parse_args()
    if not options.source_uri and (not options.source_uri2):
        sys.stderr.write('Error: no file name given\n')
        parser.print_help()
        sys.exit(1)
    elif options.source_uri2 is not None:
        options.source_uri = options.source_uri2
    (timestamps, total_frames) = _cut_analyze(options)
    duration = float(total_frames) / float(options.samplerate)
    base_info = '%(source_uri)s' % {'source_uri': options.source_uri}
    base_info += ' (total %(duration).2fs at %(samplerate)dHz)\n' % {'duration': duration, 'samplerate': options.samplerate}
    info = 'found %d timestamps in ' % len(timestamps)
    info += base_info
    sys.stderr.write(info)
    if options.cut:
        _cut_slice(options, timestamps)
        info = 'created %d slices from ' % len(timestamps)
        info += base_info
        sys.stderr.write(info)