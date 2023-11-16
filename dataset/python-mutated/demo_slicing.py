import sys
import os.path
from aubio import source, sink
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: %s <inputfile> <duration>' % sys.argv[0])
        sys.exit(1)
    source_file = sys.argv[1]
    duration = float(sys.argv[2])
    (source_base_name, source_ext) = os.path.splitext(os.path.basename(source_file))
    hopsize = 256
    (slice_n, total_frames_written, read) = (0, 0, hopsize)

    def new_sink_name(source_base_name, slice_n, duration=duration):
        if False:
            while True:
                i = 10
        return source_base_name + '_%02.3f' % (slice_n * duration) + '.wav'
    f = source(source_file, 0, hopsize)
    samplerate = f.samplerate
    g = sink(new_sink_name(source_base_name, slice_n), samplerate)
    while read == hopsize:
        (vec, read) = f()
        start_of_next_region = int(duration * samplerate * (slice_n + 1))
        remaining = start_of_next_region - total_frames_written
        if remaining <= read:
            g(vec[0:remaining], remaining)
            del g
            slice_n += 1
            g = sink(new_sink_name(source_base_name, slice_n), samplerate)
            g(vec[remaining:read], read - remaining)
        else:
            g(vec[0:read], read)
        total_frames_written += read
    total_duration = total_frames_written / float(samplerate)
    slice_n += 1
    outstr = 'created %(slice_n)s slices from %(source_base_name)s%(source_ext)s' % locals()
    outstr += ' (total duration %(total_duration).2fs)' % locals()
    print(outstr)
    del f, g