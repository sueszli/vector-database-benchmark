from gnuradio import gr, gr_unittest, blocks
import os
from os.path import getsize
g_in_file = os.path.join(os.getenv('srcdir'), 'test_16bit_1chunk.wav')
g_in_file_normal = os.path.join(os.getenv('srcdir'), 'test_16bit_1chunk_normal.wav')
g_extra_header_offset = 36
g_extra_header_len = 22

class test_wavefile(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_001_checkwavread(self):
        if False:
            print('Hello World!')
        wf = blocks.wavfile_source(g_in_file)
        self.assertEqual(wf.sample_rate(), 8000)

    def test_002_checkwavcopy(self):
        if False:
            return 10
        infile = g_in_file
        outfile = 'test_out.wav'
        wf_in = blocks.wavfile_source(infile)
        wf_out = blocks.wavfile_sink(outfile, wf_in.channels(), wf_in.sample_rate(), blocks.FORMAT_WAV, blocks.FORMAT_PCM_16)
        self.tb.connect(wf_in, wf_out)
        self.tb.run()
        wf_out.close()
        import wave
        try:
            with wave.open(infile, 'rb') as f:
                pass
            with wave.open(outfile, 'rb') as f:
                pass
        except BaseException:
            raise AssertionError('Invalid WAV file')
        self.assertEqual(getsize(infile) - g_extra_header_len, getsize(outfile))
        with open(infile, 'rb') as f:
            in_data = bytearray(f.read())
        with open(outfile, 'rb') as f:
            out_data = bytearray(f.read())
        os.remove(outfile)
        in_data[4:8] = b'\x00\x00\x00\x00'
        out_data[4:8] = b'\x00\x00\x00\x00'
        self.assertEqual(in_data[:g_extra_header_offset] + in_data[g_extra_header_offset + g_extra_header_len:], out_data)

    def test_003_checkwav_append_copy(self):
        if False:
            i = 10
            return i + 15
        infile = g_in_file_normal
        outfile = 'test_out_append.wav'
        from shutil import copyfile
        copyfile(infile, outfile)
        wf_in = blocks.wavfile_source(infile)
        wf_out = blocks.wavfile_sink(outfile, wf_in.channels(), wf_in.sample_rate(), blocks.FORMAT_WAV, blocks.FORMAT_PCM_16, True)
        self.tb.connect(wf_in, wf_out)
        self.tb.run()
        wf_out.close()
        wf_in = blocks.wavfile_source(infile)
        halver = blocks.multiply_const_ff(0.5)
        wf_out = blocks.wavfile_sink(outfile, wf_in.channels(), wf_in.sample_rate(), blocks.FORMAT_WAV, blocks.FORMAT_PCM_16, True)
        self.tb.connect(wf_in, halver, wf_out)
        self.tb.run()
        wf_out.close()
        import wave
        try:
            with wave.open(infile, 'rb') as w_in:
                in_params = w_in.getparams()
                data_in = wav_read_frames(w_in)
            with wave.open(outfile, 'rb') as w_out:
                out_params = w_out.getparams()
                data_out = wav_read_frames(w_out)
        except BaseException:
            raise AssertionError('Invalid WAV file')
        expected_params = in_params._replace(nframes=3 * in_params.nframes)
        self.assertEqual(out_params, expected_params)
        self.assertEqual(data_in, data_out[:len(data_in)])
        self.assertEqual(data_in, data_out[len(data_in):2 * len(data_in)])
        data_in_halved = [int(round(d / 2)) for d in data_in]
        self.assertEqual(data_in_halved, data_out[2 * len(data_in):])
        os.remove(outfile)

    def test_003_checkwav_append_non_existent_should_error(self):
        if False:
            while True:
                i = 10
        outfile = 'no_file.wav'
        with self.assertRaisesRegex(RuntimeError, "Can't open WAV file."):
            blocks.wavfile_sink(outfile, 1, 44100, blocks.FORMAT_WAV, blocks.FORMAT_PCM_16, True)
        os.remove(outfile)

def wav_read_frames(w):
    if False:
        print('Hello World!')
    import struct

    def grouper(iterable, n):
        if False:
            i = 10
            return i + 15
        return list(zip(*[iter(iterable)] * n))
    assert w.getsampwidth() == 2
    return [struct.unpack('h', bytes(frame_g))[0] for frame_g in grouper(w.readframes(w.getnframes()), 2)]
if __name__ == '__main__':
    gr_unittest.run(test_wavefile)