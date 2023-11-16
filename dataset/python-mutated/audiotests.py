from test.support import findfile
from test.support.os_helper import TESTFN, unlink
import array
import io
import pickle

class UnseekableIO(io.FileIO):

    def tell(self):
        if False:
            return 10
        raise io.UnsupportedOperation

    def seek(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise io.UnsupportedOperation

class AudioTests:
    close_fd = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.f = self.fout = None

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if self.f is not None:
            self.f.close()
        if self.fout is not None:
            self.fout.close()
        unlink(TESTFN)

    def check_params(self, f, nchannels, sampwidth, framerate, nframes, comptype, compname):
        if False:
            return 10
        self.assertEqual(f.getnchannels(), nchannels)
        self.assertEqual(f.getsampwidth(), sampwidth)
        self.assertEqual(f.getframerate(), framerate)
        self.assertEqual(f.getnframes(), nframes)
        self.assertEqual(f.getcomptype(), comptype)
        self.assertEqual(f.getcompname(), compname)
        params = f.getparams()
        self.assertEqual(params, (nchannels, sampwidth, framerate, nframes, comptype, compname))
        self.assertEqual(params.nchannels, nchannels)
        self.assertEqual(params.sampwidth, sampwidth)
        self.assertEqual(params.framerate, framerate)
        self.assertEqual(params.nframes, nframes)
        self.assertEqual(params.comptype, comptype)
        self.assertEqual(params.compname, compname)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            dump = pickle.dumps(params, proto)
            self.assertEqual(pickle.loads(dump), params)

class AudioWriteTests(AudioTests):

    def create_file(self, testfile):
        if False:
            while True:
                i = 10
        f = self.fout = self.module.open(testfile, 'wb')
        f.setnchannels(self.nchannels)
        f.setsampwidth(self.sampwidth)
        f.setframerate(self.framerate)
        f.setcomptype(self.comptype, self.compname)
        return f

    def check_file(self, testfile, nframes, frames):
        if False:
            while True:
                i = 10
        with self.module.open(testfile, 'rb') as f:
            self.assertEqual(f.getnchannels(), self.nchannels)
            self.assertEqual(f.getsampwidth(), self.sampwidth)
            self.assertEqual(f.getframerate(), self.framerate)
            self.assertEqual(f.getnframes(), nframes)
            self.assertEqual(f.readframes(nframes), frames)

    def test_write_params(self):
        if False:
            i = 10
            return i + 15
        f = self.create_file(TESTFN)
        f.setnframes(self.nframes)
        f.writeframes(self.frames)
        self.check_params(f, self.nchannels, self.sampwidth, self.framerate, self.nframes, self.comptype, self.compname)
        f.close()

    def test_write_context_manager_calls_close(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(self.module.Error):
            with self.module.open(TESTFN, 'wb'):
                pass
        with self.assertRaises(self.module.Error):
            with open(TESTFN, 'wb') as testfile:
                with self.module.open(testfile):
                    pass

    def test_context_manager_with_open_file(self):
        if False:
            i = 10
            return i + 15
        with open(TESTFN, 'wb') as testfile:
            with self.module.open(testfile) as f:
                f.setnchannels(self.nchannels)
                f.setsampwidth(self.sampwidth)
                f.setframerate(self.framerate)
                f.setcomptype(self.comptype, self.compname)
            self.assertEqual(testfile.closed, self.close_fd)
        with open(TESTFN, 'rb') as testfile:
            with self.module.open(testfile) as f:
                self.assertFalse(f.getfp().closed)
                params = f.getparams()
                self.assertEqual(params.nchannels, self.nchannels)
                self.assertEqual(params.sampwidth, self.sampwidth)
                self.assertEqual(params.framerate, self.framerate)
            if not self.close_fd:
                self.assertIsNone(f.getfp())
            self.assertEqual(testfile.closed, self.close_fd)

    def test_context_manager_with_filename(self):
        if False:
            print('Hello World!')
        with self.module.open(TESTFN, 'wb') as f:
            f.setnchannels(self.nchannels)
            f.setsampwidth(self.sampwidth)
            f.setframerate(self.framerate)
            f.setcomptype(self.comptype, self.compname)
        with self.module.open(TESTFN) as f:
            self.assertFalse(f.getfp().closed)
            params = f.getparams()
            self.assertEqual(params.nchannels, self.nchannels)
            self.assertEqual(params.sampwidth, self.sampwidth)
            self.assertEqual(params.framerate, self.framerate)
        if not self.close_fd:
            self.assertIsNone(f.getfp())

    def test_write(self):
        if False:
            for i in range(10):
                print('nop')
        f = self.create_file(TESTFN)
        f.setnframes(self.nframes)
        f.writeframes(self.frames)
        f.close()
        self.check_file(TESTFN, self.nframes, self.frames)

    def test_write_bytearray(self):
        if False:
            i = 10
            return i + 15
        f = self.create_file(TESTFN)
        f.setnframes(self.nframes)
        f.writeframes(bytearray(self.frames))
        f.close()
        self.check_file(TESTFN, self.nframes, self.frames)

    def test_write_array(self):
        if False:
            for i in range(10):
                print('nop')
        f = self.create_file(TESTFN)
        f.setnframes(self.nframes)
        f.writeframes(array.array('h', self.frames))
        f.close()
        self.check_file(TESTFN, self.nframes, self.frames)

    def test_write_memoryview(self):
        if False:
            while True:
                i = 10
        f = self.create_file(TESTFN)
        f.setnframes(self.nframes)
        f.writeframes(memoryview(self.frames))
        f.close()
        self.check_file(TESTFN, self.nframes, self.frames)

    def test_incompleted_write(self):
        if False:
            print('Hello World!')
        with open(TESTFN, 'wb') as testfile:
            testfile.write(b'ababagalamaga')
            f = self.create_file(testfile)
            f.setnframes(self.nframes + 1)
            f.writeframes(self.frames)
            f.close()
        with open(TESTFN, 'rb') as testfile:
            self.assertEqual(testfile.read(13), b'ababagalamaga')
            self.check_file(testfile, self.nframes, self.frames)

    def test_multiple_writes(self):
        if False:
            print('Hello World!')
        with open(TESTFN, 'wb') as testfile:
            testfile.write(b'ababagalamaga')
            f = self.create_file(testfile)
            f.setnframes(self.nframes)
            framesize = self.nchannels * self.sampwidth
            f.writeframes(self.frames[:-framesize])
            f.writeframes(self.frames[-framesize:])
            f.close()
        with open(TESTFN, 'rb') as testfile:
            self.assertEqual(testfile.read(13), b'ababagalamaga')
            self.check_file(testfile, self.nframes, self.frames)

    def test_overflowed_write(self):
        if False:
            i = 10
            return i + 15
        with open(TESTFN, 'wb') as testfile:
            testfile.write(b'ababagalamaga')
            f = self.create_file(testfile)
            f.setnframes(self.nframes - 1)
            f.writeframes(self.frames)
            f.close()
        with open(TESTFN, 'rb') as testfile:
            self.assertEqual(testfile.read(13), b'ababagalamaga')
            self.check_file(testfile, self.nframes, self.frames)

    def test_unseekable_read(self):
        if False:
            i = 10
            return i + 15
        with self.create_file(TESTFN) as f:
            f.setnframes(self.nframes)
            f.writeframes(self.frames)
        with UnseekableIO(TESTFN, 'rb') as testfile:
            self.check_file(testfile, self.nframes, self.frames)

    def test_unseekable_write(self):
        if False:
            for i in range(10):
                print('nop')
        with UnseekableIO(TESTFN, 'wb') as testfile:
            with self.create_file(testfile) as f:
                f.setnframes(self.nframes)
                f.writeframes(self.frames)
        self.check_file(TESTFN, self.nframes, self.frames)

    def test_unseekable_incompleted_write(self):
        if False:
            print('Hello World!')
        with UnseekableIO(TESTFN, 'wb') as testfile:
            testfile.write(b'ababagalamaga')
            f = self.create_file(testfile)
            f.setnframes(self.nframes + 1)
            try:
                f.writeframes(self.frames)
            except OSError:
                pass
            try:
                f.close()
            except OSError:
                pass
        with open(TESTFN, 'rb') as testfile:
            self.assertEqual(testfile.read(13), b'ababagalamaga')
            self.check_file(testfile, self.nframes + 1, self.frames)

    def test_unseekable_overflowed_write(self):
        if False:
            print('Hello World!')
        with UnseekableIO(TESTFN, 'wb') as testfile:
            testfile.write(b'ababagalamaga')
            f = self.create_file(testfile)
            f.setnframes(self.nframes - 1)
            try:
                f.writeframes(self.frames)
            except OSError:
                pass
            try:
                f.close()
            except OSError:
                pass
        with open(TESTFN, 'rb') as testfile:
            self.assertEqual(testfile.read(13), b'ababagalamaga')
            framesize = self.nchannels * self.sampwidth
            self.check_file(testfile, self.nframes - 1, self.frames[:-framesize])

class AudioTestsWithSourceFile(AudioTests):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.sndfilepath = findfile(cls.sndfilename, subdir='audiodata')

    def test_read_params(self):
        if False:
            print('Hello World!')
        f = self.f = self.module.open(self.sndfilepath)
        self.check_params(f, self.nchannels, self.sampwidth, self.framerate, self.sndfilenframes, self.comptype, self.compname)

    def test_close(self):
        if False:
            for i in range(10):
                print('nop')
        with open(self.sndfilepath, 'rb') as testfile:
            f = self.f = self.module.open(testfile)
            self.assertFalse(testfile.closed)
            f.close()
            self.assertEqual(testfile.closed, self.close_fd)
        with open(TESTFN, 'wb') as testfile:
            fout = self.fout = self.module.open(testfile, 'wb')
            self.assertFalse(testfile.closed)
            with self.assertRaises(self.module.Error):
                fout.close()
            self.assertEqual(testfile.closed, self.close_fd)
            fout.close()

    def test_read(self):
        if False:
            for i in range(10):
                print('nop')
        framesize = self.nchannels * self.sampwidth
        chunk1 = self.frames[:2 * framesize]
        chunk2 = self.frames[2 * framesize:4 * framesize]
        f = self.f = self.module.open(self.sndfilepath)
        self.assertEqual(f.readframes(0), b'')
        self.assertEqual(f.tell(), 0)
        self.assertEqual(f.readframes(2), chunk1)
        f.rewind()
        pos0 = f.tell()
        self.assertEqual(pos0, 0)
        self.assertEqual(f.readframes(2), chunk1)
        pos2 = f.tell()
        self.assertEqual(pos2, 2)
        self.assertEqual(f.readframes(2), chunk2)
        f.setpos(pos2)
        self.assertEqual(f.readframes(2), chunk2)
        f.setpos(pos0)
        self.assertEqual(f.readframes(2), chunk1)
        with self.assertRaises(self.module.Error):
            f.setpos(-1)
        with self.assertRaises(self.module.Error):
            f.setpos(f.getnframes() + 1)

    def test_copy(self):
        if False:
            return 10
        f = self.f = self.module.open(self.sndfilepath)
        fout = self.fout = self.module.open(TESTFN, 'wb')
        fout.setparams(f.getparams())
        i = 0
        n = f.getnframes()
        while n > 0:
            i += 1
            fout.writeframes(f.readframes(i))
            n -= i
        fout.close()
        fout = self.fout = self.module.open(TESTFN, 'rb')
        f.rewind()
        self.assertEqual(f.getparams(), fout.getparams())
        self.assertEqual(f.readframes(f.getnframes()), fout.readframes(fout.getnframes()))

    def test_read_not_from_start(self):
        if False:
            for i in range(10):
                print('nop')
        with open(TESTFN, 'wb') as testfile:
            testfile.write(b'ababagalamaga')
            with open(self.sndfilepath, 'rb') as f:
                testfile.write(f.read())
        with open(TESTFN, 'rb') as testfile:
            self.assertEqual(testfile.read(13), b'ababagalamaga')
            with self.module.open(testfile, 'rb') as f:
                self.assertEqual(f.getnchannels(), self.nchannels)
                self.assertEqual(f.getsampwidth(), self.sampwidth)
                self.assertEqual(f.getframerate(), self.framerate)
                self.assertEqual(f.getnframes(), self.sndfilenframes)
                self.assertEqual(f.readframes(self.nframes), self.frames)