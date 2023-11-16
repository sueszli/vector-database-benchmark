import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame

class MixerMusicModuleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        pygame.mixer.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        pygame.mixer.quit()

    def setUp(cls):
        if False:
            print('Hello World!')
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()

    def test_load_mp3(self):
        if False:
            print('Hello World!')
        '|tags:music|'
        self.music_load('mp3')

    def test_load_ogg(self):
        if False:
            for i in range(10):
                print('nop')
        '|tags:music|'
        self.music_load('ogg')

    def test_load_wav(self):
        if False:
            print('Hello World!')
        '|tags:music|'
        self.music_load('wav')

    def music_load(self, format):
        if False:
            print('Hello World!')
        data_fname = example_path('data')
        path = os.path.join(data_fname, f'house_lo.{format}')
        if os.sep == '\\':
            path = path.replace('\\', '\\\\')
        umusfn = str(path)
        bmusfn = umusfn.encode()
        pygame.mixer.music.load(umusfn)
        pygame.mixer.music.load(bmusfn)

    def test_load_object(self):
        if False:
            print('Hello World!')
        'test loading music from file-like objects.'
        formats = ['ogg', 'wav']
        data_fname = example_path('data')
        for f in formats:
            path = os.path.join(data_fname, f'house_lo.{f}')
            if os.sep == '\\':
                path = path.replace('\\', '\\\\')
            bmusfn = path.encode()
            with open(bmusfn, 'rb') as musf:
                pygame.mixer.music.load(musf)

    def test_object_namehint(self):
        if False:
            return 10
        'test loading & queuing music from file-like objects with namehint argument.'
        formats = ['wav', 'ogg']
        data_fname = example_path('data')
        for f in formats:
            path = os.path.join(data_fname, f'house_lo.{f}')
            if os.sep == '\\':
                path = path.replace('\\', '\\\\')
            bmusfn = path.encode()
            with open(bmusfn, 'rb') as musf:
                pygame.mixer.music.load(musf, f)
            with open(bmusfn, 'rb') as musf:
                pygame.mixer.music.queue(musf, f)
            with open(bmusfn, 'rb') as musf:
                pygame.mixer.music.load(musf, namehint=f)
            with open(bmusfn, 'rb') as musf:
                pygame.mixer.music.queue(musf, namehint=f)

    def test_load_unicode(self):
        if False:
            i = 10
            return i + 15
        'test non-ASCII unicode path'
        import shutil
        ep = example_path('data')
        temp_file = os.path.join(ep, '你好.wav')
        org_file = os.path.join(ep, 'house_lo.wav')
        try:
            with open(temp_file, 'w') as f:
                pass
            os.remove(temp_file)
        except OSError:
            raise unittest.SkipTest('the path cannot be opened')
        shutil.copy(org_file, temp_file)
        try:
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.load(org_file)
        finally:
            os.remove(temp_file)

    def test_unload(self):
        if False:
            while True:
                i = 10
        import shutil
        import tempfile
        ep = example_path('data')
        org_file = os.path.join(ep, 'house_lo.wav')
        (tmpfd, tmppath) = tempfile.mkstemp('.wav')
        os.close(tmpfd)
        shutil.copy(org_file, tmppath)
        try:
            pygame.mixer.music.load(tmppath)
            pygame.mixer.music.unload()
        finally:
            os.remove(tmppath)

    def test_queue_mp3(self):
        if False:
            return 10
        'Ensures queue() accepts mp3 files.\n\n        |tags:music|\n        '
        filename = example_path(os.path.join('data', 'house_lo.mp3'))
        pygame.mixer.music.queue(filename)

    def test_queue_ogg(self):
        if False:
            while True:
                i = 10
        'Ensures queue() accepts ogg files.\n\n        |tags:music|\n        '
        filename = example_path(os.path.join('data', 'house_lo.ogg'))
        pygame.mixer.music.queue(filename)

    def test_queue_wav(self):
        if False:
            print('Hello World!')
        'Ensures queue() accepts wav files.\n\n        |tags:music|\n        '
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        pygame.mixer.music.queue(filename)

    def test_queue__multiple_calls(self):
        if False:
            return 10
        'Ensures queue() can be called multiple times.'
        ogg_file = example_path(os.path.join('data', 'house_lo.ogg'))
        wav_file = example_path(os.path.join('data', 'house_lo.wav'))
        pygame.mixer.music.queue(ogg_file)
        pygame.mixer.music.queue(wav_file)

    def test_queue__arguments(self):
        if False:
            i = 10
            return i + 15
        'Ensures queue() can be called with proper arguments.'
        wav_file = example_path(os.path.join('data', 'house_lo.wav'))
        pygame.mixer.music.queue(wav_file, loops=2)
        pygame.mixer.music.queue(wav_file, namehint='')
        pygame.mixer.music.queue(wav_file, '')
        pygame.mixer.music.queue(wav_file, '', 2)

    def test_queue__no_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures queue() correctly handles missing the file argument.'
        with self.assertRaises(TypeError):
            pygame.mixer.music.queue()

    def test_queue__invalid_sound_type(self):
        if False:
            return 10
        'Ensures queue() correctly handles invalid file types.'
        not_a_sound_file = example_path(os.path.join('data', 'city.png'))
        with self.assertRaises(pygame.error):
            pygame.mixer.music.queue(not_a_sound_file)

    def test_queue__invalid_filename(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures queue() correctly handles invalid filenames.'
        with self.assertRaises(pygame.error):
            pygame.mixer.music.queue('')

    def test_music_pause__unpause(self):
        if False:
            i = 10
            return i + 15
        'Ensure music has the correct position immediately after unpausing\n\n        |tags:music|\n        '
        filename = example_path(os.path.join('data', 'house_lo.mp3'))
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        time.sleep(0.05)
        pygame.mixer.music.pause()
        time.sleep(0.05)
        before_unpause = pygame.mixer.music.get_pos()
        pygame.mixer.music.unpause()
        after_unpause = pygame.mixer.music.get_pos()
        self.assertEqual(before_unpause, after_unpause)

    def test_stop(self):
        if False:
            return 10
        filename = example_path(os.path.join('data', 'house_lo.mp3'))
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        pygame.mixer.music.stop()
        self.assertEqual(pygame.mixer.music.get_busy(), False)

    def todo_test_rewind(self):
        if False:
            print('Hello World!')
        self.fail()

    def todo_test_get_pos(self):
        if False:
            i = 10
            return i + 15
        self.fail()

    @unittest.skipIf(os.environ.get('SDL_AUDIODRIVER') == 'disk', 'disk audio driver "playback" writing to disk is slow')
    def test_play__start_time(self):
        if False:
            return 10
        pygame.display.init()
        filename = example_path(os.path.join('data', 'house_lo.ogg'))
        pygame.mixer.music.load(filename)
        start_time_in_seconds = 6.0
        music_finished = False
        clock = pygame.time.Clock()
        start_time_in_ms = clock.tick()
        pygame.mixer.music.play(0, start=start_time_in_seconds)
        running = True
        while running:
            pygame.event.pump()
            if not (pygame.mixer.music.get_busy() or music_finished):
                music_finished = True
                time_to_finish = (clock.tick() - start_time_in_ms) // 1000
                self.assertEqual(time_to_finish, 1)
                running = False

    def test_play(self):
        if False:
            for i in range(10):
                print('nop')
        filename = example_path(os.path.join('data', 'house_lo.mp3'))
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        self.assertTrue(pygame.mixer.music.get_busy())
        pygame.mixer.music.stop()

    def todo_test_load(self):
        if False:
            for i in range(10):
                print('nop')
        self.fail()

    def test_get_volume(self):
        if False:
            i = 10
            return i + 15
        filename = example_path(os.path.join('data', 'house_lo.mp3'))
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        vol = pygame.mixer.music.get_volume()
        self.assertGreaterEqual(vol, 0)
        self.assertLessEqual(vol, 1)
        pygame.mixer.music.stop()

    def todo_test_set_endevent(self):
        if False:
            print('Hello World!')
        self.fail()

    def test_pause(self):
        if False:
            i = 10
            return i + 15
        self.music_load('ogg')
        self.assertFalse(pygame.mixer.music.get_busy())
        pygame.mixer.music.play()
        self.assertTrue(pygame.mixer.music.get_busy())
        pygame.mixer.music.pause()
        self.assertFalse(pygame.mixer.music.get_busy())

    def test_get_busy(self):
        if False:
            print('Hello World!')
        self.music_load('ogg')
        self.assertFalse(pygame.mixer.music.get_busy())
        pygame.mixer.music.play()
        self.assertTrue(pygame.mixer.music.get_busy())
        pygame.mixer.music.pause()
        self.assertFalse(pygame.mixer.music.get_busy())

    def todo_test_get_endevent(self):
        if False:
            return 10
        self.fail()

    def test_unpause(self):
        if False:
            print('Hello World!')
        filename = example_path(os.path.join('data', 'house_lo.mp3'))
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        self.assertTrue(pygame.mixer.music.get_busy())
        time.sleep(0.1)
        pygame.mixer.music.pause()
        self.assertFalse(pygame.mixer.music.get_busy())
        before = pygame.mixer.music.get_pos()
        pygame.mixer.music.unpause()
        after = pygame.mixer.music.get_pos()
        self.assertTrue(pygame.mixer.music.get_busy())
        self.assertTrue(before - 1 <= after <= before + 1)
        pygame.mixer.music.stop()

    def test_set_volume(self):
        if False:
            while True:
                i = 10
        filename = example_path(os.path.join('data', 'house_lo.mp3'))
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(0.5)
        vol = pygame.mixer.music.get_volume()
        self.assertEqual(vol, 0.5)
        pygame.mixer.music.stop()

    def todo_test_set_pos(self):
        if False:
            print('Hello World!')
        self.fail()

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        'issue #955. unload music whenever mixer.quit() is called'
        import tempfile
        import shutil
        testfile = example_path(os.path.join('data', 'house_lo.wav'))
        tempcopy = os.path.join(tempfile.gettempdir(), 'tempfile.wav')
        for i in range(10):
            pygame.mixer.init()
            try:
                shutil.copy2(testfile, tempcopy)
                pygame.mixer.music.load(tempcopy)
                pygame.mixer.quit()
            finally:
                os.remove(tempcopy)
if __name__ == '__main__':
    unittest.main()