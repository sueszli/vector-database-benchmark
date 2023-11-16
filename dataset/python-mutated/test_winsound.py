import functools
import time
import unittest
from test import support
from test.support import import_helper
support.requires('audio')
winsound = import_helper.import_module('winsound')

def sound_func(func):
    if False:
        while True:
            i = 10

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        try:
            ret = func(*args, **kwargs)
        except RuntimeError as e:
            if support.verbose:
                print(func.__name__, 'failed:', e)
        else:
            if support.verbose:
                print(func.__name__, 'returned')
            return ret
    return wrapper
safe_Beep = sound_func(winsound.Beep)
safe_MessageBeep = sound_func(winsound.MessageBeep)
safe_PlaySound = sound_func(winsound.PlaySound)

class BeepTest(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, winsound.Beep)
        self.assertRaises(ValueError, winsound.Beep, 36, 75)
        self.assertRaises(ValueError, winsound.Beep, 32768, 75)

    def test_extremes(self):
        if False:
            while True:
                i = 10
        safe_Beep(37, 75)
        safe_Beep(32767, 75)

    def test_increasingfrequency(self):
        if False:
            while True:
                i = 10
        for i in range(100, 2000, 100):
            safe_Beep(i, 75)

    def test_keyword_args(self):
        if False:
            for i in range(10):
                print('nop')
        safe_Beep(duration=75, frequency=2000)

class MessageBeepTest(unittest.TestCase):

    def tearDown(self):
        if False:
            while True:
                i = 10
        time.sleep(0.5)

    def test_default(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, winsound.MessageBeep, 'bad')
        self.assertRaises(TypeError, winsound.MessageBeep, 42, 42)
        safe_MessageBeep()

    def test_ok(self):
        if False:
            while True:
                i = 10
        safe_MessageBeep(winsound.MB_OK)

    def test_asterisk(self):
        if False:
            i = 10
            return i + 15
        safe_MessageBeep(winsound.MB_ICONASTERISK)

    def test_exclamation(self):
        if False:
            for i in range(10):
                print('nop')
        safe_MessageBeep(winsound.MB_ICONEXCLAMATION)

    def test_hand(self):
        if False:
            while True:
                i = 10
        safe_MessageBeep(winsound.MB_ICONHAND)

    def test_question(self):
        if False:
            for i in range(10):
                print('nop')
        safe_MessageBeep(winsound.MB_ICONQUESTION)

    def test_keyword_args(self):
        if False:
            print('Hello World!')
        safe_MessageBeep(type=winsound.MB_OK)

class PlaySoundTest(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, winsound.PlaySound)
        self.assertRaises(TypeError, winsound.PlaySound, 'bad', 'bad')
        self.assertRaises(RuntimeError, winsound.PlaySound, 'none', winsound.SND_ASYNC | winsound.SND_MEMORY)
        self.assertRaises(TypeError, winsound.PlaySound, b'bad', 0)
        self.assertRaises(TypeError, winsound.PlaySound, 'bad', winsound.SND_MEMORY)
        self.assertRaises(TypeError, winsound.PlaySound, 1, 0)
        self.assertRaises(ValueError, winsound.PlaySound, 'bad\x00', 0)

    def test_keyword_args(self):
        if False:
            for i in range(10):
                print('nop')
        safe_PlaySound(flags=winsound.SND_ALIAS, sound='SystemExit')

    def test_snd_memory(self):
        if False:
            print('Hello World!')
        with open(support.findfile('pluck-pcm8.wav', subdir='audiodata'), 'rb') as f:
            audio_data = f.read()
        safe_PlaySound(audio_data, winsound.SND_MEMORY)
        audio_data = bytearray(audio_data)
        safe_PlaySound(audio_data, winsound.SND_MEMORY)

    def test_snd_filename(self):
        if False:
            for i in range(10):
                print('nop')
        fn = support.findfile('pluck-pcm8.wav', subdir='audiodata')
        safe_PlaySound(fn, winsound.SND_FILENAME | winsound.SND_NODEFAULT)

    def test_aliases(self):
        if False:
            i = 10
            return i + 15
        aliases = ['SystemAsterisk', 'SystemExclamation', 'SystemExit', 'SystemHand', 'SystemQuestion']
        for alias in aliases:
            with self.subTest(alias=alias):
                safe_PlaySound(alias, winsound.SND_ALIAS)

    def test_alias_fallback(self):
        if False:
            print('Hello World!')
        safe_PlaySound('!"$%&/(#+*', winsound.SND_ALIAS)

    def test_alias_nofallback(self):
        if False:
            print('Hello World!')
        safe_PlaySound('!"$%&/(#+*', winsound.SND_ALIAS | winsound.SND_NODEFAULT)

    def test_stopasync(self):
        if False:
            return 10
        safe_PlaySound('SystemQuestion', winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_LOOP)
        time.sleep(0.5)
        safe_PlaySound('SystemQuestion', winsound.SND_ALIAS | winsound.SND_NOSTOP)
        winsound.PlaySound(None, winsound.SND_PURGE)
if __name__ == '__main__':
    unittest.main()