from unittest import TestCase, mock
from mycroft.messagebus import Message
from mycroft.skills.audioservice import AudioService

class TestAudioServiceControls(TestCase):

    def assertLastMessageTypeEqual(self, bus, msg_type):
        if False:
            i = 10
            return i + 15
        message = bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, msg_type)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.bus = mock.Mock(name='bus')
        self.audioservice = AudioService(self.bus)

    def test_pause(self):
        if False:
            i = 10
            return i + 15
        self.audioservice.pause()
        self.assertLastMessageTypeEqual(self.bus, 'mycroft.audio.service.pause')

    def test_resume(self):
        if False:
            while True:
                i = 10
        self.audioservice.resume()
        self.assertLastMessageTypeEqual(self.bus, 'mycroft.audio.service.resume')

    def test_next(self):
        if False:
            print('Hello World!')
        self.audioservice.next()
        self.assertLastMessageTypeEqual(self.bus, 'mycroft.audio.service.next')

    def test_prev(self):
        if False:
            print('Hello World!')
        self.audioservice.prev()
        self.assertLastMessageTypeEqual(self.bus, 'mycroft.audio.service.prev')

    def test_stop(self):
        if False:
            return 10
        self.audioservice.stop()
        self.assertLastMessageTypeEqual(self.bus, 'mycroft.audio.service.stop')

    def test_seek(self):
        if False:
            for i in range(10):
                print('nop')
        self.audioservice.seek()
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.seek_forward')
        self.assertEqual(message.data['seconds'], 1)
        self.audioservice.seek(5)
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.seek_forward')
        self.assertEqual(message.data['seconds'], 5)
        self.audioservice.seek(-5)
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.seek_backward')
        self.assertEqual(message.data['seconds'], 5)

class TestAudioServicePlay(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.bus = mock.Mock(name='bus')
        self.audioservice = AudioService(self.bus)

    def test_proper_uri(self):
        if False:
            i = 10
            return i + 15
        self.audioservice.play('file:///hello_nasty.mp3')
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.play')
        self.assertEqual(message.data['tracks'], ['file:///hello_nasty.mp3'])
        self.assertEqual(message.data['repeat'], False)

    def test_path(self):
        if False:
            print('Hello World!')
        self.audioservice.play('/hello_nasty.mp3')
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.play')
        self.assertEqual(message.data['tracks'], ['file:///hello_nasty.mp3'])
        self.assertEqual(message.data['repeat'], False)

    def test_tuple(self):
        if False:
            print('Hello World!')
        'Test path together with mimetype.'
        self.audioservice.play(('/hello_nasty.mp3', 'audio/mp3'))
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.play')
        self.assertEqual(message.data['tracks'], [('file:///hello_nasty.mp3', 'audio/mp3')])
        self.assertEqual(message.data['repeat'], False)

    def test_invalid(self):
        if False:
            return 10
        'Test play request with invalid type.'
        with self.assertRaises(ValueError):
            self.audioservice.play(12)

    def test_extra_arguments(self):
        if False:
            while True:
                i = 10
        'Test sending along utterance and setting repeat.'
        self.audioservice.play('/hello_nasty.mp3', 'on vlc', True)
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.play')
        self.assertEqual(message.data['tracks'], ['file:///hello_nasty.mp3'])
        self.assertEqual(message.data['repeat'], True)
        self.assertEqual(message.data['utterance'], 'on vlc')

class TestAudioServiceQueue(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.bus = mock.Mock(name='bus')
        self.audioservice = AudioService(self.bus)

    def test_uri(self):
        if False:
            for i in range(10):
                print('nop')
        self.audioservice.queue('file:///hello_nasty.mp3')
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.queue')
        self.assertEqual(message.data['tracks'], ['file:///hello_nasty.mp3'])

    def test_path(self):
        if False:
            while True:
                i = 10
        self.audioservice.queue('/hello_nasty.mp3')
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.queue')
        self.assertEqual(message.data['tracks'], ['file:///hello_nasty.mp3'])

    def test_tuple(self):
        if False:
            while True:
                i = 10
        self.audioservice.queue(('/hello_nasty.mp3', 'audio/mp3'))
        message = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(message.msg_type, 'mycroft.audio.service.queue')
        self.assertEqual(message.data['tracks'], [('file:///hello_nasty.mp3', 'audio/mp3')])

    def test_invalid(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            self.audioservice.queue(12)

class TestAudioServiceMisc(TestCase):

    def test_lifecycle(self):
        if False:
            return 10
        bus = mock.Mock(name='bus')
        audioservice = AudioService(bus)
        self.assertEqual(audioservice.bus, bus)

    def test_available_backends(self):
        if False:
            print('Hello World!')
        bus = mock.Mock(name='bus')
        audioservice = AudioService(bus)
        available_backends = {'simple': {'suported_uris': ['http', 'file'], 'default': True, 'remote': False}}
        bus.wait_for_response.return_value = Message('test_msg', available_backends)
        response = audioservice.available_backends()
        self.assertEqual(available_backends, response)
        bus.wait_for_response.return_value = None
        response = audioservice.available_backends()
        self.assertEqual({}, response)

    def test_track_info(self):
        if False:
            while True:
                i = 10
        'Test is_playing property.'
        bus = mock.Mock(name='bus')
        audioservice = AudioService(bus)
        info = {'album': 'Hello Nasty', 'artist': 'Beastie Boys', 'name': 'Intergalactic'}
        bus.wait_for_response.return_value = Message('test_msg', info)
        self.assertEqual(audioservice.track_info(), info)
        bus.wait_for_response.return_value = None
        self.assertEqual(audioservice.track_info(), {})

    def test_is_playing(self):
        if False:
            for i in range(10):
                print('nop')
        'Test is_playing property.'
        bus = mock.Mock(name='bus')
        audioservice = AudioService(bus)
        audioservice.track_info = mock.Mock()
        audioservice.track_info.return_value = {'track': 'one cool song'}
        self.assertTrue(audioservice.is_playing)
        audioservice.track_info.return_value = {}
        self.assertFalse(audioservice.is_playing)