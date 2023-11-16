from os.path import dirname, join, abspath
import unittest
import unittest.mock as mock
import mycroft.audio.audioservice as audio_service
from mycroft.messagebus import Message
from .services.working import WorkingBackend
'Tests for Audioservice class'
seek_message = Message('seek', data={'seconds': 5})

class MockEmitter:

    def __init__(self):
        if False:
            return 10
        self.reset()

    def once(self, event_type, function):
        if False:
            while True:
                i = 10
        pass

    def on(self, event_type, function):
        if False:
            return 10
        pass

    def emit(self, message):
        if False:
            while True:
                i = 10
        self.types.append(message.msg_type)
        self.results.append(message.data)

    def get_types(self):
        if False:
            i = 10
            return i + 15
        return self.types

    def get_results(self):
        if False:
            return 10
        return self.results

    def remove(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    def reset(self):
        if False:
            while True:
                i = 10
        self.types = []
        self.results = []

def setup_mock_backends(mock_load_services, emitter):
    if False:
        for i in range(10):
            print('nop')
    backend = WorkingBackend({}, emitter)
    second_backend = WorkingBackend({}, emitter, 'second')
    mock_load_services.return_value = [backend, second_backend]
    return (backend, second_backend)

class TestService(unittest.TestCase):
    emitter = MockEmitter()
    service_path = abspath(join(dirname(__file__), 'services'))

    def test_load(self):
        if False:
            return 10
        service = audio_service.load_services({}, TestService.emitter, TestService.service_path)
        self.assertEqual(len(service), 1)

    @mock.patch('mycroft.audio.audioservice.load_services')
    def test_audio_backend_shutdown(self, mock_load_services):
        if False:
            for i in range(10):
                print('nop')
        'Test shutdown of audio backend.'
        (backend, second_backend) = setup_mock_backends(mock_load_services, self.emitter)
        service = audio_service.AudioService(self.emitter)
        service.load_services()
        service.default = backend
        service.shutdown()
        self.assertTrue(backend.shutdown.called)
        self.assertTrue(second_backend.shutdown.called)

    @mock.patch('mycroft.audio.audioservice.load_services')
    def test_audio_service_track_start(self, mock_load_services):
        if False:
            i = 10
            return i + 15
        'Test start of new track messages.'
        (backend, second_backend) = setup_mock_backends(mock_load_services, self.emitter)
        service = audio_service.AudioService(self.emitter)
        service.load_services()
        service.default = backend
        self.emitter.reset()
        service.track_start('The universe song')
        service.track_start(None)
        self.assertEqual(self.emitter.types, ['mycroft.audio.playing_track', 'mycroft.audio.queue_end'])
        self.assertEqual(self.emitter.results, [{'track': 'The universe song'}, {}])
        service.shutdown()

    @mock.patch('mycroft.audio.audioservice.load_services')
    def test_audio_service_methods_not_playing(self, mock_load_services):
        if False:
            i = 10
            return i + 15
        "Check that backend methods aren't called when stopped."
        (backend, second_backend) = setup_mock_backends(mock_load_services, self.emitter)
        mock_load_services.return_value = [backend, second_backend]
        service = audio_service.AudioService(self.emitter)
        service.load_services()
        service.default = backend
        service._next()
        self.assertFalse(backend.next.called)
        service._prev()
        self.assertFalse(backend.previous.called)
        service._pause()
        self.assertFalse(backend.pause.called)
        service._resume()
        self.assertFalse(backend.resume.called)
        service._seek_forward(seek_message)
        self.assertFalse(backend.seek_forward.called)
        service._seek_backward(seek_message)
        self.assertFalse(backend.seek_backward.called)
        service._lower_volume()
        self.assertFalse(service.volume_is_low)
        self.assertFalse(backend.lower_volume.called)
        service._restore_volume()
        self.assertFalse(backend.lower_volume.called)
        service.shutdown()

    @mock.patch('mycroft.audio.audioservice.load_services')
    def test_audio_service_methods_playing(self, mock_load_services):
        if False:
            for i in range(10):
                print('nop')
        'Check that backend methods are called during playback.'
        (backend, second_backend) = setup_mock_backends(mock_load_services, self.emitter)
        mock_load_services.return_value = [backend, second_backend]
        service = audio_service.AudioService(self.emitter)
        service.load_services()
        service.default = backend
        m = Message('audio.service.play', data={'tracks': ['asdf://hello']})
        service._play(m)
        self.assertFalse(backend.play.called)
        m = Message('audio.service.play', data={'tracks': ['http://hello']})
        service._play(m)
        self.assertTrue(backend.play.called)
        service._next()
        self.assertTrue(backend.next.called)
        service._prev()
        self.assertTrue(backend.previous.called)
        service._pause()
        self.assertTrue(backend.pause.called)
        service._resume()
        self.assertTrue(backend.resume.called)
        service._lower_volume()
        self.assertTrue(service.volume_is_low)
        self.assertTrue(backend.lower_volume.called)
        service._restore_volume()
        self.assertFalse(service.volume_is_low)
        self.assertTrue(backend.lower_volume.called)
        m = Message('audio.service.play', data={'tracks': [['http://hello', 'audio/mp3']], 'utterance': 'using second'})
        service._play(m)
        self.assertTrue(second_backend.play.called)
        service._seek_forward(seek_message)
        second_backend.seek_forward.assert_called_with(5)
        service._seek_backward(seek_message)
        second_backend.seek_backward.assert_called_with(5)
        second_backend.stop.reset_mock()
        self.assertFalse(second_backend.stop.called)
        service._stop()
        self.assertFalse(second_backend.stop.called)
        service.play_start_time -= 1
        service._stop()
        self.assertTrue(second_backend.stop.called)
        service.shutdown()

    @mock.patch('mycroft.audio.audioservice.load_services')
    def test_audio_service_queue_methods(self, mock_load_services):
        if False:
            while True:
                i = 10
        'Check that backend methods are called during playback.'
        (backend, second_backend) = setup_mock_backends(mock_load_services, self.emitter)
        mock_load_services.return_value = [backend, second_backend]
        service = audio_service.AudioService(self.emitter)
        service.load_services()
        service.default = backend
        backend.play.reset_mock()
        backend.add_list.reset_mock()
        m = Message('audio.service.queue', data={'tracks': ['http://hello']})
        service._queue(m)
        backend.add_list.called_with(['http://hello'])
        self.assertTrue(backend.play.called)
        backend.play.reset_mock()
        backend.add_list.reset_mock()
        service._queue(m)
        backend.add_list.called_with(['http://hello'])
        self.assertFalse(backend.play.called)
        service.shutdown()
if __name__ == '__main__':
    unittest.main()