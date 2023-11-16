import unittest
from unittest import mock
from mopidy import mixer

class MixerListenerTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.listener = mixer.MixerListener()

    def test_on_event_forwards_to_specific_handler(self):
        if False:
            print('Hello World!')
        self.listener.volume_changed = mock.Mock()
        self.listener.on_event('volume_changed', volume=60)
        self.listener.volume_changed.assert_called_with(volume=60)

    def test_listener_has_default_impl_for_volume_changed(self):
        if False:
            print('Hello World!')
        self.listener.volume_changed(volume=60)

    def test_listener_has_default_impl_for_mute_changed(self):
        if False:
            i = 10
            return i + 15
        self.listener.mute_changed(mute=True)