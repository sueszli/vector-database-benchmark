from unittest import TestCase, mock
from mycroft.messagebus import Message
from mycroft.skills.common_play_skill import CommonPlaySkill, CPSMatchLevel
from mycroft.skills.audioservice import AudioService
from test.unittests.mocks import AnyCallable

class TestCommonPlay(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.skill = CPSTest()
        self.bus = mock.Mock(name='bus')
        self.skill.bind(self.bus)
        self.audioservice = mock.Mock(name='audioservice')
        self.skill.audioservice = self.audioservice

    def test_lifecycle(self):
        if False:
            for i in range(10):
                print('nop')
        skill = CPSTest()
        bus = mock.Mock(name='bus')
        skill.bind(bus)
        self.assertTrue(isinstance(skill.audioservice, AudioService))
        bus.on.assert_any_call('play:query', AnyCallable())
        bus.on.assert_any_call('play:start', AnyCallable())
        skill.shutdown()

    def test_handle_start_playback(self):
        if False:
            while True:
                i = 10
        'Test common play start method.'
        self.skill.audioservice.is_playing = True
        start_playback = self.bus.on.call_args_list[-1][0][1]
        phrase = "Don't open until doomsday"
        start_playback(Message('play:start', data={'phrase': phrase, 'skill_id': 'asdf'}))
        self.skill.CPS_start.assert_not_called()
        self.bus.emit.reset_mock()
        start_playback(Message('play:start', data={'phrase': phrase, 'skill_id': self.skill.skill_id}))
        self.audioservice.stop.assert_called_once_with()
        self.skill.CPS_start.assert_called_once_with(phrase, None)

    def test_cps_play(self):
        if False:
            return 10
        'Test audioservice play helper.'
        self.skill.play_service_string = 'play on godzilla'
        self.skill.CPS_play(['looking_for_freedom.mp3'], utterance='play on mothra')
        self.audioservice.play.assert_called_once_with(['looking_for_freedom.mp3'], utterance='play on mothra')
        self.audioservice.play.reset_mock()
        self.skill.CPS_play(['looking_for_freedom.mp3'])
        self.audioservice.play.assert_called_once_with(['looking_for_freedom.mp3'], utterance='play on godzilla')

    def test_stop(self):
        if False:
            while True:
                i = 10
        'Test default reaction to stop command.'
        self.audioservice.is_playing = False
        self.assertFalse(self.skill.stop())
        self.audioservice.is_playing = True
        self.assertTrue(self.skill.stop())

class TestCPSQuery(TestCase):

    def setUp(self):
        if False:
            return 10
        self.skill = CPSTest()
        self.bus = mock.Mock(name='bus')
        self.skill.bind(self.bus)
        self.audioservice = mock.Mock(name='audioservice')
        self.skill.audioservice = self.audioservice
        self.query_phrase = self.bus.on.call_args_list[-2][0][1]

    def test_handle_play_query_no_match(self):
        if False:
            return 10
        'Test common play match when no match is found.'
        self.skill.CPS_match_query_phrase.return_value = None
        self.query_phrase(Message('play:query', data={'phrase': 'Monster mash'}))
        extension = self.bus.emit.call_args_list[-2][0][0]
        self.assertEqual(extension.data['phrase'], 'Monster mash')
        self.assertEqual(extension.data['skill_id'], self.skill.skill_id)
        self.assertEqual(extension.data['searching'], True)
        response = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(response.data['phrase'], 'Monster mash')
        self.assertEqual(response.data['skill_id'], self.skill.skill_id)
        self.assertEqual(response.data['searching'], False)

    def test_play_query_match(self):
        if False:
            while True:
                i = 10
        'Test common play match when a match is found.'
        phrase = "Don't open until doomsday"
        self.skill.CPS_match_query_phrase.return_value = (phrase, CPSMatchLevel.TITLE)
        self.query_phrase(Message('play:query', data={'phrase': phrase}))
        response = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(response.data['phrase'], phrase)
        self.assertEqual(response.data['skill_id'], self.skill.skill_id)
        self.assertAlmostEqual(response.data['conf'], 0.85)
        self.skill.CPS_match_query_phrase.return_value = ('until doomsday', CPSMatchLevel.TITLE)
        self.query_phrase(Message('play:query', data={'phrase': phrase}))
        response = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(response.data['phrase'], phrase)
        self.assertEqual(response.data['skill_id'], self.skill.skill_id)
        self.assertAlmostEqual(response.data['conf'], 0.825)

class CPSTest(CommonPlaySkill):
    """Simple skill for testing the CommonPlaySkill"""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.CPS_match_query_phrase = mock.Mock(name='match_phrase')
        self.CPS_start = mock.Mock(name='start_playback')
        self.skill_id = 'CPSTest'

    def CPS_match_query_phrase(self, phrase):
        if False:
            for i in range(10):
                print('nop')
        pass

    def CPS_start(self, data):
        if False:
            for i in range(10):
                print('nop')
        pass