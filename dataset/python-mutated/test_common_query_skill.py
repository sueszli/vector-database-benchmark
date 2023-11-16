from unittest import TestCase, mock
from mycroft.messagebus import Message
from mycroft.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel, CQSVisualMatchLevel
from test.unittests.mocks import AnyCallable

class TestCommonQuerySkill(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.skill = CQSTest()
        self.bus = mock.Mock(name='bus')
        self.skill.bind(self.bus)
        self.skill.config_core = {'enclosure': {'platform': 'mycroft_mark_1'}}

    def test_lifecycle(self):
        if False:
            for i in range(10):
                print('nop')
        'Test startup and shutdown.'
        skill = CQSTest()
        bus = mock.Mock(name='bus')
        skill.bind(bus)
        bus.on.assert_any_call('question:query', AnyCallable())
        bus.on.assert_any_call('question:action', AnyCallable())
        skill.shutdown()

    def test_common_test_skill_action(self):
        if False:
            print('Hello World!')
        'Test that the optional action is triggered.'
        query_action = self.bus.on.call_args_list[-1][0][1]
        query_action(Message('query:action', data={'phrase': "What's the meaning of life", 'skill_id': 'asdf'}))
        self.skill.CQS_action.assert_not_called()
        query_action(Message('query:action', data={'phrase': "What's the meaning of life", 'skill_id': 'CQSTest'}))
        self.skill.CQS_action.assert_called_once_with("What's the meaning of life", None)

class TestCommonQueryMatching(TestCase):
    """Tests for CQS_match_query_phrase."""

    def setUp(self):
        if False:
            return 10
        self.skill = CQSTest()
        self.bus = mock.Mock(name='bus')
        self.skill.bind(self.bus)
        self.skill.config_core = {'enclosure': {'platform': 'mycroft_mark_1'}}
        self.query_phrase = self.bus.on.call_args_list[-2][0][1]

    def test_failing_match_query_phrase(self):
        if False:
            print('Hello World!')
        self.skill.CQS_match_query_phrase.return_value = None
        self.query_phrase(Message('question:query', data={'phrase': "What's the meaning of life"}))
        extension = self.bus.emit.call_args_list[-2][0][0]
        self.assertEqual(extension.data['phrase'], "What's the meaning of life")
        self.assertEqual(extension.data['skill_id'], self.skill.skill_id)
        self.assertEqual(extension.data['searching'], True)
        response = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(response.data['phrase'], "What's the meaning of life")
        self.assertEqual(response.data['skill_id'], self.skill.skill_id)
        self.assertEqual(response.data['searching'], False)

    def test_successful_match_query_phrase(self):
        if False:
            return 10
        self.skill.CQS_match_query_phrase.return_value = ("What's the meaning of life", CQSMatchLevel.EXACT, '42')
        self.query_phrase(Message('question:query', data={'phrase': "What's the meaning of life"}))
        extension = self.bus.emit.call_args_list[-2][0][0]
        self.assertEqual(extension.data['phrase'], "What's the meaning of life")
        self.assertEqual(extension.data['skill_id'], self.skill.skill_id)
        self.assertEqual(extension.data['searching'], True)
        response = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(response.data['phrase'], "What's the meaning of life")
        self.assertEqual(response.data['skill_id'], self.skill.skill_id)
        self.assertEqual(response.data['answer'], '42')
        self.assertEqual(response.data['conf'], 1.12)

    def test_successful_visual_match_query_phrase(self):
        if False:
            while True:
                i = 10
        self.skill.gui.connected = True
        query_phrase = self.bus.on.call_args_list[-2][0][1]
        self.skill.CQS_match_query_phrase.return_value = ("What's the meaning of life", CQSVisualMatchLevel.EXACT, '42')
        query_phrase(Message('question:query', data={'phrase': "What's the meaning of life"}))
        extension = self.bus.emit.call_args_list[-2][0][0]
        self.assertEqual(extension.data['phrase'], "What's the meaning of life")
        self.assertEqual(extension.data['skill_id'], self.skill.skill_id)
        self.assertEqual(extension.data['searching'], True)
        response = self.bus.emit.call_args_list[-1][0][0]
        self.assertEqual(response.data['phrase'], "What's the meaning of life")
        self.assertEqual(response.data['skill_id'], self.skill.skill_id)
        self.assertEqual(response.data['answer'], '42')
        self.assertEqual(response.data['conf'], 1.2200000000000002)

class CQSTest(CommonQuerySkill):
    """Simple skill for testing the CommonQuerySkill"""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.CQS_match_query_phrase = mock.Mock(name='match_phrase')
        self.CQS_action = mock.Mock(name='selected_action')
        self.skill_id = 'CQSTest'
        self.gui = MockGUI()

    def CQS_match_query_phrase(self, phrase):
        if False:
            for i in range(10):
                print('nop')
        pass

    def CQS_action(self, phrase, data):
        if False:
            for i in range(10):
                print('nop')
        pass

class MockGUI:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.connected = False
        self.setup_default_handlers = AnyCallable