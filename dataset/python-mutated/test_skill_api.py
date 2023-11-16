from unittest import TestCase, mock
from mycroft import MycroftSkill
from mycroft.messagebus import Message
from mycroft.skills import skill_api_method
from mycroft.skills.api import SkillApi

class Skill(MycroftSkill):
    """Test skill with registered API methods."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.registered_methods = {}

    def add_event(self, event_type, func):
        if False:
            for i in range(10):
                print('nop')
        'Mock handler of add_event, simply storing type and method.\n\n        Used in testing to verify the wrapped methods\n        '
        self.registered_methods[event_type] = func

    @skill_api_method
    def test_method(self):
        if False:
            while True:
                i = 10
        'Documentation.'
        return True

    @skill_api_method
    def test_method2(self, arg):
        if False:
            while True:
                i = 10
        'Documentation.'
        return 'TestResult'

def load_test_skill():
    if False:
        for i in range(10):
            print('nop')
    'Helper for setting up the test skill.\n\n    Returns:\n        (MycroftSkill): created test skill\n    '
    bus = mock.Mock()
    test_skill = Skill()
    test_skill.skill_id = 'test_skill'
    test_skill.bind(bus)
    return test_skill

def create_skill_api_from_skill(skill):
    if False:
        i = 10
        return i + 15
    'Helper creating an api from a skill.\n\n    Args:\n        skill (MycroftSkill): Skill to create api from.\n\n    Returns:\n        (SkillApi): API for the skill.\n    '
    SkillApi.connect_bus(skill.bus)
    return SkillApi(skill.public_api)

class testSkillMethod(TestCase):
    """Tests for the MycroftSkill class API setup."""

    def test_public_api_event(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that public api event handler is created.'
        test_skill = load_test_skill()
        self.assertTrue('test_skill.public_api' in test_skill.registered_methods)

    def test_public_api(self):
        if False:
            while True:
                i = 10
        'Test that the public_api structure matches the decorators.'
        test_skill = load_test_skill()
        self.assertTrue('test_method' in test_skill.public_api)
        self.assertTrue('test_method2' in test_skill.public_api)
        self.assertEqual(test_skill.public_api['test_method']['help'], 'Documentation.')
        self.assertEqual(test_skill.public_api['test_method']['type'], '{}.{}'.format(test_skill.skill_id, 'test_method'))

    def test_public_api_method(self):
        if False:
            while True:
                i = 10
        'Verify message from wrapped api method.'
        test_skill = load_test_skill()
        api_method = test_skill.registered_methods['test_skill.test_method']
        call_msg = Message('test_skill.test_method', data={'args': [], 'kwargs': {}})
        api_method(call_msg)
        response = test_skill.bus.emit.call_args[0][0]
        self.assertEqual(response.data['result'], test_skill.test_method())

    def test_public_api_request(self):
        if False:
            for i in range(10):
                print('nop')
        "Test public api request handling.\n\n        Ensures that a request for the skill's available public api returns\n        expected content.\n        "
        test_skill = load_test_skill()
        sent_message = None

        def capture_sent_message(message):
            if False:
                print('Hello World!')
            'Capture sent message.'
            nonlocal sent_message
            sent_message = message
        test_skill.bus.emit.side_effect = capture_sent_message
        get_api_method = test_skill.registered_methods['test_skill.public_api']
        request_api_msg = Message('test_skill.public_api')
        get_api_method(request_api_msg)
        public_api = sent_message.data
        self.assertTrue('test_method' in public_api)
        self.assertTrue('test_method2' in public_api)
        self.assertEqual(len(public_api), 2)

class TestApiObject(TestCase):
    """Tests for the generated SkillApi objects."""

    def test_create_api_object(self):
        if False:
            while True:
                i = 10
        'Check that expected methods are available.'
        test_skill = load_test_skill()
        test_api = create_skill_api_from_skill(test_skill)
        hasattr(test_api, 'test_method')
        hasattr(test_api, 'test_method2')

    def test_call_api_method(self):
        if False:
            print('Hello World!')
        'Ensure that calling the methods works as expected.'
        test_skill = load_test_skill()
        test_api = create_skill_api_from_skill(test_skill)
        expected_response = 'all is good'
        sent_message = None

        def capture_sent_message(message):
            if False:
                for i in range(10):
                    print('nop')
            'Capture sent message and return expected response message.'
            nonlocal sent_message
            sent_message = message
            return Message('', data={'result': expected_response})
        test_api.bus.wait_for_response.side_effect = capture_sent_message
        response = test_api.test_method('hello', person='you')
        self.assertEqual(response, expected_response)
        self.assertEqual(sent_message.msg_type, 'test_skill.test_method')
        self.assertEqual(sent_message.data['args'], ('hello',))
        self.assertEqual(sent_message.data['kwargs'], {'person': 'you'})