from unittest import TestCase, mock
from adapt.intent import IntentBuilder
from mycroft.configuration import Configuration
from mycroft.messagebus import Message
from mycroft.skills.intent_service import IntentService, _get_message_lang
from mycroft.skills.intent_services.adapt_service import ContextManager, AdaptIntent
from test.util import base_config
BASE_CONF = base_config()
BASE_CONF['lang'] = 'it-it'
NO_LANG_CONF = base_config()
NO_LANG_CONF.pop('lang')

class MockEmitter(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.reset()

    def emit(self, message):
        if False:
            i = 10
            return i + 15
        self.types.append(message.msg_type)
        self.results.append(message.data)

    def get_types(self):
        if False:
            return 10
        return self.types

    def get_results(self):
        if False:
            for i in range(10):
                print('nop')
        return self.results

    def reset(self):
        if False:
            return 10
        self.types = []
        self.results = []

class ContextManagerTest(TestCase):
    emitter = MockEmitter()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.context_manager = ContextManager(3)

    def test_add_context(self):
        if False:
            return 10
        entity = {'confidence': 1.0}
        context = 'TestContext'
        word = 'TestWord'
        entity['data'] = [(word, context)]
        entity['match'] = word
        entity['key'] = word
        self.assertEqual(len(self.context_manager.frame_stack), 0)
        self.context_manager.inject_context(entity)
        self.assertEqual(len(self.context_manager.frame_stack), 1)

    def test_remove_context(self):
        if False:
            return 10
        entity = {'confidence': 1.0}
        context = 'TestContext'
        word = 'TestWord'
        entity['data'] = [(word, context)]
        entity['match'] = word
        entity['key'] = word
        self.context_manager.inject_context(entity)
        self.assertEqual(len(self.context_manager.frame_stack), 1)
        self.context_manager.remove_context('TestContext')
        self.assertEqual(len(self.context_manager.frame_stack), 0)

def check_converse_request(message, skill_id):
    if False:
        i = 10
        return i + 15
    return message.msg_type == 'skill.converse.request' and message.data['skill_id'] == skill_id

class ConversationTest(TestCase):

    def setUp(self):
        if False:
            return 10
        bus = mock.Mock()
        self.intent_service = IntentService(bus)
        self.intent_service.add_active_skill('atari_skill')
        self.intent_service.add_active_skill('c64_skill')

    def test_converse(self):
        if False:
            return 10
        'Check that the _converse method reports if the utterance is handled.\n\n        Also check that the skill that handled the query is moved to the\n        top of the active skill list.\n        '

        def response(message, return_msg_type):
            if False:
                print('Hello World!')
            c64 = Message(return_msg_type, {'skill_id': 'c64_skill', 'result': False})
            atari = Message(return_msg_type, {'skill_id': 'atari_skill', 'result': True})
            msgs = {'c64_skill': c64, 'atari_skill': atari}
            return msgs[message.data['skill_id']]
        self.intent_service.bus.wait_for_response.side_effect = response
        hello = ['hello old friend']
        utterance_msg = Message('recognizer_loop:utterance', data={'lang': 'en-US', 'utterances': hello})
        result = self.intent_service._converse(hello, 'en-US', utterance_msg)
        self.intent_service.add_active_skill(result.skill_id)
        first_active_skill = self.intent_service.active_skills[0][0]
        self.assertEqual(first_active_skill, 'atari_skill')
        self.assertTrue(result)

    def test_converse_error(self):
        if False:
            i = 10
            return i + 15
        "Check that all skill IDs in the active_skills list are called.\n        even if there's an error.\n        "

        def response(message, return_msg_type):
            if False:
                return 10
            c64 = Message(return_msg_type, {'skill_id': 'c64_skill', 'result': False})
            amiga = Message(return_msg_type, {'skill_id': 'amiga_skill', 'error': 'skill id does not exist'})
            atari = Message(return_msg_type, {'skill_id': 'atari_skill', 'result': False})
            msgs = {'c64_skill': c64, 'atari_skill': atari, 'amiga_skill': amiga}
            return msgs[message.data['skill_id']]
        self.intent_service.add_active_skill('amiga_skill')
        self.intent_service.bus.wait_for_response.side_effect = response
        hello = ['hello old friend']
        utterance_msg = Message('recognizer_loop:utterance', data={'lang': 'en-US', 'utterances': hello})
        result = self.intent_service._converse(hello, 'en-US', utterance_msg)
        self.assertFalse(result)
        call_args = self.intent_service.bus.wait_for_response.call_args_list
        sent_skill_ids = [call[0][0].data['skill_id'] for call in call_args]
        self.assertEqual(sent_skill_ids, ['amiga_skill', 'c64_skill', 'atari_skill'])

    def test_reset_converse(self):
        if False:
            while True:
                i = 10
        'Check that a blank stt sends the reset signal to the skills.'

        def response(message, return_msg_type):
            if False:
                print('Hello World!')
            c64 = Message(return_msg_type, {'skill_id': 'c64_skill', 'error': 'skill id does not exist'})
            atari = Message(return_msg_type, {'skill_id': 'atari_skill', 'result': False})
            msgs = {'c64_skill': c64, 'atari_skill': atari}
            return msgs[message.data['skill_id']]
        reset_msg = Message('mycroft.speech.recognition.unknown', data={'lang': 'en-US'})
        self.intent_service.bus.wait_for_response.side_effect = response
        self.intent_service.reset_converse(reset_msg)
        wait_for_response_mock = self.intent_service.bus.wait_for_response
        c64_message = wait_for_response_mock.call_args_list[0][0][0]
        self.assertTrue(check_converse_request(c64_message, 'c64_skill'))
        atari_message = wait_for_response_mock.call_args_list[1][0][0]
        self.assertTrue(check_converse_request(atari_message, 'atari_skill'))
        first_active_skill = self.intent_service.active_skills[0][0]
        self.assertEqual(first_active_skill, 'atari_skill')

class TestLanguageExtraction(TestCase):

    @mock.patch.dict(Configuration._Configuration__config, BASE_CONF)
    def test_no_lang_in_message(self):
        if False:
            print('Hello World!')
        'No lang in message should result in lang from config.'
        msg = Message('test msg', data={})
        self.assertEqual(_get_message_lang(msg), 'it-it')

    @mock.patch.dict(Configuration._Configuration__config, NO_LANG_CONF)
    def test_no_lang_at_all(self):
        if False:
            for i in range(10):
                print('nop')
        'Not in message and not in config, should result in en-us.'
        msg = Message('test msg', data={})
        self.assertEqual(_get_message_lang(msg), 'en-us')

    @mock.patch.dict(Configuration._Configuration__config, BASE_CONF)
    def test_lang_exists(self):
        if False:
            while True:
                i = 10
        'Message has a lang code in data, it should be used.'
        msg = Message('test msg', data={'lang': 'de-de'})
        self.assertEqual(_get_message_lang(msg), 'de-de')
        msg = Message('test msg', data={'lang': 'sv-se'})
        self.assertEqual(_get_message_lang(msg), 'sv-se')

def create_old_style_vocab_msg(keyword, value):
    if False:
        while True:
            i = 10
    'Create a message for registering an adapt keyword.'
    return Message('register_vocab', {'start': value, 'end': keyword})

def create_vocab_msg(keyword, value):
    if False:
        for i in range(10):
            print('nop')
    'Create a message for registering an adapt keyword.'
    return Message('register_vocab', {'entity_value': value, 'entity_type': keyword})

def get_last_message(bus):
    if False:
        print('Hello World!')
    'Get last sent message on mock bus.'
    last = bus.emit.call_args
    return last[0][0]

class TestIntentServiceApi(TestCase):

    def setUp(self):
        if False:
            return 10
        self.intent_service = IntentService(mock.Mock())

    def setup_simple_adapt_intent(self, msg=create_vocab_msg('testKeyword', 'test')):
        if False:
            i = 10
            return i + 15
        self.intent_service.handle_register_vocab(msg)
        intent = IntentBuilder('skill:testIntent').require('testKeyword')
        msg = Message('register_intent', intent.__dict__)
        self.intent_service.handle_register_intent(msg)

    def test_keyword_backwards_compatibility(self):
        if False:
            return 10
        self.setup_simple_adapt_intent(create_old_style_vocab_msg('testKeyword', 'test'))
        msg = Message('intent.service.adapt.get', data={'utterance': 'test'})
        self.intent_service.handle_get_adapt(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent']['intent_type'], 'skill:testIntent')

    def test_get_adapt_intent(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_simple_adapt_intent()
        msg = Message('intent.service.adapt.get', data={'utterance': 'test'})
        self.intent_service.handle_get_adapt(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent']['intent_type'], 'skill:testIntent')

    def test_get_adapt_intent_no_match(self):
        if False:
            print('Hello World!')
        "Check that if the intent doesn't match at all None is returned."
        self.setup_simple_adapt_intent()
        msg = Message('intent.service.adapt.get', data={'utterance': 'five'})
        self.intent_service.handle_get_adapt(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent'], None)

    def test_get_intent(self):
        if False:
            print('Hello World!')
        'Check that the registered adapt intent is triggered.'
        self.setup_simple_adapt_intent()
        msg = Message('intent.service.adapt.get', data={'utterance': 'test'})
        self.intent_service.handle_get_intent(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent']['intent_type'], 'skill:testIntent')

    def test_get_intent_no_match(self):
        if False:
            while True:
                i = 10
        "Check that if the intent doesn't match at all None is returned."
        self.setup_simple_adapt_intent()
        msg = Message('intent.service.intent.get', data={'utterance': 'five'})
        self.intent_service.handle_get_intent(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent'], None)

    def test_get_intent_manifest(self):
        if False:
            for i in range(10):
                print('nop')
        "Check that if the intent doesn't match at all None is returned."
        self.setup_simple_adapt_intent()
        msg = Message('intent.service.intent.get', data={'utterance': 'five'})
        self.intent_service.handle_get_intent(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent'], None)

    def test_get_adapt_intent_manifest(self):
        if False:
            return 10
        'Make sure the manifest returns a list of Intent Parser objects.'
        self.setup_simple_adapt_intent()
        msg = Message('intent.service.adapt.manifest.get')
        self.intent_service.handle_adapt_manifest(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intents'][0]['name'], 'skill:testIntent')

    def test_get_adapt_vocab_manifest(self):
        if False:
            while True:
                i = 10
        self.setup_simple_adapt_intent()
        msg = Message('intent.service.adapt.vocab.manifest.get')
        self.intent_service.handle_vocab_manifest(msg)
        reply = get_last_message(self.intent_service.bus)
        value = reply.data['vocab'][0]['entity_value']
        keyword = reply.data['vocab'][0]['entity_type']
        self.assertEqual(keyword, 'testKeyword')
        self.assertEqual(value, 'test')

    def test_get_no_match_after_detach(self):
        if False:
            return 10
        "Check that a removed intent doesn't match."
        self.setup_simple_adapt_intent()
        msg = Message('detach_intent', data={'intent_name': 'skill:testIntent'})
        self.intent_service.handle_detach_intent(msg)
        msg = Message('intent.service.adapt.get', data={'utterance': 'test'})
        self.intent_service.handle_get_adapt(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent'], None)

    def test_get_no_match_after_detach_skill(self):
        if False:
            print('Hello World!')
        "Check that a removed skill's intent doesn't match."
        self.setup_simple_adapt_intent()
        msg = Message('detach_intent', data={'skill_id': 'skill'})
        self.intent_service.handle_detach_skill(msg)
        msg = Message('intent.service.adapt.get', data={'utterance': 'test'})
        self.intent_service.handle_get_adapt(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent'], None)

class TestAdaptIntent(TestCase):
    """Test the AdaptIntent wrapper."""

    def test_named_intent(self):
        if False:
            while True:
                i = 10
        intent = AdaptIntent('CallEaglesIntent')
        self.assertEqual(intent.name, 'CallEaglesIntent')

    def test_unnamed_intent(self):
        if False:
            return 10
        intent = AdaptIntent()
        self.assertEqual(intent.name, '')