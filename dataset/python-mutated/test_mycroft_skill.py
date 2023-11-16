import sys
import unittest
from unittest.mock import MagicMock, patch
from adapt.intent import IntentBuilder
from os.path import join, dirname, abspath
from re import error
from datetime import datetime
import json
from mycroft.configuration import Configuration
from mycroft.messagebus.message import Message
from mycroft.skills.skill_data import load_regex_from_file, load_regex, load_vocabulary, read_vocab_file
from mycroft.skills import MycroftSkill, resting_screen_handler, intent_handler
from mycroft.skills.intent_service import open_intent_envelope
from test.util import base_config
BASE_CONF = base_config()

class MockEmitter(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.reset()

    def emit(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.types.append(message.msg_type)
        self.results.append(message.data)

    def get_types(self):
        if False:
            i = 10
            return i + 15
        return self.types

    def get_results(self):
        if False:
            print('Hello World!')
        return self.results

    def on(self, event, f):
        if False:
            return 10
        pass

    def reset(self):
        if False:
            print('Hello World!')
        self.types = []
        self.results = []

def vocab_base_path():
    if False:
        while True:
            i = 10
    return join(dirname(__file__), '..', 'vocab_test')

class TestFunction(unittest.TestCase):

    def test_resting_screen_handler(self):
        if False:
            i = 10
            return i + 15

        class T(MycroftSkill):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.name = 'TestObject'

            @resting_screen_handler('humbug')
            def f(self):
                if False:
                    print('Hello World!')
                pass
        test_class = T()
        self.assertTrue('resting_handler' in dir(test_class.f))
        self.assertEqual(test_class.f.resting_handler, 'humbug')

class TestMycroftSkill(unittest.TestCase):
    emitter = MockEmitter()
    regex_path = abspath(join(dirname(__file__), '../regex_test'))
    vocab_path = abspath(join(dirname(__file__), '../vocab_test'))

    def setUp(self):
        if False:
            while True:
                i = 10
        self.emitter.reset()
        self.local_settings_mock = self._mock_local_settings()

    def _mock_local_settings(self):
        if False:
            return 10
        local_settings_patch = patch('mycroft.skills.mycroft_skill.mycroft_skill.get_local_settings')
        self.addCleanup(local_settings_patch.stop)
        local_settings_mock = local_settings_patch.start()
        local_settings_mock.return_value = True
        return local_settings_mock

    def check_vocab(self, filename, results=None):
        if False:
            return 10
        results = results or {}
        intents = load_vocabulary(join(self.vocab_path, filename), 'A')
        self.compare_dicts(intents, results)

    def check_regex_from_file(self, filename, result_list=None):
        if False:
            for i in range(10):
                print('nop')
        result_list = result_list or []
        regex_file = join(self.regex_path, filename)
        self.assertEqual(sorted(load_regex_from_file(regex_file, 'A')), sorted(result_list))

    def compare_dicts(self, d1, d2):
        if False:
            i = 10
            return i + 15
        self.assertEqual(json.dumps(d1, sort_keys=True), json.dumps(d2, sort_keys=True))

    def check_read_vocab_file(self, path, result_list=None):
        if False:
            i = 10
            return i + 15
        resultlist = result_list or []
        self.assertEqual(sorted(read_vocab_file(path)), sorted(result_list))

    def check_regex(self, path, result_list=None):
        if False:
            for i in range(10):
                print('nop')
        result_list = result_list or []
        self.assertEqual(sorted(load_regex(path, 'A')), sorted(result_list))

    def check_emitter(self, result_list):
        if False:
            while True:
                i = 10
        for msg_type in self.emitter.get_types():
            self.assertEqual(msg_type, 'register_vocab')
        self.assertEqual(sorted(self.emitter.get_results(), key=lambda d: sorted(d.items())), sorted(result_list, key=lambda d: sorted(d.items())))
        self.emitter.reset()

    def test_load_regex_from_file_single(self):
        if False:
            print('Hello World!')
        self.check_regex_from_file('valid/single.rx', ['(?P<ASingleTest>.*)'])

    def test_load_regex_from_file_multiple(self):
        if False:
            return 10
        self.check_regex_from_file('valid/multiple.rx', ['(?P<AMultipleTest1>.*)', '(?P<AMultipleTest2>.*)'])

    def test_load_regex_from_file_none(self):
        if False:
            return 10
        self.check_regex_from_file('invalid/none.rx')

    def test_load_regex_from_file_invalid(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(error):
            self.check_regex_from_file('invalid/invalid.rx')

    def test_load_regex_from_file_does_not_exist(self):
        if False:
            print('Hello World!')
        with self.assertRaises(IOError):
            self.check_regex_from_file('does_not_exist.rx')

    def test_load_regex_full(self):
        if False:
            return 10
        self.check_regex(join(self.regex_path, 'valid'), ['(?P<AMultipleTest1>.*)', '(?P<AMultipleTest2>.*)', '(?P<ASingleTest>.*)'])

    def test_load_regex_empty(self):
        if False:
            while True:
                i = 10
        self.check_regex(join(dirname(__file__), 'empty_dir'))

    def test_load_regex_fail(self):
        if False:
            print('Hello World!')
        try:
            self.check_regex(join(dirname(__file__), 'regex_test_fail'))
        except OSError as e:
            self.assertEqual(e.strerror, 'No such file or directory')

    def test_load_vocab_file_single(self):
        if False:
            print('Hello World!')
        self.check_read_vocab_file(join(vocab_base_path(), 'valid/single.voc'), [['test']])

    def test_load_vocab_from_file_single_alias(self):
        if False:
            return 10
        self.check_read_vocab_file(join(vocab_base_path(), 'valid/singlealias.voc'), [['water', 'watering']])

    def test_load_vocab_from_file_multiple_alias(self):
        if False:
            i = 10
            return i + 15
        self.check_read_vocab_file(join(vocab_base_path(), 'valid/multiplealias.voc'), [['chair', 'chairs'], ['table', 'tables']])

    def test_load_vocab_from_file_does_not_exist(self):
        if False:
            return 10
        try:
            self.check_read_vocab_file('does_not_exist.voc')
        except IOError as e:
            self.assertEqual(e.strerror, 'No such file or directory')

    def test_load_vocab_full(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_vocab(join(self.vocab_path, 'valid'), {'Asingle': [['test']], 'Asinglealias': [['water', 'watering']], 'Amultiple': [['animal'], ['animals']], 'Amultiplealias': [['chair', 'chairs'], ['table', 'tables']]})

    def test_load_vocab_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_vocab(join(dirname(__file__), 'empty_dir'))

    def test_load_vocab_fail(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.check_regex(join(dirname(__file__), 'vocab_test_fail'))
        except OSError as e:
            self.assertEqual(e.strerror, 'No such file or directory')

    def test_open_envelope(self):
        if False:
            while True:
                i = 10
        name = 'Jerome'
        intent = IntentBuilder(name).require('Keyword')
        intent.name = name
        m = Message('register_intent', intent.__dict__)
        unpacked_intent = open_intent_envelope(m)
        self.assertEqual(intent.__dict__, unpacked_intent.__dict__)

    def check_detach_intent(self):
        if False:
            while True:
                i = 10
        self.assertTrue(len(self.emitter.get_types()) > 0)
        for msg_type in self.emitter.get_types():
            self.assertEqual(msg_type, 'detach_intent')
        self.emitter.reset()

    def check_register_intent(self, result_list):
        if False:
            for i in range(10):
                print('nop')
        for msg_type in self.emitter.get_types():
            self.assertEqual(msg_type, 'register_intent')
        self.assertEqual(sorted(self.emitter.get_results()), sorted(result_list))
        self.emitter.reset()

    def check_register_vocabulary(self, result_list):
        if False:
            print('Hello World!')
        for msg_type in self.emitter.get_types():
            self.assertEqual(msg_type, 'register_vocab')
        self.assertEqual(sorted(self.emitter.get_results()), sorted(result_list))
        self.emitter.reset()

    def test_register_intent(self):
        if False:
            i = 10
            return i + 15
        s = SimpleSkill1()
        s.bind(self.emitter)
        s.initialize()
        expected = [{'at_least_one': [], 'name': 'A:a', 'optional': [], 'requires': [('AKeyword', 'AKeyword')]}]
        self.check_register_intent(expected)
        s = SimpleSkill2()
        s.bind(self.emitter)
        s.initialize()
        expected = [{'at_least_one': [], 'name': 'A:a', 'optional': [], 'requires': [('AKeyword', 'AKeyword')]}]
        self.check_register_intent(expected)
        with self.assertRaises(ValueError):
            s = SimpleSkill3()
            s.bind(self.emitter)
            s.initialize()

    def test_enable_disable_intent(self):
        if False:
            i = 10
            return i + 15
        'Test disable/enable intent.'
        s = SimpleSkill1()
        s.bind(self.emitter)
        s.initialize()
        expected = [{'at_least_one': [], 'name': 'A:a', 'optional': [], 'requires': [('AKeyword', 'AKeyword')]}]
        self.check_register_intent(expected)
        s.disable_intent('a')
        self.check_detach_intent()
        s.enable_intent('a')
        self.check_register_intent(expected)

    def test_enable_disable_intent_handlers(self):
        if False:
            print('Hello World!')
        'Test disable/enable intent.'
        s = SimpleSkill1()
        s.bind(self.emitter)
        s.initialize()
        expected = [{'at_least_one': [], 'name': 'A:a', 'optional': [], 'requires': [('AKeyword', 'AKeyword')]}]
        self.check_register_intent(expected)
        msg = Message('test.msg', data={'intent_name': 'a'})
        s.handle_disable_intent(msg)
        self.check_detach_intent()
        s.handle_enable_intent(msg)
        self.check_register_intent(expected)

    def test_register_vocab(self):
        if False:
            while True:
                i = 10
        'Test disable/enable intent.'
        s = SimpleSkill1()
        s.bind(self.emitter)
        s.initialize()
        self.emitter.reset()
        expected = [{'start': 'hello', 'end': 'AHelloKeyword', 'entity_value': 'hello', 'entity_type': 'AHelloKeyword'}]
        s.register_vocabulary('hello', 'HelloKeyword')
        self.check_register_vocabulary(expected)
        s.register_regex('weird (?P<Weird>.+) stuff')
        expected = [{'regex': 'weird (?P<AWeird>.+) stuff'}]
        self.check_register_vocabulary(expected)

    def check_register_object_file(self, types_list, result_list):
        if False:
            return 10
        self.assertEqual(sorted(self.emitter.get_types()), sorted(types_list))
        self.assertEqual(sorted(self.emitter.get_results(), key=lambda d: sorted(d.items())), sorted(result_list, key=lambda d: sorted(d.items())))
        self.emitter.reset()

    def test_register_intent_file(self):
        if False:
            while True:
                i = 10
        self._test_intent_file(SimpleSkill4())

    def test_register_intent_intent_file(self):
        if False:
            i = 10
            return i + 15
        'Test register intent files using register_intent.'
        self._test_intent_file(SimpleSkill6())

    def _test_intent_file(self, s):
        if False:
            for i in range(10):
                print('nop')
        s.root_dir = abspath(join(dirname(__file__), 'intent_file'))
        s.bind(self.emitter)
        s.initialize()
        expected_types = ['padatious:register_intent', 'padatious:register_entity']
        expected_results = [{'file_name': join(dirname(__file__), 'intent_file', 'vocab', 'en-us', 'test.intent'), 'name': str(s.skill_id) + ':test.intent'}, {'file_name': join(dirname(__file__), 'intent_file', 'vocab', 'en-us', 'test_ent.entity'), 'name': str(s.skill_id) + ':test_ent'}]
        self.check_register_object_file(expected_types, expected_results)

    def check_register_decorators(self, result_list):
        if False:
            return 10
        self.assertEqual(sorted(self.emitter.get_results(), key=lambda d: sorted(d.items())), sorted(result_list, key=lambda d: sorted(d.items())))
        self.emitter.reset()

    def test_register_decorators(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test decorated intents '
        path_orig = sys.path
        sys.path.append(abspath(dirname(__file__)))
        SimpleSkill5 = __import__('decorator_test_skill').TestSkill
        s = SimpleSkill5()
        s.skill_id = 'A'
        s.bind(self.emitter)
        s.root_dir = abspath(join(dirname(__file__), 'intent_file'))
        s.initialize()
        s._register_decorated()
        expected = [{'at_least_one': [], 'name': 'A:a', 'optional': [], 'requires': [('AKeyword', 'AKeyword')]}, {'file_name': join(dirname(__file__), 'intent_file', 'vocab', 'en-us', 'test.intent'), 'name': str(s.skill_id) + ':test.intent'}]
        self.check_register_decorators(expected)
        sys.path = path_orig

    def test_failing_set_context(self):
        if False:
            for i in range(10):
                print('nop')
        s = SimpleSkill1()
        s.bind(self.emitter)
        with self.assertRaises(ValueError):
            s.set_context(1)
        with self.assertRaises(ValueError):
            s.set_context(1, 1)
        with self.assertRaises(ValueError):
            s.set_context('Kowabunga', 1)

    def test_set_context(self):
        if False:
            while True:
                i = 10

        def check_set_context(result_list):
            if False:
                while True:
                    i = 10
            for msg_type in self.emitter.get_types():
                self.assertEqual(msg_type, 'add_context')
            self.assertEqual(sorted(self.emitter.get_results()), sorted(result_list))
            self.emitter.reset()
        s = SimpleSkill1()
        s.bind(self.emitter)
        s.set_context('TurtlePower')
        expected = [{'context': 'ATurtlePower', 'origin': '', 'word': ''}]
        check_set_context(expected)
        s.set_context('Technodrome', 'Shredder')
        expected = [{'context': 'ATechnodrome', 'origin': '', 'word': 'Shredder'}]
        check_set_context(expected)
        s.set_context('Smörgåsbord€15')
        expected = [{'context': 'ASmörgåsbord€15', 'origin': '', 'word': ''}]
        check_set_context(expected)
        self.emitter.reset()

    def test_failing_remove_context(self):
        if False:
            return 10
        s = SimpleSkill1()
        s.bind(self.emitter)
        with self.assertRaises(ValueError):
            s.remove_context(1)

    def test_remove_context(self):
        if False:
            for i in range(10):
                print('nop')

        def check_remove_context(result_list):
            if False:
                print('Hello World!')
            for type in self.emitter.get_types():
                self.assertEqual(type, 'remove_context')
            self.assertEqual(sorted(self.emitter.get_results()), sorted(result_list))
            self.emitter.reset()
        s = SimpleSkill1()
        s.bind(self.emitter)
        s.remove_context('Donatello')
        expected = [{'context': 'ADonatello'}]
        check_remove_context(expected)

    @patch.dict(Configuration._Configuration__config, BASE_CONF)
    def test_skill_location(self):
        if False:
            print('Hello World!')
        s = SimpleSkill1()
        self.assertEqual(s.location, BASE_CONF.get('location'))
        self.assertEqual(s.location_pretty, BASE_CONF['location']['city']['name'])
        self.assertEqual(s.location_timezone, BASE_CONF['location']['timezone']['code'])

    @patch.dict(Configuration._Configuration__config, BASE_CONF)
    def test_add_event(self):
        if False:
            for i in range(10):
                print('nop')
        emitter = MagicMock()
        s = SimpleSkill1()
        s.bind(emitter)
        s.add_event('handler1', s.handler)
        self.assertEqual(emitter.on.call_args[0][0], 'handler1')
        self.assertTrue('handler1' in [e[0] for e in s.events])

    @patch.dict(Configuration._Configuration__config, BASE_CONF)
    def test_remove_event(self):
        if False:
            for i in range(10):
                print('nop')
        emitter = MagicMock()
        s = SimpleSkill1()
        s.bind(emitter)
        s.add_event('handler1', s.handler)
        self.assertTrue('handler1' in [e[0] for e in s.events])
        s.remove_event('handler1')
        self.assertTrue('handler1' not in [e[0] for e in s.events])
        self.assertEqual(emitter.remove_all_listeners.call_args[0][0], 'handler1')

    @patch.dict(Configuration._Configuration__config, BASE_CONF)
    def test_add_scheduled_event(self):
        if False:
            for i in range(10):
                print('nop')
        emitter = MagicMock()
        s = SimpleSkill1()
        s.bind(emitter)
        s.schedule_event(s.handler, datetime.now(), name='datetime_handler')
        self.assertEqual(emitter.once.call_args[0][0], 'A:datetime_handler')
        sched_events = [e[0] for e in s.event_scheduler.events]
        self.assertTrue('A:datetime_handler' in sched_events)
        s.schedule_event(s.handler, 1, name='int_handler')
        self.assertEqual(emitter.once.call_args[0][0], 'A:int_handler')
        sched_events = [e[0] for e in s.event_scheduler.events]
        self.assertTrue('A:int_handler' in sched_events)
        s.schedule_event(s.handler, 0.5, name='float_handler')
        self.assertEqual(emitter.once.call_args[0][0], 'A:float_handler')
        sched_events = [e[0] for e in s.event_scheduler.events]
        self.assertTrue('A:float_handler' in sched_events)

    @patch.dict(Configuration._Configuration__config, BASE_CONF)
    def test_remove_scheduled_event(self):
        if False:
            i = 10
            return i + 15
        emitter = MagicMock()
        s = SimpleSkill1()
        s.bind(emitter)
        s.schedule_event(s.handler, datetime.now(), name='sched_handler1')
        events = [e[0] for e in s.event_scheduler.events]
        print(events)
        self.assertTrue('A:sched_handler1' in events)
        s.cancel_scheduled_event('sched_handler1')
        self.assertEqual(emitter.remove_all_listeners.call_args[0][0], 'A:sched_handler1')
        events = [e[0] for e in s.event_scheduler.events]
        self.assertTrue('A:sched_handler1' not in events)

    @patch.dict(Configuration._Configuration__config, BASE_CONF)
    def test_run_scheduled_event(self):
        if False:
            i = 10
            return i + 15
        emitter = MagicMock()
        s = SimpleSkill1()
        with patch.object(s, '_settings', create=True, value=MagicMock()):
            s.bind(emitter)
            s.schedule_event(s.handler, datetime.now(), name='sched_handler1')
            emitter.once.call_args[0][1](Message('message'))
            self.assertTrue(s.handler_run)
            self.assertTrue('A:sched_handler1' not in [e[0] for e in s.events])

    def test_voc_match(self):
        if False:
            while True:
                i = 10
        s = SimpleSkill1()
        s.root_dir = abspath(dirname(__file__))
        self.assertTrue(s.voc_match('turn off the lights', 'turn_off_test'))
        self.assertTrue(s.voc_match('would you please turn off the lights', 'turn_off_test'))
        self.assertFalse(s.voc_match('return office', 'turn_off_test'))
        self.assertTrue(s.voc_match('switch off the lights', 'turn_off_test'))
        self.assertFalse(s.voc_match('', 'turn_off_test'))
        self.assertFalse(s.voc_match('switch', 'turn_off_test'))
        self.assertFalse(s.voc_match('My hovercraft is full of eels', 'turn_off_test'))
        self.assertTrue(s.voc_match('turn off the lights', 'turn_off2_test'))
        self.assertFalse(s.voc_match('return office', 'turn_off2_test'))
        self.assertTrue(s.voc_match('switch off the lights', 'turn_off2_test'))
        self.assertFalse(s.voc_match('', 'turn_off_test'))
        self.assertFalse(s.voc_match('switch', 'turn_off_test'))
        self.assertFalse(s.voc_match('My hovercraft is full of eels', 'turn_off_test'))

    def test_voc_match_exact(self):
        if False:
            print('Hello World!')
        s = SimpleSkill1()
        s.root_dir = abspath(dirname(__file__))
        self.assertTrue(s.voc_match('yes', 'yes', exact=True))
        self.assertFalse(s.voc_match('yes please', 'yes', exact=True))
        self.assertTrue(s.voc_match('switch off', 'turn_off_test', exact=True))
        self.assertFalse(s.voc_match('would you please turn off the lights', 'turn_off_test', exact=True))

    def test_translate_locations(self):
        if False:
            while True:
                i = 10
        'Assert that the a translatable list can be loaded from dialog and\n        locale.\n        '
        s = SimpleSkill1()
        s.root_dir = abspath(join(dirname(__file__), 'translate', 'in-dialog/'))
        lst = s.translate_list('good_things')
        self.assertTrue(isinstance(lst, list))
        vals = s.translate_namedvalues('named_things')
        self.assertTrue(isinstance(vals, dict))
        template = s.translate_template('test', data={'thing': 'test framework'})
        self.assertEqual(template, ["Oh look it's my favourite test framework"])
        s = SimpleSkill1()
        s.root_dir = abspath(join(dirname(__file__), 'translate', 'in-locale'))
        lst = s.translate_list('good_things')
        self.assertTrue(isinstance(lst, list))
        vals = s.translate_namedvalues('named_things')
        self.assertTrue(isinstance(vals, dict))
        template = s.translate_template('test', data={'thing': 'test framework'})
        self.assertEqual(template, ["Oh look it's my favourite test framework"])
        s = SimpleSkill1()
        s.config_core['lang'] = 'de-de'
        s.root_dir = abspath(join(dirname(__file__), 'translate', 'in-locale'))
        lst = s.translate_list('good_things')
        self.assertEqual(lst, ['sonne', 'mycroft', 'zahne'])
        vals = s.translate_namedvalues('named_things')
        self.assertEqual(vals['blau'], '2')
        template = s.translate_template('test', data={'thing': 'test framework'})
        self.assertEqual(template, ['Aber setzen sie sich herr test framework'])
        lst = s.translate_list('not_in_german')
        self.assertEqual(lst, ['not', 'in', 'German'])
        s.config_core['lang'] = 'en-us'

    def test_speak_dialog_render_not_initialized(self):
        if False:
            return 10
        "Test that non-initialized dialog_renderer won't raise an error."
        s = SimpleSkill1()
        s.bind(self.emitter)
        s.dialog_renderer = None
        s.speak_dialog(key='key')

class TestIntentCollisions(unittest.TestCase):

    def test_two_intents_with_same_name(self):
        if False:
            print('Hello World!')
        emitter = MockEmitter()
        skill = SameIntentNameSkill()
        skill.bind(emitter)
        with self.assertRaises(ValueError):
            skill.initialize()

    def test_two_anonymous_intent_decorators(self):
        if False:
            for i in range(10):
                print('nop')
        'Two anonymous intent handlers should be ok.'
        emitter = MockEmitter()
        skill = SameAnonymousIntentDecoratorsSkill()
        skill.bind(emitter)
        skill._register_decorated()
        self.assertEqual(len(skill.intent_service.registered_intents), 2)

class _TestSkill(MycroftSkill):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.skill_id = 'A'

class SimpleSkill1(_TestSkill):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(SimpleSkill1, self).__init__()
        self.handler_run = False
    ' Test skill for normal intent builder syntax '

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        i = IntentBuilder('a').require('Keyword').build()
        self.register_intent(i, self.handler)

    def handler(self, message):
        if False:
            while True:
                i = 10
        self.handler_run = True

    def stop(self):
        if False:
            return 10
        pass

class SimpleSkill2(_TestSkill):
    """ Test skill for intent builder without .build() """
    skill_id = 'A'

    def initialize(self):
        if False:
            return 10
        i = IntentBuilder('a').require('Keyword')
        self.register_intent(i, self.handler)

    def handler(self, message):
        if False:
            for i in range(10):
                print('nop')
        pass

    def stop(self):
        if False:
            i = 10
            return i + 15
        pass

class SimpleSkill3(_TestSkill):
    """ Test skill for invalid Intent for register_intent """
    skill_id = 'A'

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.register_intent('string', self.handler)

    def handler(self, message):
        if False:
            return 10
        pass

    def stop(self):
        if False:
            while True:
                i = 10
        pass

class SimpleSkill4(_TestSkill):
    """ Test skill for padatious intent """
    skill_id = 'A'

    def initialize(self):
        if False:
            while True:
                i = 10
        self.register_intent_file('test.intent', self.handler)
        self.register_entity_file('test_ent.entity')

    def handler(self, message):
        if False:
            while True:
                i = 10
        pass

    def stop(self):
        if False:
            while True:
                i = 10
        pass

class SimpleSkill6(_TestSkill):
    """ Test skill for padatious intent """
    skill_id = 'A'

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.register_intent('test.intent', self.handler)
        self.register_entity_file('test_ent.entity')

    def handler(self, message):
        if False:
            return 10
        pass

class SameIntentNameSkill(_TestSkill):
    """Test skill for duplicate intent namesr."""
    skill_id = 'A'

    def initialize(self):
        if False:
            i = 10
            return i + 15
        intent = IntentBuilder('TheName').require('Keyword')
        intent2 = IntentBuilder('TheName').require('Keyword')
        self.register_intent(intent, self.handler)
        self.register_intent(intent2, self.handler)

    def handler(self, message):
        if False:
            while True:
                i = 10
        pass

class SameAnonymousIntentDecoratorsSkill(_TestSkill):
    """Test skill for duplicate anonymous intent handlers."""
    skill_id = 'A'

    @intent_handler(IntentBuilder('').require('Keyword'))
    @intent_handler(IntentBuilder('').require('OtherKeyword'))
    def handler(self, message):
        if False:
            return 10
        pass