"""The module execute a test of one skill intent.

Using a mocked message bus this module is responsible for sending utterences
and testing that the intent is called.

The module runner can test:
    That the expected intent in the skill is activated
    That the expected parameters are extracted from the utterance
    That Mycroft contexts are set or removed
    That the skill speak the intended answer
    The content of any message exchanged between the skill and the mycroft core

To set up a test the test runner can
    Send an utterance, as the user would normally speak
    Set up and remove context
    Set up a custom timeout for the test runner, to allow for skills that runs
    for a very long time

"""
from queue import Queue, Empty
from copy import copy
import json
import time
import os
import re
import ast
from os.path import join, isdir, basename
from pyee import EventEmitter
from numbers import Number
from mycroft.messagebus.message import Message
from mycroft.skills.core import MycroftSkill, FallbackSkill
from mycroft.skills.skill_loader import SkillLoader
from mycroft.configuration import Configuration
from mycroft.util.log import LOG
from logging import StreamHandler
from io import StringIO
from contextlib import contextmanager
from .colors import color
from .rules import intent_type_check, play_query_check, question_check, expected_data_check, expected_dialog_check, changed_context_check
MainModule = '__init__'
DEFAULT_EVALUAITON_TIMEOUT = 30
Configuration.get()['test_env'] = True

class SkillTestError(Exception):
    pass

@contextmanager
def temporary_handler(log, handler):
    if False:
        i = 10
        return i + 15
    'Context manager to replace the default logger with a temporary logger.\n\n    Args:\n        log (LOG): mycroft LOG object\n        handler (logging.Handler): Handler object to use\n    '
    old_handler = log.handler
    log.handler = handler
    yield
    log.handler = old_handler

def create_skill_descriptor(skill_path):
    if False:
        for i in range(10):
            print('nop')
    return {'path': skill_path}

def get_skills(skills_folder):
    if False:
        while True:
            i = 10
    'Find skills in the skill folder or sub folders.\n\n        Recursive traversal into subfolders stop when a __init__.py file\n        is discovered\n\n        Args:\n            skills_folder:  Folder to start a search for skills __init__.py\n                            files\n\n        Returns:\n            list: the skills\n    '
    skills = []

    def _get_skill_descriptor(skills_folder):
        if False:
            i = 10
            return i + 15
        if not isdir(skills_folder):
            return
        if MainModule + '.py' in os.listdir(skills_folder):
            skills.append(create_skill_descriptor(skills_folder))
            return
        possible_skills = os.listdir(skills_folder)
        for i in possible_skills:
            _get_skill_descriptor(join(skills_folder, i))
    _get_skill_descriptor(skills_folder)
    skills = sorted(skills, key=lambda p: basename(p['path']))
    return skills

def load_skills(emitter, skills_root):
    if False:
        i = 10
        return i + 15
    'Load all skills and set up emitter\n\n        Args:\n            emitter: The emmitter to use\n            skills_root: Directory of the skills __init__.py\n\n        Returns:\n            tuple: (list of loaded skills, dict with logs for each skill)\n\n    '
    skill_list = []
    log = {}
    for skill in get_skills(skills_root):
        path = skill['path']
        skill_id = 'test-' + basename(path)
        from mycroft.util.log import LOG as skills_log
        buf = StringIO()
        with temporary_handler(skills_log, StreamHandler(buf)):
            skill_loader = SkillLoader(emitter, path)
            skill_loader.skill_id = skill_id
            skill_loader.load()
            skill_list.append(skill_loader.instance)
        if skill_loader.instance:
            skill_loader.instance.log = LOG.create_logger(skill_loader.instance.name)
        log[path] = buf.getvalue()
    return (skill_list, log)

def unload_skills(skills):
    if False:
        print('Hello World!')
    for s in skills:
        s.default_shutdown()

class InterceptEmitter(object):
    """
    This class intercepts and allows emitting events between the
    skill_tester and the skill being tested.
    When a test is running emitted communication is intercepted for analysis
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.emitter = EventEmitter()
        self.q = None

    def on(self, event, f):
        if False:
            for i in range(10):
                print('nop')
        print('Event: ', event)
        self.emitter.on(event, f)

    def emit(self, event, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        event_name = event.msg_type
        if self.q:
            self.q.put(event)
        self.emitter.emit(event_name, event, *args, **kwargs)

    def wait_for_response(self, event, reply_type=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Simple single thread implementation of wait_for_response.'
        message_type = reply_type or event.msg_type + '.response'
        response = None

        def response_handler(msg):
            if False:
                print('Hello World!')
            nonlocal response
            response = msg
        self.emitter.once(message_type, response_handler)
        self.emitter.emit(event.msg_type, event)
        return response

    def once(self, event, f):
        if False:
            for i in range(10):
                print('nop')
        self.emitter.once(event, f)

    def remove(self, event_name, func):
        if False:
            for i in range(10):
                print('nop')
        pass

    def remove_all_listeners(self, event_name):
        if False:
            return 10
        pass

class MockSkillsLoader(object):
    """Load a skill and set up emitter
    """

    def __init__(self, skills_root):
        if False:
            for i in range(10):
                print('nop')
        self.load_log = None
        self.skills_root = skills_root
        self.emitter = InterceptEmitter()
        from mycroft.skills.intent_service import IntentService
        self.ih = IntentService(self.emitter)
        self.skills = None
        self.emitter.on('mycroft.skills.fallback', FallbackSkill.make_intent_failure_handler(self.emitter))

        def make_response(message):
            if False:
                while True:
                    i = 10
            skill_id = message.data.get('skill_id', '')
            data = dict(result=False, skill_id=skill_id)
            self.emitter.emit(Message('skill.converse.response', data))
        self.emitter.on('skill.converse.request', make_response)

    def load_skills(self):
        if False:
            i = 10
            return i + 15
        (skills, self.load_log) = load_skills(self.emitter, self.skills_root)
        self.skills = [s for s in skills if s]
        self.ih.padatious_service.train(Message('', data=dict(single_thread=True)))
        return self.emitter.emitter

    def unload_skills(self):
        if False:
            print('Hello World!')
        unload_skills(self.skills)

def load_test_case_file(test_case_file):
    if False:
        i = 10
        return i + 15
    'Load a test case to run.'
    print('')
    print(color.HEADER + '=' * 20 + ' RUNNING TEST ' + '=' * 20 + color.RESET)
    print('Test file: ', test_case_file)
    with open(test_case_file, 'r') as f:
        test_case = json.load(f)
    print('Test:', json.dumps(test_case, indent=4, sort_keys=False))
    return test_case

class SkillTest(object):
    """
        This class is instantiated for each skill being tested. It holds the
        data needed for the test, and contains the methods doing the test

    """

    def __init__(self, skill, test_case_file, emitter, test_status=None):
        if False:
            for i in range(10):
                print('nop')
        self.skill = skill
        self.test_case_file = test_case_file
        self.emitter = emitter
        self.dict = dict
        self.output_file = None
        self.returned_intent = False
        self.test_status = test_status
        self.failure_msg = None
        self.end_of_skill = False

    def run(self, loader):
        if False:
            while True:
                i = 10
        ' Execute the test\n\n        Run a test for a skill. The skill, test_case_file and emitter is\n        already set up in the __init__ method.\n\n        This method does all the preparation and cleanup and calls\n        self.execute_test() to perform the actual test.\n\n        Args:\n            bool: Test results -- only True if all passed\n        '
        self.end_of_skill = False
        s = [s for s in loader.skills if s and s.root_dir == self.skill]
        if s:
            s = s[0]
        else:
            if self.skill in loader.load_log:
                print('\n {} Captured Logs from loading {}'.format('=' * 15, '=' * 15))
                print(loader.load_log.pop(self.skill))
            raise SkillTestError("Skill couldn't be loaded")
        orig_get_response = s.get_response
        original_settings = s.settings
        try:
            return self.execute_test(s)
        finally:
            s.get_response = orig_get_response
            s.settings = original_settings

    def send_play_query(self, s, test_case):
        if False:
            print('Hello World!')
        'Emit an event triggering the a check for playback possibilities.'
        play_query = test_case['play_query']
        print('PLAY QUERY', color.USER_UTT + play_query + color.RESET)
        self.emitter.emit('play:query', Message('play:query:', {'phrase': play_query}))

    def send_play_start(self, s, test_case):
        if False:
            while True:
                i = 10
        'Emit an event starting playback from the skill.'
        print('PLAY START')
        callback_data = test_case['play_start']
        callback_data['skill_id'] = s.skill_id
        self.emitter.emit('play:start', Message('play:start', callback_data))

    def send_question(self, test_case):
        if False:
            return 10
        'Emit a Question to the loaded skills.'
        print('QUESTION: {}'.format(test_case['question']))
        callback_data = {'phrase': test_case['question']}
        self.emitter.emit('question:query', Message('question:query', data=callback_data))

    def send_utterance(self, test_case):
        if False:
            while True:
                i = 10
        'Emit an utterance to the loaded skills.'
        utt = test_case['utterance']
        print('UTTERANCE:', color.USER_UTT + utt + color.RESET)
        self.emitter.emit('recognizer_loop:utterance', Message('recognizer_loop:utterance', {'utterances': [utt]}))

    def apply_test_settings(self, s, test_case):
        if False:
            for i in range(10):
                print('nop')
        'Replace the skills settings with settings from the test_case.'
        s.settings = copy(test_case['settings'])
        print(color.YELLOW, 'will run test with custom settings:', '\n{}'.format(s.settings), color.RESET)

    def setup_get_response(self, s, test_case):
        if False:
            for i in range(10):
                print('nop')
        'Setup interception of get_response calls.'

        def get_response(dialog='', data=None, announcement='', validator=None, on_fail=None, num_retries=-1):
            if False:
                i = 10
                return i + 15
            data = data or {}
            utt = announcement or s.dialog_renderer.render(dialog, data)
            print(color.MYCROFT + '>> ' + utt + color.RESET)
            s.speak(utt)
            response = test_case['responses'].pop(0)
            print('SENDING RESPONSE:', color.USER_UTT + response + color.RESET)
            return response
        s.get_response = get_response

    def remove_context(self, s, cxt):
        if False:
            return 10
        'remove an adapt context.'
        if isinstance(cxt, list):
            for x in cxt:
                MycroftSkill.remove_context(s, x)
        else:
            MycroftSkill.remove_context(s, cxt)

    def set_context(self, s, cxt):
        if False:
            print('Hello World!')
        'Set an adapt context.'
        for (key, value) in cxt.items():
            MycroftSkill.set_context(s, key, value)

    def send_test_input(self, s, test_case):
        if False:
            for i in range(10):
                print('nop')
        'Emit an utterance, just like the STT engine does. This sends the\n        provided text to the skill engine for intent matching and it then\n        invokes the skill.\n\n        It also handles some special cases for common play skills and common\n        query skills.\n        '
        if 'utterance' in test_case:
            self.send_utterance(test_case)
        elif 'play_query' in test_case:
            self.send_play_query(s, test_case)
        elif 'play_start' in test_case:
            self.send_play_start(s, test_case)
        elif 'question' in test_case:
            self.send_question(test_case)
        else:
            raise SkillTestError('No input provided in test case')

    def execute_test(self, s):
        if False:
            i = 10
            return i + 15
        ' Execute test case.\n\n        Args:\n            s (MycroftSkill): mycroft skill to test\n\n        Returns:\n            (bool) True if the test succeeded completely.\n        '
        test_case = load_test_case_file(self.test_case_file)
        if 'settings' in test_case:
            self.apply_test_settings(s, test_case)
        if 'responses' in test_case:
            self.setup_get_response(s, test_case)
        if self.test_status:
            self.test_status.append_intent(s)
            if 'intent_type' in test_case:
                self.test_status.set_tested(test_case['intent_type'])
        evaluation_rule = EvaluationRule(test_case, s)
        q = Queue()
        s.bus.q = q
        cxt = test_case.get('remove_context', None)
        if cxt:
            self.remove_context(s, cxt)
        cxt = test_case.get('set_context', None)
        if cxt:
            self.set_context(s, cxt)
        self.send_test_input(s, test_case)
        timeout = self.get_timeout(test_case)
        while not evaluation_rule.all_succeeded():
            if self.check_queue(q, evaluation_rule) or time.time() > timeout:
                break
        self.shutdown_emitter(s)
        return self.results(evaluation_rule)

    def get_timeout(self, test_case):
        if False:
            i = 10
            return i + 15
        'Find any timeout specified in test case.\n\n        If no timeout is specified return the default.\n        '
        if test_case.get('evaluation_timeout', None) and isinstance(test_case['evaluation_timeout'], int):
            return time.time() + int(test_case.get('evaluation_timeout'))
        else:
            return time.time() + DEFAULT_EVALUAITON_TIMEOUT

    def check_queue(self, q, evaluation_rule):
        if False:
            return 10
        'Check the queue for events.\n\n        If event indicating skill completion is found returns True, else False.\n        '
        try:
            event = q.get(timeout=1)
            if ':' in event.msg_type:
                event.data['__type__'] = event.msg_type.split(':')[1]
            else:
                event.data['__type__'] = event.msg_type
            evaluation_rule.evaluate(event.data)
            if event.msg_type == 'mycroft.skill.handler.complete':
                self.end_of_skill = True
        except Empty:
            pass
        if q.empty() and self.end_of_skill:
            return True
        else:
            return False

    def shutdown_emitter(self, s):
        if False:
            while True:
                i = 10
        'Shutdown the skill connection to the bus.'
        s.bus.q = None
        self.emitter.remove_all_listeners('speak')
        self.emitter.remove_all_listeners('mycroft.skill.handler.complete')

    def results(self, evaluation_rule):
        if False:
            for i in range(10):
                print('nop')
        'Display and report the results.'
        if not evaluation_rule.all_succeeded():
            self.failure_msg = str(evaluation_rule.get_failure())
            print(color.FAIL + 'Evaluation failed' + color.RESET)
            print(color.FAIL + 'Failure:', self.failure_msg + color.RESET)
            return False
        return True
HIDDEN_MESSAGES = ['skill.converse.request', 'skill.converse.response', 'gui.page.show', 'gui.value.set']

class EvaluationRule:
    """
        This class initially convert the test_case json file to internal rule
        format, which is stored throughout the testcase run. All Messages on
        the event bus can be evaluated against the rules (test_case)

        This approach makes it easier to add new tests, since Message and rule
        traversal is already set up for the internal rule format.
        The test writer can use the internal rule format directly in the
        test_case using the assert keyword, which allows for more
        powerfull/individual test cases than the standard dictionaly
    """

    def __init__(self, test_case, skill=None):
        if False:
            print('Hello World!')
        ' Convert test_case read from file to internal rule format\n\n        Args:\n            test_case:  The loaded test case\n            skill:      optional skill to test, used to fetch dialogs\n        '
        self.rule = []
        _x = ['and']
        if 'utterance' in test_case and 'intent_type' in test_case:
            intent_type = str(test_case['intent_type'])
            _x.append(intent_type_check(intent_type))
        if test_case.get('intent', None):
            for item in test_case['intent'].items():
                _x.append(['equal', str(item[0]), str(item[1])])
        if 'play_query_match' in test_case:
            match = test_case['play_query_match']
            phrase = match.get('phrase', test_case.get('play_query'))
            self.rule.append(play_query_check(skill, match, phrase))
        elif 'expected_answer' in test_case:
            question = test_case['question']
            expected_answer = test_case['expected_answer']
            self.rule.append(question_check(skill, question, expected_answer))
        if test_case.get('expected_data'):
            expected_items = test_case['expected_data'].items()
            self.rule.append(expected_data_check(expected_items))
        if _x != ['and']:
            self.rule.append(_x)
        if isinstance(test_case.get('expected_response', None), str):
            self.rule.append(['match', 'utterance', str(test_case['expected_response'])])
        elif isinstance(test_case.get('expected_response', None), list):
            texts = test_case['expected_response']
            rules = [['match', 'utterance', str(r)] for r in texts]
            self.rule.append(['or'] + rules)
        if test_case.get('expected_dialog', None):
            if not skill:
                print(color.FAIL + "Skill is missing, can't run expected_dialog test" + color.RESET)
            else:
                expected_dialog = test_case['expected_dialog']
                self.rule.append(['or'] + expected_dialog_check(expected_dialog, skill))
        if test_case.get('changed_context', None):
            ctx = test_case['changed_context']
            for c in changed_context_check(ctx):
                self.rule.append(c)
        if test_case.get('assert', None):
            for _x in ast.literal_eval(test_case['assert']):
                self.rule.append(_x)
        print('Rule created ', self.rule)

    def evaluate(self, msg):
        if False:
            return 10
        ' Main entry for evaluating a message against the rules.\n\n        The rules are prepared in the __init__\n        This method is usually called several times with different\n        messages using the same rule set. Each call contributing\n        to fulfilling all the rules\n\n        Args:\n            msg:  The message event to evaluate\n        '
        if msg.get('__type__', '') not in HIDDEN_MESSAGES:
            print('\nEvaluating message: ', msg)
        for r in self.rule:
            self._partial_evaluate(r, msg)

    def _get_field_value(self, rule, msg):
        if False:
            while True:
                i = 10
        if isinstance(rule, list):
            value = msg.get(rule[0], None)
            if len(rule) > 1 and value:
                for field in rule[1:]:
                    value = value.get(field, None)
                    if not value:
                        break
        else:
            value = msg.get(rule, None)
        return value

    def _partial_evaluate(self, rule, msg):
        if False:
            return 10
        ' Evaluate the message against a part of the rules\n\n        Recursive over rules\n\n        Args:\n            rule:  A rule or a part of the rules to be broken down further\n            msg:   The message event being evaluated\n\n        Returns:\n            Bool: True if a partial evaluation succeeded\n        '
        if 'succeeded' in rule:
            return True
        if rule[0] == 'equal':
            if self._get_field_value(rule[1], msg) != rule[2]:
                return False
        if rule[0] == 'lt':
            if not isinstance(self._get_field_value(rule[1], msg), Number):
                return False
            if self._get_field_value(rule[1], msg) >= rule[2]:
                return False
        if rule[0] == 'gt':
            if not isinstance(self._get_field_value(rule[1], msg), Number):
                return False
            if self._get_field_value(rule[1], msg) <= rule[2]:
                return False
        if rule[0] == 'notEqual':
            if self._get_field_value(rule[1], msg) == rule[2]:
                return False
        if rule[0] == 'endsWith':
            if not (self._get_field_value(rule[1], msg) and self._get_field_value(rule[1], msg).endswith(rule[2])):
                return False
        if rule[0] == 'exists':
            if not self._get_field_value(rule[1], msg):
                return False
        if rule[0] == 'match':
            if not (self._get_field_value(rule[1], msg) and re.match(rule[2], self._get_field_value(rule[1], msg))):
                return False
        if rule[0] == 'and':
            for i in rule[1:]:
                if not self._partial_evaluate(i, msg):
                    return False
        if rule[0] == 'or':
            for i in rule[1:]:
                if self._partial_evaluate(i, msg):
                    break
            else:
                return False
        rule.append('succeeded')
        return True

    def get_failure(self):
        if False:
            i = 10
            return i + 15
        ' Get the first rule which has not succeeded\n\n        Returns:\n            str: The failed rule\n        '
        for x in self.rule:
            if x[-1] != 'succeeded':
                return x
        return None

    def all_succeeded(self):
        if False:
            return 10
        ' Test if all rules succeeded\n\n        Returns:\n            bool: True if all rules succeeded\n        '
        return len([x for x in self.rule if x[-1] != 'succeeded']) == 0