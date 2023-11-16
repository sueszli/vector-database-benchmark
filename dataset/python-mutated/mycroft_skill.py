"""Common functionality relating to the implementation of mycroft skills."""
from copy import deepcopy
import sys
import re
import traceback
from itertools import chain
from os import walk
from os.path import join, abspath, dirname, basename, exists
from pathlib import Path
from threading import Event, Timer, Lock
from xdg import BaseDirectory
from adapt.intent import Intent, IntentBuilder
from mycroft import dialog
from mycroft.api import DeviceApi
from mycroft.audio import wait_while_speaking
from mycroft.enclosure.api import EnclosureAPI
from mycroft.enclosure.gui import SkillGUI
from mycroft.configuration import Configuration
from mycroft.dialog import load_dialogs
from mycroft.filesystem import FileSystemAccess
from mycroft.messagebus.message import Message, dig_for_message
from mycroft.metrics import report_metric
from mycroft.util import resolve_resource_file, play_audio_file, camel_case_split
from mycroft.util.log import LOG
from mycroft.util.format import pronounce_number, join_list
from mycroft.util.parse import match_one, extract_number
from .event_container import EventContainer, create_wrapper, get_handler_name
from ..event_scheduler import EventSchedulerInterface
from ..intent_service_interface import IntentServiceInterface
from ..settings import get_local_settings, save_settings
from ..skill_data import load_vocabulary, load_regex, to_alnum, munge_regex, munge_intent_parser, read_vocab_file, read_value_file, read_translated_file

def simple_trace(stack_trace):
    if False:
        for i in range(10):
            print('nop')
    'Generate a simplified traceback.\n\n    Args:\n        stack_trace: Stack trace to simplify\n\n    Returns: (str) Simplified stack trace.\n    '
    stack_trace = stack_trace[:-1]
    tb = 'Traceback:\n'
    for line in stack_trace:
        if line.strip():
            tb += line
    return tb

def get_non_properties(obj):
    if False:
        i = 10
        return i + 15
    'Get attibutes that are not properties from object.\n\n    Will return members of object class along with bases down to MycroftSkill.\n\n    Args:\n        obj: object to scan\n\n    Returns:\n        Set of attributes that are not a property.\n    '

    def check_class(cls):
        if False:
            while True:
                i = 10
        'Find all non-properties in a class.'
        d = cls.__dict__
        np = [k for k in d if not isinstance(d[k], property)]
        for b in [b for b in cls.__bases__ if b not in (object, MycroftSkill)]:
            np += check_class(b)
        return np
    return set(check_class(obj.__class__))

class MycroftSkill:
    """Base class for mycroft skills providing common behaviour and parameters
    to all Skill implementations.

    For information on how to get started with creating mycroft skills see
    https://mycroft.ai/documentation/skills/introduction-developing-skills/

    Args:
        name (str): skill name
        bus (MycroftWebsocketClient): Optional bus connection
        use_settings (bool): Set to false to not use skill settings at all
    """

    def __init__(self, name=None, bus=None, use_settings=True):
        if False:
            while True:
                i = 10
        self.name = name or self.__class__.__name__
        self.resting_name = None
        self.skill_id = ''
        self.settings_meta = None
        self.root_dir = dirname(abspath(sys.modules[self.__module__].__file__))
        self.gui = SkillGUI(self)
        self._bus = None
        self._enclosure = None
        self.bind(bus)
        self.config_core = Configuration.get()
        self.settings = None
        self.settings_write_path = None
        if use_settings:
            self._init_settings()
        self.settings_change_callback = None
        self.dialog_renderer = None
        self.file_system = FileSystemAccess(join('skills', self.name))
        self.log = LOG.create_logger(self.name)
        self.reload_skill = True
        self.events = EventContainer(bus)
        self.voc_match_cache = {}
        self.event_scheduler = EventSchedulerInterface(self.name)
        self.intent_service = IntentServiceInterface()
        self.intent_service_lock = Lock()
        self.public_api = {}

    def _init_settings(self):
        if False:
            while True:
                i = 10
        'Setup skill settings.'
        self.settings_write_path = Path(self.root_dir)
        if not self.settings_write_path.joinpath('settings.json').exists():
            self.settings_write_path = Path(BaseDirectory.save_config_path('mycroft', 'skills', basename(self.root_dir)))
        settings_read_path = Path(self.root_dir)
        if not settings_read_path.joinpath('settings.json').exists():
            for dir in BaseDirectory.load_config_paths('mycroft', 'skills', basename(self.root_dir)):
                path = Path(dir)
                if path.joinpath('settings.json').exists():
                    settings_read_path = path
                    break
        self.settings = get_local_settings(settings_read_path, self.name)
        self._initial_settings = deepcopy(self.settings)

    @property
    def enclosure(self):
        if False:
            return 10
        if self._enclosure:
            return self._enclosure
        else:
            LOG.error('Skill not fully initialized. Move code ' + 'from  __init__() to initialize() to correct this.')
            LOG.error(simple_trace(traceback.format_stack()))
            raise Exception('Accessed MycroftSkill.enclosure in __init__')

    @property
    def bus(self):
        if False:
            return 10
        if self._bus:
            return self._bus
        else:
            LOG.error('Skill not fully initialized. Move code ' + 'from __init__() to initialize() to correct this.')
            LOG.error(simple_trace(traceback.format_stack()))
            raise Exception('Accessed MycroftSkill.bus in __init__')

    @property
    def location(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the JSON data struction holding location information.'
        return self.config_core.get('location')

    @property
    def location_pretty(self):
        if False:
            return 10
        "Get a more 'human' version of the location as a string."
        loc = self.location
        if type(loc) is dict and loc['city']:
            return loc['city']['name']
        return None

    @property
    def location_timezone(self):
        if False:
            for i in range(10):
                print('nop')
        "Get the timezone code, such as 'America/Los_Angeles'"
        loc = self.location
        if type(loc) is dict and loc['timezone']:
            return loc['timezone']['code']
        return None

    @property
    def lang(self):
        if False:
            return 10
        'Get the configured language.'
        return self.config_core.get('lang')

    def bind(self, bus):
        if False:
            for i in range(10):
                print('nop')
        'Register messagebus emitter with skill.\n\n        Args:\n            bus: Mycroft messagebus connection\n        '
        if bus:
            self._bus = bus
            self.events.set_bus(bus)
            self.intent_service.set_bus(bus)
            self.event_scheduler.set_bus(bus)
            self.event_scheduler.set_id(self.skill_id)
            self._enclosure = EnclosureAPI(bus, self.name)
            self._register_system_event_handlers()
            self.gui.setup_default_handlers()
            self._register_public_api()

    def _register_public_api(self):
        if False:
            for i in range(10):
                print('nop')
        ' Find and register api methods.\n        Api methods has been tagged with the api_method member, for each\n        method where this is found the method a message bus handler is\n        registered.\n        Finally create a handler for fetching the api info from any requesting\n        skill.\n        '

        def wrap_method(func):
            if False:
                for i in range(10):
                    print('nop')
            'Boiler plate for returning the response to the sender.'

            def wrapper(message):
                if False:
                    while True:
                        i = 10
                result = func(*message.data['args'], **message.data['kwargs'])
                self.bus.emit(message.response(data={'result': result}))
            return wrapper
        methods = [attr_name for attr_name in get_non_properties(self) if hasattr(getattr(self, attr_name), '__name__')]
        for attr_name in methods:
            method = getattr(self, attr_name)
            if hasattr(method, 'api_method'):
                doc = method.__doc__ or ''
                name = method.__name__
                self.public_api[name] = {'help': doc, 'type': '{}.{}'.format(self.skill_id, name), 'func': method}
        for key in self.public_api:
            if 'type' in self.public_api[key] and 'func' in self.public_api[key]:
                LOG.debug('Adding api method: {}'.format(self.public_api[key]['type']))
                func = self.public_api[key].pop('func')
                self.add_event(self.public_api[key]['type'], wrap_method(func))
        if self.public_api:
            self.add_event('{}.public_api'.format(self.skill_id), self._send_public_api)

    def _register_system_event_handlers(self):
        if False:
            for i in range(10):
                print('nop')
        'Add all events allowing the standard interaction with the Mycroft\n        system.\n        '

        def stop_is_implemented():
            if False:
                return 10
            return self.__class__.stop is not MycroftSkill.stop
        if stop_is_implemented():
            self.add_event('mycroft.stop', self.__handle_stop)
        self.add_event('mycroft.skill.enable_intent', self.handle_enable_intent)
        self.add_event('mycroft.skill.disable_intent', self.handle_disable_intent)
        self.add_event('mycroft.skill.set_cross_context', self.handle_set_cross_context)
        self.add_event('mycroft.skill.remove_cross_context', self.handle_remove_cross_context)
        self.events.add('mycroft.skills.settings.changed', self.handle_settings_change)

    def handle_settings_change(self, message):
        if False:
            i = 10
            return i + 15
        'Update settings if the remote settings changes apply to this skill.\n\n        The skill settings downloader uses a single API call to retrieve the\n        settings for all skills.  This is done to limit the number API calls.\n        A "mycroft.skills.settings.changed" event is emitted for each skill\n        that had their settings changed.  Only update this skill\'s settings\n        if its remote settings were among those changed\n        '
        if self.settings_meta is None or self.settings_meta.skill_gid is None:
            LOG.error('The skill_gid was not set when {} was loaded!'.format(self.name))
        else:
            remote_settings = message.data.get(self.settings_meta.skill_gid)
            if remote_settings is not None:
                LOG.info('Updating settings for skill ' + self.name)
                self.settings.update(**remote_settings)
                save_settings(self.settings_write_path, self.settings)
                if self.settings_change_callback is not None:
                    self.settings_change_callback()

    def detach(self):
        if False:
            return 10
        with self.intent_service_lock:
            for (name, _) in self.intent_service:
                name = '{}:{}'.format(self.skill_id, name)
                self.intent_service.detach_intent(name)

    def initialize(self):
        if False:
            while True:
                i = 10
        'Perform any final setup needed for the skill.\n\n        Invoked after the skill is fully constructed and registered with the\n        system.\n        '
        pass

    def _send_public_api(self, message):
        if False:
            return 10
        "Respond with the skill's public api."
        self.bus.emit(message.response(data=self.public_api))

    def get_intro_message(self):
        if False:
            while True:
                i = 10
        'Get a message to speak on first load of the skill.\n\n        Useful for post-install setup instructions.\n\n        Returns:\n            str: message that will be spoken to the user\n        '
        return None

    def converse(self, message=None):
        if False:
            print('Hello World!')
        'Handle conversation.\n\n        This method gets a peek at utterances before the normal intent\n        handling process after a skill has been invoked once.\n\n        To use, override the converse() method and return True to\n        indicate that the utterance has been handled.\n\n        utterances and lang are depreciated\n\n        Args:\n            message:    a message object containing a message type with an\n                        optional JSON data packet\n\n        Returns:\n            bool: True if an utterance was handled, otherwise False\n        '
        return False

    def __get_response(self):
        if False:
            print('Hello World!')
        "Helper to get a response from the user\n\n        NOTE:  There is a race condition here.  There is a small amount of\n        time between the end of the device speaking and the converse method\n        being overridden in this method.  If an utterance is injected during\n        this time, the wrong converse method is executed.  The condition is\n        hidden during normal use due to the amount of time it takes a user\n        to speak a response. The condition is revealed when an automated\n        process injects an utterance quicker than this method can flip the\n        converse methods.\n\n        Returns:\n            str: user's response or None on a timeout\n        "
        event = Event()

        def converse(utterances, lang=None):
            if False:
                print('Hello World!')
            converse.response = utterances[0] if utterances else None
            event.set()
            return True
        self.make_active()
        converse.response = None
        default_converse = self.converse
        self.converse = converse
        event.wait(15)
        self.converse = default_converse
        return converse.response

    def get_response(self, dialog='', data=None, validator=None, on_fail=None, num_retries=-1):
        if False:
            for i in range(10):
                print('nop')
        'Get response from user.\n\n        If a dialog is supplied it is spoken, followed immediately by listening\n        for a user response. If the dialog is omitted listening is started\n        directly.\n\n        The response can optionally be validated before returning.\n\n        Example::\n\n            color = self.get_response(\'ask.favorite.color\')\n\n        Args:\n            dialog (str): Optional dialog to speak to the user\n            data (dict): Data used to render the dialog\n            validator (any): Function with following signature::\n\n                def validator(utterance):\n                    return utterance != "red"\n\n            on_fail (any):\n                Dialog or function returning literal string to speak on\n                invalid input. For example::\n\n                    def on_fail(utterance):\n                        return "nobody likes the color red, pick another"\n\n            num_retries (int): Times to ask user for input, -1 for infinite\n                NOTE: User can not respond and timeout or say "cancel" to stop\n\n        Returns:\n            str: User\'s reply or None if timed out or canceled\n        '
        data = data or {}

        def on_fail_default(utterance):
            if False:
                while True:
                    i = 10
            fail_data = data.copy()
            fail_data['utterance'] = utterance
            if on_fail:
                return self.dialog_renderer.render(on_fail, fail_data)
            else:
                return self.dialog_renderer.render(dialog, data)

        def is_cancel(utterance):
            if False:
                print('Hello World!')
            return self.voc_match(utterance, 'cancel')

        def validator_default(utterance):
            if False:
                for i in range(10):
                    print('nop')
            return not is_cancel(utterance)
        on_fail_fn = on_fail if callable(on_fail) else on_fail_default
        validator = validator or validator_default
        dialog_exists = self.dialog_renderer.render(dialog, data)
        if dialog_exists:
            self.speak_dialog(dialog, data, expect_response=True, wait=True)
        else:
            self.bus.emit(Message('mycroft.mic.listen'))
        return self._wait_response(is_cancel, validator, on_fail_fn, num_retries)

    def _wait_response(self, is_cancel, validator, on_fail, num_retries):
        if False:
            while True:
                i = 10
        'Loop until a valid response is received from the user or the retry\n        limit is reached.\n\n        Args:\n            is_cancel (callable): function checking cancel criteria\n            validator (callbale): function checking for a valid response\n            on_fail (callable): function handling retries\n\n        '
        num_fails = 0
        while True:
            response = self.__get_response()
            if response is None:
                num_none_fails = 1 if num_retries < 0 else num_retries
                if num_fails >= num_none_fails:
                    return None
            else:
                if validator(response):
                    return response
                if is_cancel(response):
                    return None
            num_fails += 1
            if 0 < num_retries < num_fails:
                return None
            line = on_fail(response)
            if line:
                self.speak(line, expect_response=True)
            else:
                self.bus.emit(Message('mycroft.mic.listen'))

    def ask_yesno(self, prompt, data=None):
        if False:
            for i in range(10):
                print('nop')
        "Read prompt and wait for a yes/no answer\n\n        This automatically deals with translation and common variants,\n        such as 'yeah', 'sure', etc.\n\n        Args:\n              prompt (str): a dialog id or string to read\n              data (dict): response data\n        Returns:\n              string:  'yes', 'no' or whatever the user response if not\n                       one of those, including None\n        "
        resp = self.get_response(dialog=prompt, data=data)
        if self.voc_match(resp, 'yes'):
            return 'yes'
        elif self.voc_match(resp, 'no'):
            return 'no'
        else:
            return resp

    def ask_selection(self, options, dialog='', data=None, min_conf=0.65, numeric=False):
        if False:
            i = 10
            return i + 15
        'Read options, ask dialog question and wait for an answer.\n\n        This automatically deals with fuzzy matching and selection by number\n        e.g.\n\n        * "first option"\n        * "last option"\n        * "second option"\n        * "option number four"\n\n        Args:\n              options (list): list of options to present user\n              dialog (str): a dialog id or string to read AFTER all options\n              data (dict): Data used to render the dialog\n              min_conf (float): minimum confidence for fuzzy match, if not\n                                reached return None\n              numeric (bool): speak options as a numeric menu\n        Returns:\n              string: list element selected by user, or None\n        '
        assert isinstance(options, list)
        if not len(options):
            return None
        elif len(options) == 1:
            return options[0]
        if numeric:
            for (idx, opt) in enumerate(options):
                opt_str = '{number}, {option_text}'.format(number=pronounce_number(idx + 1, self.lang), option_text=opt)
                self.speak(opt_str, wait=True)
        else:
            opt_str = join_list(options, 'or', lang=self.lang) + '?'
            self.speak(opt_str, wait=True)
        resp = self.get_response(dialog=dialog, data=data)
        if resp:
            (match, score) = match_one(resp, options)
            if score < min_conf:
                if self.voc_match(resp, 'last'):
                    resp = options[-1]
                else:
                    num = extract_number(resp, ordinals=True, lang=self.lang)
                    resp = None
                    if num and num <= len(options):
                        resp = options[num - 1]
            else:
                resp = match
        return resp

    def voc_match(self, utt, voc_filename, lang=None, exact=False):
        if False:
            return 10
        'Determine if the given utterance contains the vocabulary provided.\n\n        By default the method checks if the utterance contains the given vocab\n        thereby allowing the user to say things like "yes, please" and still\n        match against "Yes.voc" containing only "yes". An exact match can be\n        requested.\n\n        The method first checks in the current Skill\'s .voc files and secondly\n        in the "res/text" folder of mycroft-core. The result is cached to\n        avoid hitting the disk each time the method is called.\n\n        Args:\n            utt (str): Utterance to be tested\n            voc_filename (str): Name of vocabulary file (e.g. \'yes\' for\n                                \'res/text/en-us/yes.voc\')\n            lang (str): Language code, defaults to self.long\n            exact (bool): Whether the vocab must exactly match the utterance\n\n        Returns:\n            bool: True if the utterance has the given vocabulary it\n        '
        lang = lang or self.lang
        cache_key = lang + voc_filename
        if cache_key not in self.voc_match_cache:
            voc = self.find_resource(voc_filename + '.voc', 'vocab')
            if not voc:
                voc = resolve_resource_file(join('text', lang, voc_filename + '.voc'))
            if not voc or not exists(voc):
                raise FileNotFoundError('Could not find {}.voc file'.format(voc_filename))
            vocab = read_vocab_file(voc)
            self.voc_match_cache[cache_key] = list(chain(*vocab))
        if utt:
            if exact:
                return any((i.strip() == utt for i in self.voc_match_cache[cache_key]))
            else:
                return any([re.match('.*\\b' + i + '\\b.*', utt) for i in self.voc_match_cache[cache_key]])
        else:
            return False

    def report_metric(self, name, data):
        if False:
            i = 10
            return i + 15
        'Report a skill metric to the Mycroft servers.\n\n        Args:\n            name (str): Name of metric. Must use only letters and hyphens\n            data (dict): JSON dictionary to report. Must be valid JSON\n        '
        report_metric('{}:{}'.format(basename(self.root_dir), name), data)

    def send_email(self, title, body):
        if False:
            while True:
                i = 10
        "Send an email to the registered user's email.\n\n        Args:\n            title (str): Title of email\n            body  (str): HTML body of email. This supports\n                         simple HTML like bold and italics\n        "
        DeviceApi().send_email(title, body, basename(self.root_dir))

    def make_active(self):
        if False:
            while True:
                i = 10
        'Bump skill to active_skill list in intent_service.\n\n        This enables converse method to be called even without skill being\n        used in last 5 minutes.\n        '
        self.bus.emit(Message('active_skill_request', {'skill_id': self.skill_id}))

    def _handle_collect_resting(self, _=None):
        if False:
            return 10
        'Handler for collect resting screen messages.\n\n        Sends info on how to trigger this skills resting page.\n        '
        self.log.info('Registering resting screen')
        message = Message('mycroft.mark2.register_idle', data={'name': self.resting_name, 'id': self.skill_id})
        self.bus.emit(message)

    def register_resting_screen(self):
        if False:
            while True:
                i = 10
        'Registers resting screen from the resting_screen_handler decorator.\n\n        This only allows one screen and if two is registered only one\n        will be used.\n        '
        for attr_name in get_non_properties(self):
            method = getattr(self, attr_name)
            if hasattr(method, 'resting_handler'):
                self.resting_name = method.resting_handler
                self.log.info('Registering resting screen {} for {}.'.format(method, self.resting_name))
                msg_type = '{}.{}'.format(self.skill_id, 'idle')
                self.add_event(msg_type, method)
                self.add_event('mycroft.mark2.collect_idle', self._handle_collect_resting)
                self._handle_collect_resting()
                break

    def _register_decorated(self):
        if False:
            print('Hello World!')
        "Register all intent handlers that are decorated with an intent.\n\n        Looks for all functions that have been marked by a decorator\n        and read the intent data from them.  The intent handlers aren't the\n        only decorators used.  Skip properties as calling getattr on them\n        executes the code which may have unintended side-effects\n        "
        for attr_name in get_non_properties(self):
            method = getattr(self, attr_name)
            if hasattr(method, 'intents'):
                for intent in getattr(method, 'intents'):
                    self.register_intent(intent, method)
            if hasattr(method, 'intent_files'):
                for intent_file in getattr(method, 'intent_files'):
                    self.register_intent_file(intent_file, method)

    def translate(self, text, data=None):
        if False:
            print('Hello World!')
        "Load a translatable single string resource\n\n        The string is loaded from a file in the skill's dialog subdirectory\n        'dialog/<lang>/<text>.dialog'\n\n        The string is randomly chosen from the file and rendered, replacing\n        mustache placeholders with values found in the data dictionary.\n\n        Args:\n            text (str): The base filename  (no extension needed)\n            data (dict, optional): a JSON dictionary\n\n        Returns:\n            str: A randomly chosen string from the file\n        "
        return self.dialog_renderer.render(text, data or {})

    def find_resource(self, res_name, res_dirname=None):
        if False:
            for i in range(10):
                print('nop')
        "Find a resource file.\n\n        Searches for the given filename using this scheme:\n\n        1. Search the resource lang directory:\n\n           <skill>/<res_dirname>/<lang>/<res_name>\n\n        2. Search the resource directory:\n\n           <skill>/<res_dirname>/<res_name>\n\n        3. Search the locale lang directory or other subdirectory:\n\n           <skill>/locale/<lang>/<res_name> or\n\n           <skill>/locale/<lang>/.../<res_name>\n\n        Args:\n            res_name (string): The resource name to be found\n            res_dirname (string, optional): A skill resource directory, such\n                                            'dialog', 'vocab', 'regex' or 'ui'.\n                                            Defaults to None.\n\n        Returns:\n            string: The full path to the resource file or None if not found\n        "
        result = self._find_resource(res_name, self.lang, res_dirname)
        if not result and self.lang != 'en-us':
            LOG.warning("Resource '{}' for lang '{}' not found: trying 'en-us'".format(res_name, self.lang))
            result = self._find_resource(res_name, 'en-us', res_dirname)
        return result

    def _find_resource(self, res_name, lang, res_dirname=None):
        if False:
            print('Hello World!')
        'Finds a resource by name, lang and dir\n        '
        if res_dirname:
            path = join(self.root_dir, res_dirname, lang, res_name)
            if exists(path):
                return path
            path = join(self.root_dir, res_dirname, res_name)
            if exists(path):
                return path
        root_path = join(self.root_dir, 'locale', lang)
        for (path, _, files) in walk(root_path):
            if res_name in files:
                return join(path, res_name)
        return None

    def translate_namedvalues(self, name, delim=','):
        if False:
            return 10
        "Load translation dict containing names and values.\n\n        This loads a simple CSV from the 'dialog' folders.\n        The name is the first list item, the value is the\n        second.  Lines prefixed with # or // get ignored\n\n        Args:\n            name (str): name of the .value file, no extension needed\n            delim (char): delimiter character used, default is ','\n\n        Returns:\n            dict: name and value dictionary, or empty dict if load fails\n        "
        if not name.endswith('.value'):
            name += '.value'
        try:
            filename = self.find_resource(name, 'dialog')
            return read_value_file(filename, delim)
        except Exception:
            return {}

    def translate_template(self, template_name, data=None):
        if False:
            i = 10
            return i + 15
        "Load a translatable template.\n\n        The strings are loaded from a template file in the skill's dialog\n        subdirectory.\n        'dialog/<lang>/<template_name>.template'\n\n        The strings are loaded and rendered, replacing mustache placeholders\n        with values found in the data dictionary.\n\n        Args:\n            template_name (str): The base filename (no extension needed)\n            data (dict, optional): a JSON dictionary\n\n        Returns:\n            list of str: The loaded template file\n        "
        return self.__translate_file(template_name + '.template', data)

    def translate_list(self, list_name, data=None):
        if False:
            return 10
        "Load a list of translatable string resources\n\n        The strings are loaded from a list file in the skill's dialog\n        subdirectory.\n        'dialog/<lang>/<list_name>.list'\n\n        The strings are loaded and rendered, replacing mustache placeholders\n        with values found in the data dictionary.\n\n        Args:\n            list_name (str): The base filename (no extension needed)\n            data (dict, optional): a JSON dictionary\n\n        Returns:\n            list of str: The loaded list of strings with items in consistent\n                         positions regardless of the language.\n        "
        return self.__translate_file(list_name + '.list', data)

    def __translate_file(self, name, data):
        if False:
            i = 10
            return i + 15
        'Load and render lines from dialog/<lang>/<name>'
        filename = self.find_resource(name, 'dialog')
        return read_translated_file(filename, data)

    def add_event(self, name, handler, handler_info=None, once=False):
        if False:
            print('Hello World!')
        'Create event handler for executing intent or other event.\n\n        Args:\n            name (string): IntentParser name\n            handler (func): Method to call\n            handler_info (string): Base message when reporting skill event\n                                   handler status on messagebus.\n            once (bool, optional): Event handler will be removed after it has\n                                   been run once.\n        '
        skill_data = {'name': get_handler_name(handler)}

        def on_error(e):
            if False:
                i = 10
                return i + 15
            'Speak and log the error.'
            handler_name = camel_case_split(self.name)
            msg_data = {'skill': handler_name}
            msg = dialog.get('skill.error', self.lang, msg_data)
            self.speak(msg)
            LOG.exception(msg)
            skill_data['exception'] = repr(e)

        def on_start(message):
            if False:
                return 10
            'Indicate that the skill handler is starting.'
            if handler_info:
                msg_type = handler_info + '.start'
                self.bus.emit(message.forward(msg_type, skill_data))

        def on_end(message):
            if False:
                return 10
            'Store settings and indicate that the skill handler has completed\n            '
            if self.settings != self._initial_settings:
                save_settings(self.settings_write_path, self.settings)
                self._initial_settings = deepcopy(self.settings)
            if handler_info:
                msg_type = handler_info + '.complete'
                self.bus.emit(message.forward(msg_type, skill_data))
        wrapper = create_wrapper(handler, self.skill_id, on_start, on_end, on_error)
        return self.events.add(name, wrapper, once)

    def remove_event(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Removes an event from bus emitter and events list.\n\n        Args:\n            name (string): Name of Intent or Scheduler Event\n        Returns:\n            bool: True if found and removed, False if not found\n        '
        return self.events.remove(name)

    def _register_adapt_intent(self, intent_parser, handler):
        if False:
            for i in range(10):
                print('nop')
        'Register an adapt intent.\n\n        Will handle registration of anonymous\n        Args:\n            intent_parser: Intent object to parse utterance for the handler.\n            handler (func): function to register with intent\n        '
        is_anonymous = not intent_parser.name
        name = intent_parser.name or handler.__name__
        if is_anonymous:
            original_name = name
            nbr = 0
            while name in self.intent_service:
                nbr += 1
                name = f'{original_name}{nbr}'
        elif name in self.intent_service:
            raise ValueError(f'The intent name {name} is already taken')
        munge_intent_parser(intent_parser, name, self.skill_id)
        self.intent_service.register_adapt_intent(name, intent_parser)
        if handler:
            self.add_event(intent_parser.name, handler, 'mycroft.skill.handler')

    def register_intent(self, intent_parser, handler):
        if False:
            for i in range(10):
                print('nop')
        'Register an Intent with the intent service.\n\n        Args:\n            intent_parser: Intent, IntentBuilder object or padatious intent\n                           file to parse utterance for the handler.\n            handler (func): function to register with intent\n        '
        with self.intent_service_lock:
            self._register_intent(intent_parser, handler)

    def _register_intent(self, intent_parser, handler):
        if False:
            return 10
        'Register an Intent with the intent service.\n\n        Args:\n            intent_parser: Intent, IntentBuilder object or padatious intent\n                           file to parse utterance for the handler.\n            handler (func): function to register with intent\n        '
        if isinstance(intent_parser, IntentBuilder):
            intent_parser = intent_parser.build()
        if isinstance(intent_parser, str) and intent_parser.endswith('.intent'):
            return self.register_intent_file(intent_parser, handler)
        elif not isinstance(intent_parser, Intent):
            raise ValueError('"' + str(intent_parser) + '" is not an Intent')
        return self._register_adapt_intent(intent_parser, handler)

    def register_intent_file(self, intent_file, handler):
        if False:
            return 10
        "Register an Intent file with the intent service.\n\n        For example:\n\n        === food.order.intent ===\n        Order some {food}.\n        Order some {food} from {place}.\n        I'm hungry.\n        Grab some {food} from {place}.\n\n        Optionally, you can also use <register_entity_file>\n        to specify some examples of {food} and {place}\n\n        In addition, instead of writing out multiple variations\n        of the same sentence you can write:\n\n        === food.order.intent ===\n        (Order | Grab) some {food} (from {place} | ).\n        I'm hungry.\n\n        Args:\n            intent_file: name of file that contains example queries\n                         that should activate the intent.  Must end with\n                         '.intent'\n            handler:     function to register with intent\n        "
        name = '{}:{}'.format(self.skill_id, intent_file)
        filename = self.find_resource(intent_file, 'vocab')
        if not filename:
            raise FileNotFoundError('Unable to find "{}"'.format(intent_file))
        self.intent_service.register_padatious_intent(name, filename)
        if handler:
            self.add_event(name, handler, 'mycroft.skill.handler')

    def register_entity_file(self, entity_file):
        if False:
            i = 10
            return i + 15
        "Register an Entity file with the intent service.\n\n        An Entity file lists the exact values that an entity can hold.\n        For example:\n\n        === ask.day.intent ===\n        Is it {weekend}?\n\n        === weekend.entity ===\n        Saturday\n        Sunday\n\n        Args:\n            entity_file (string): name of file that contains examples of an\n                                  entity.  Must end with '.entity'\n        "
        if entity_file.endswith('.entity'):
            entity_file = entity_file.replace('.entity', '')
        filename = self.find_resource(entity_file + '.entity', 'vocab')
        if not filename:
            raise FileNotFoundError('Unable to find "{}"'.format(entity_file))
        name = '{}:{}'.format(self.skill_id, entity_file)
        with self.intent_service_lock:
            self.intent_service.register_padatious_entity(name, filename)

    def handle_enable_intent(self, message):
        if False:
            while True:
                i = 10
        'Listener to enable a registered intent if it belongs to this skill.\n        '
        intent_name = message.data['intent_name']
        return self.enable_intent(intent_name)

    def handle_disable_intent(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Listener to disable a registered intent if it belongs to this skill.\n        '
        intent_name = message.data['intent_name']
        self.disable_intent(intent_name)

    def disable_intent(self, intent_name):
        if False:
            print('Hello World!')
        "Disable a registered intent if it belongs to this skill.\n\n        Args:\n            intent_name (string): name of the intent to be disabled\n\n        Returns:\n                bool: True if disabled, False if it wasn't registered\n        "
        with self.intent_service_lock:
            if intent_name in self.intent_service:
                LOG.info('Disabling intent ' + intent_name)
                name = '{}:{}'.format(self.skill_id, intent_name)
                self.intent_service.detach_intent(name)
                return True
            else:
                LOG.error(f"Could not disable {intent_name}, it hasn't been registered.")
                return False

    def enable_intent(self, intent_name):
        if False:
            i = 10
            return i + 15
        "(Re)Enable a registered intent if it belongs to this skill.\n\n        Args:\n            intent_name: name of the intent to be enabled\n\n        Returns:\n            bool: True if enabled, False if it wasn't registered\n        "
        intent = self.intent_service.get_intent(intent_name)
        with self.intent_service_lock:
            if intent and self.intent_service.intent_is_detached(intent_name):
                if '.intent' in intent_name:
                    self.register_intent_file(intent_name, None)
                else:
                    intent.name = intent_name
                    self._register_intent(intent, None)
                LOG.debug('Enabling intent {}'.format(intent_name))
                return True
            elif intent:
                LOG.error(f"Could not enable {intent_name}, it's not detached")
            else:
                LOG.error(f"Could not enable {intent_name}, it hasn't been registered.")
            return False

    def set_context(self, context, word='', origin=''):
        if False:
            while True:
                i = 10
        'Add context to intent service\n\n        Args:\n            context:    Keyword\n            word:       word connected to keyword\n            origin:     origin of context\n        '
        if not isinstance(context, str):
            raise ValueError('Context should be a string')
        if not isinstance(word, str):
            raise ValueError('Word should be a string')
        context = to_alnum(self.skill_id) + context
        self.intent_service.set_adapt_context(context, word, origin)

    def handle_set_cross_context(self, message):
        if False:
            while True:
                i = 10
        'Add global context to intent service.'
        context = message.data.get('context')
        word = message.data.get('word')
        origin = message.data.get('origin')
        self.set_context(context, word, origin)

    def handle_remove_cross_context(self, message):
        if False:
            while True:
                i = 10
        'Remove global context from intent service.'
        context = message.data.get('context')
        self.remove_context(context)

    def set_cross_skill_context(self, context, word=''):
        if False:
            print('Hello World!')
        'Tell all skills to add a context to intent service\n\n        Args:\n            context:    Keyword\n            word:       word connected to keyword\n        '
        self.bus.emit(Message('mycroft.skill.set_cross_context', {'context': context, 'word': word, 'origin': self.skill_id}))

    def remove_cross_skill_context(self, context):
        if False:
            while True:
                i = 10
        'Tell all skills to remove a keyword from the context manager.'
        if not isinstance(context, str):
            raise ValueError('context should be a string')
        self.bus.emit(Message('mycroft.skill.remove_cross_context', {'context': context}))

    def remove_context(self, context):
        if False:
            i = 10
            return i + 15
        'Remove a keyword from the context manager.'
        if not isinstance(context, str):
            raise ValueError('context should be a string')
        context = to_alnum(self.skill_id) + context
        self.intent_service.remove_adapt_context(context)

    def register_vocabulary(self, entity, entity_type):
        if False:
            return 10
        ' Register a word to a keyword\n\n        Args:\n            entity:         word to register\n            entity_type:    Intent handler entity to tie the word to\n        '
        keyword_type = to_alnum(self.skill_id) + entity_type
        with self.intent_service_lock:
            self.intent_service.register_adapt_keyword(keyword_type, entity)

    def register_regex(self, regex_str):
        if False:
            while True:
                i = 10
        'Register a new regex.\n        Args:\n            regex_str: Regex string\n        '
        self.log.debug('registering regex string: ' + regex_str)
        regex = munge_regex(regex_str, self.skill_id)
        re.compile(regex)
        with self.intent_service_lock:
            self.intent_service.register_adapt_regex(regex)

    def speak(self, utterance, expect_response=False, wait=False, meta=None):
        if False:
            while True:
                i = 10
        'Speak a sentence.\n\n        Args:\n            utterance (str):        sentence mycroft should speak\n            expect_response (bool): set to True if Mycroft should listen\n                                    for a response immediately after\n                                    speaking the utterance.\n            wait (bool):            set to True to block while the text\n                                    is being spoken.\n            meta:                   Information of what built the sentence.\n        '
        meta = meta or {}
        meta['skill'] = self.name
        self.enclosure.register(self.name)
        data = {'utterance': utterance, 'expect_response': expect_response, 'meta': meta}
        message = dig_for_message()
        m = message.forward('speak', data) if message else Message('speak', data)
        self.bus.emit(m)
        if wait:
            wait_while_speaking()

    def speak_dialog(self, key, data=None, expect_response=False, wait=False):
        if False:
            i = 10
            return i + 15
        ' Speak a random sentence from a dialog file.\n\n        Args:\n            key (str): dialog file key (e.g. "hello" to speak from the file\n                                        "locale/en-us/hello.dialog")\n            data (dict): information used to populate sentence\n            expect_response (bool): set to True if Mycroft should listen\n                                    for a response immediately after\n                                    speaking the utterance.\n            wait (bool):            set to True to block while the text\n                                    is being spoken.\n        '
        if self.dialog_renderer:
            data = data or {}
            self.speak(self.dialog_renderer.render(key, data), expect_response, wait, meta={'dialog': key, 'data': data})
        else:
            self.log.warning('dialog_render is None, does the locale/dialog folder exist?')
            self.speak(key, expect_response, wait, {})

    def acknowledge(self):
        if False:
            print('Hello World!')
        'Acknowledge a successful request.\n\n        This method plays a sound to acknowledge a request that does not\n        require a verbal response. This is intended to provide simple feedback\n        to the user that their request was handled successfully.\n        '
        audio_file = resolve_resource_file(self.config_core.get('sounds').get('acknowledge'))
        if not audio_file:
            LOG.warning("Could not find 'acknowledge' audio file!")
            return
        process = play_audio_file(audio_file)
        if not process:
            LOG.warning("Unable to play 'acknowledge' audio file!")

    def init_dialog(self, root_directory):
        if False:
            return 10
        dialog_dir = join(root_directory, 'dialog', self.lang)
        if exists(dialog_dir):
            self.dialog_renderer = load_dialogs(dialog_dir)
        elif exists(join(root_directory, 'locale', self.lang)):
            locale_path = join(root_directory, 'locale', self.lang)
            self.dialog_renderer = load_dialogs(locale_path)
        else:
            LOG.debug('No dialog loaded')

    def load_data_files(self, root_directory=None):
        if False:
            while True:
                i = 10
        'Called by the skill loader to load intents, dialogs, etc.\n\n        Args:\n            root_directory (str): root folder to use when loading files.\n        '
        root_directory = root_directory or self.root_dir
        self.init_dialog(root_directory)
        self.load_vocab_files(root_directory)
        self.load_regex_files(root_directory)

    def load_vocab_files(self, root_directory):
        if False:
            print('Hello World!')
        ' Load vocab files found under root_directory.\n\n        Args:\n            root_directory (str): root folder to use when loading files\n        '
        keywords = []
        vocab_dir = join(root_directory, 'vocab', self.lang)
        locale_dir = join(root_directory, 'locale', self.lang)
        if exists(vocab_dir):
            keywords = load_vocabulary(vocab_dir, self.skill_id)
        elif exists(locale_dir):
            keywords = load_vocabulary(locale_dir, self.skill_id)
        else:
            LOG.debug('No vocab loaded')
        for vocab_type in keywords:
            for line in keywords[vocab_type]:
                entity = line[0]
                aliases = line[1:]
                with self.intent_service_lock:
                    self.intent_service.register_adapt_keyword(vocab_type, entity, aliases)

    def load_regex_files(self, root_directory):
        if False:
            return 10
        ' Load regex files found under the skill directory.\n\n        Args:\n            root_directory (str): root folder to use when loading files\n        '
        regexes = []
        regex_dir = join(root_directory, 'regex', self.lang)
        locale_dir = join(root_directory, 'locale', self.lang)
        if exists(regex_dir):
            regexes = load_regex(regex_dir, self.skill_id)
        elif exists(locale_dir):
            regexes = load_regex(locale_dir, self.skill_id)
        for regex in regexes:
            with self.intent_service_lock:
                self.intent_service.register_adapt_regex(regex)

    def __handle_stop(self, _):
        if False:
            return 10
        'Handler for the "mycroft.stop" signal. Runs the user defined\n        `stop()` method.\n        '

        def __stop_timeout():
            if False:
                for i in range(10):
                    print('nop')
            self.bus.emit(Message('mycroft.stop.handled', {'skill_id': str(self.skill_id) + ':'}))
        timer = Timer(0.1, __stop_timeout)
        try:
            if self.stop():
                self.bus.emit(Message('mycroft.stop.handled', {'by': 'skill:' + self.skill_id}))
            timer.cancel()
        except Exception:
            timer.cancel()
            LOG.error('Failed to stop skill: {}'.format(self.name), exc_info=True)

    def stop(self):
        if False:
            return 10
        'Optional method implemented by subclass.'
        pass

    def shutdown(self):
        if False:
            return 10
        'Optional shutdown proceedure implemented by subclass.\n\n        This method is intended to be called during the skill process\n        termination. The skill implementation must shutdown all processes and\n        operations in execution.\n        '
        pass

    def default_shutdown(self):
        if False:
            return 10
        'Parent function called internally to shut down everything.\n\n        Shuts down known entities and calls skill specific shutdown method.\n        '
        try:
            self.shutdown()
        except Exception as e:
            LOG.error('Skill specific shutdown function encountered an error: {}'.format(repr(e)))
        self.settings_change_callback = None
        if self.settings != self._initial_settings and Path(self.root_dir).exists():
            save_settings(self.settings_write_path, self.settings)
        if self.settings_meta:
            self.settings_meta.stop()
        self.gui.shutdown()
        self.event_scheduler.shutdown()
        self.events.clear()
        self.bus.emit(Message('detach_skill', {'skill_id': str(self.skill_id) + ':'}))
        try:
            self.stop()
        except Exception:
            LOG.error('Failed to stop skill: {}'.format(self.name), exc_info=True)

    def schedule_event(self, handler, when, data=None, name=None, context=None):
        if False:
            i = 10
            return i + 15
        'Schedule a single-shot event.\n\n        Args:\n            handler:               method to be called\n            when (datetime/int/float):   datetime (in system timezone) or\n                                   number of seconds in the future when the\n                                   handler should be called\n            data (dict, optional): data to send when the handler is called\n            name (str, optional):  reference name\n                                   NOTE: This will not warn or replace a\n                                   previously scheduled event of the same\n                                   name.\n            context (dict, optional): context (dict, optional): message\n                                      context to send when the handler\n                                      is called\n        '
        message = dig_for_message()
        context = context or message.context if message else {}
        return self.event_scheduler.schedule_event(handler, when, data, name, context=context)

    def schedule_repeating_event(self, handler, when, frequency, data=None, name=None, context=None):
        if False:
            print('Hello World!')
        'Schedule a repeating event.\n\n        Args:\n            handler:                method to be called\n            when (datetime):        time (in system timezone) for first\n                                    calling the handler, or None to\n                                    initially trigger <frequency> seconds\n                                    from now\n            frequency (float/int):  time in seconds between calls\n            data (dict, optional):  data to send when the handler is called\n            name (str, optional):   reference name, must be unique\n            context (dict, optional): context (dict, optional): message\n                                      context to send when the handler\n                                      is called\n        '
        message = dig_for_message()
        context = context or message.context if message else {}
        return self.event_scheduler.schedule_repeating_event(handler, when, frequency, data, name, context=context)

    def update_scheduled_event(self, name, data=None):
        if False:
            print('Hello World!')
        'Change data of event.\n\n        Args:\n            name (str): reference name of event (from original scheduling)\n            data (dict): event data\n        '
        return self.event_scheduler.update_scheduled_event(name, data)

    def cancel_scheduled_event(self, name):
        if False:
            while True:
                i = 10
        'Cancel a pending event. The event will no longer be scheduled\n        to be executed\n\n        Args:\n            name (str): reference name of event (from original scheduling)\n        '
        return self.event_scheduler.cancel_scheduled_event(name)

    def get_scheduled_event_status(self, name):
        if False:
            i = 10
            return i + 15
        'Get scheduled event data and return the amount of time left\n\n        Args:\n            name (str): reference name of event (from original scheduling)\n\n        Returns:\n            int: the time left in seconds\n\n        Raises:\n            Exception: Raised if event is not found\n        '
        return self.event_scheduler.get_scheduled_event_status(name)

    def cancel_all_repeating_events(self):
        if False:
            for i in range(10):
                print('nop')
        'Cancel any repeating events started by the skill.'
        return self.event_scheduler.cancel_all_repeating_events()