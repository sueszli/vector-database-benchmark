"""Mycroft's intent service, providing intent parsing since forever!"""
from copy import copy
import time
from mycroft.configuration import Configuration, set_default_lf_lang
from mycroft.util.log import LOG
from mycroft.util.parse import normalize
from mycroft.metrics import report_timing, Stopwatch
from .intent_services import AdaptService, AdaptIntent, FallbackService, PadatiousService, PadatiousMatcher, IntentMatch
from .intent_service_interface import open_intent_envelope

def _get_message_lang(message):
    if False:
        return 10
    'Get the language from the message or the default language.\n\n    Args:\n        message: message to check for language code.\n\n    Returns:\n        The languge code from the message or the default language.\n    '
    default_lang = Configuration.get().get('lang', 'en-us')
    return message.data.get('lang', default_lang).lower()

def _normalize_all_utterances(utterances):
    if False:
        return 10
    'Create normalized versions and pair them with the original utterance.\n\n    This will create a list of tuples with the original utterance as the\n    first item and if normalizing changes the utterance the normalized version\n    will be set as the second item in the tuple, if normalization doesn\'t\n    change anything the tuple will only have the "raw" original utterance.\n\n    Args:\n        utterances (list): list of utterances to normalize\n\n    Returns:\n        list of tuples, [(original utterance, normalized) ... ]\n    '
    norm_utterances = [normalize(u.lower(), remove_articles=False) for u in utterances]
    combined = []
    for (utt, norm) in zip(utterances, norm_utterances):
        if utt == norm:
            combined.append((utt,))
        else:
            combined.append((utt, norm))
    LOG.debug('Utterances: {}'.format(combined))
    return combined

class IntentService:
    """Mycroft intent service. parses utterances using a variety of systems.

    The intent service also provides the internal API for registering and
    querying the intent service.
    """

    def __init__(self, bus):
        if False:
            i = 10
            return i + 15
        self.bus = bus
        self.skill_names = {}
        config = Configuration.get()
        self.adapt_service = AdaptService(config.get('context', {}))
        try:
            self.padatious_service = PadatiousService(bus, config['padatious'])
        except Exception as err:
            LOG.exception('Failed to create padatious handlers ({})'.format(repr(err)))
        self.fallback = FallbackService(bus)
        self.bus.on('register_vocab', self.handle_register_vocab)
        self.bus.on('register_intent', self.handle_register_intent)
        self.bus.on('recognizer_loop:utterance', self.handle_utterance)
        self.bus.on('detach_intent', self.handle_detach_intent)
        self.bus.on('detach_skill', self.handle_detach_skill)
        self.bus.on('add_context', self.handle_add_context)
        self.bus.on('remove_context', self.handle_remove_context)
        self.bus.on('clear_context', self.handle_clear_context)
        self.bus.on('mycroft.speech.recognition.unknown', self.reset_converse)
        self.bus.on('mycroft.skills.loaded', self.update_skill_name_dict)

        def add_active_skill_handler(message):
            if False:
                print('Hello World!')
            self.add_active_skill(message.data['skill_id'])
        self.bus.on('active_skill_request', add_active_skill_handler)
        self.active_skills = []
        self.converse_timeout = 5
        self.registered_vocab = []
        self.bus.on('intent.service.intent.get', self.handle_get_intent)
        self.bus.on('intent.service.skills.get', self.handle_get_skills)
        self.bus.on('intent.service.active_skills.get', self.handle_get_active_skills)
        self.bus.on('intent.service.adapt.get', self.handle_get_adapt)
        self.bus.on('intent.service.adapt.manifest.get', self.handle_adapt_manifest)
        self.bus.on('intent.service.adapt.vocab.manifest.get', self.handle_vocab_manifest)
        self.bus.on('intent.service.padatious.get', self.handle_get_padatious)
        self.bus.on('intent.service.padatious.manifest.get', self.handle_padatious_manifest)
        self.bus.on('intent.service.padatious.entities.manifest.get', self.handle_entity_manifest)

    @property
    def registered_intents(self):
        if False:
            for i in range(10):
                print('nop')
        return [parser.__dict__ for parser in self.adapt_service.engine.intent_parsers]

    def update_skill_name_dict(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Messagebus handler, updates dict of id to skill name conversions.'
        self.skill_names[message.data['id']] = message.data['name']

    def get_skill_name(self, skill_id):
        if False:
            i = 10
            return i + 15
        "Get skill name from skill ID.\n\n        Args:\n            skill_id: a skill id as encoded in Intent handlers.\n\n        Returns:\n            (str) Skill name or the skill id if the skill wasn't found\n        "
        return self.skill_names.get(skill_id, skill_id)

    def reset_converse(self, message):
        if False:
            i = 10
            return i + 15
        'Let skills know there was a problem with speech recognition'
        lang = _get_message_lang(message)
        set_default_lf_lang(lang)
        for skill in copy(self.active_skills):
            self.do_converse(None, skill[0], lang, message)

    def do_converse(self, utterances, skill_id, lang, message):
        if False:
            for i in range(10):
                print('nop')
        'Call skill and ask if they want to process the utterance.\n\n        Args:\n            utterances (list of tuples): utterances paired with normalized\n                                         versions.\n            skill_id: skill to query.\n            lang (str): current language\n            message (Message): message containing interaction info.\n        '
        converse_msg = message.reply('skill.converse.request', {'skill_id': skill_id, 'utterances': utterances, 'lang': lang})
        result = self.bus.wait_for_response(converse_msg, 'skill.converse.response')
        if result and 'error' in result.data:
            self.handle_converse_error(result)
            ret = False
        elif result is not None:
            ret = result.data.get('result', False)
        else:
            ret = False
        return ret

    def handle_converse_error(self, message):
        if False:
            i = 10
            return i + 15
        'Handle error in converse system.\n\n        Args:\n            message (Message): info about the error.\n        '
        skill_id = message.data['skill_id']
        error_msg = message.data['error']
        LOG.error('{}: {}'.format(skill_id, error_msg))
        if message.data['error'] == 'skill id does not exist':
            self.remove_active_skill(skill_id)

    def remove_active_skill(self, skill_id):
        if False:
            return 10
        'Remove a skill from being targetable by converse.\n\n        Args:\n            skill_id (str): skill to remove\n        '
        for skill in self.active_skills:
            if skill[0] == skill_id:
                self.active_skills.remove(skill)

    def add_active_skill(self, skill_id):
        if False:
            return 10
        "Add a skill or update the position of an active skill.\n\n        The skill is added to the front of the list, if it's already in the\n        list it's removed so there is only a single entry of it.\n\n        Args:\n            skill_id (str): identifier of skill to be added.\n        "
        if skill_id != '':
            self.remove_active_skill(skill_id)
            self.active_skills.insert(0, [skill_id, time.time()])
        else:
            LOG.warning("Skill ID was empty, won't add to list of active skills.")

    def send_metrics(self, intent, context, stopwatch):
        if False:
            while True:
                i = 10
        'Send timing metrics to the backend.\n\n        NOTE: This only applies to those with Opt In.\n\n        Args:\n            intent (IntentMatch or None): intet match info\n            context (dict): context info about the interaction\n            stopwatch (StopWatch): Timing info about the skill parsing.\n        '
        ident = context['ident'] if 'ident' in context else None
        if intent and intent.intent_service == 'Converse':
            intent_type = '{}:{}'.format(intent.skill_id, 'converse')
        elif intent and intent.intent_service == 'Fallback':
            intent_type = 'fallback'
        elif intent:
            parts = intent.intent_type.split(':')
            intent_type = self.get_skill_name(parts[0])
            if len(parts) > 1:
                intent_type = ':'.join([intent_type] + parts[1:])
        else:
            intent_type = 'intent_failure'
        report_timing(ident, 'intent_service', stopwatch, {'intent_type': intent_type})

    def handle_utterance(self, message):
        if False:
            i = 10
            return i + 15
        "Main entrypoint for handling user utterances with Mycroft skills\n\n        Monitor the messagebus for 'recognizer_loop:utterance', typically\n        generated by a spoken interaction but potentially also from a CLI\n        or other method of injecting a 'user utterance' into the system.\n\n        Utterances then work through this sequence to be handled:\n        1) Active skills attempt to handle using converse()\n        2) Padatious high match intents (conf > 0.95)\n        3) Adapt intent handlers\n        5) High Priority Fallbacks\n        6) Padatious near match intents (conf > 0.8)\n        7) General Fallbacks\n        8) Padatious loose match intents (conf > 0.5)\n        9) Catch all fallbacks including Unknown intent handler\n\n        If all these fail the complete_intent_failure message will be sent\n        and a generic info of the failure will be spoken.\n\n        Args:\n            message (Message): The messagebus data\n        "
        try:
            lang = _get_message_lang(message)
            set_default_lf_lang(lang)
            utterances = message.data.get('utterances', [])
            combined = _normalize_all_utterances(utterances)
            stopwatch = Stopwatch()
            padatious_matcher = PadatiousMatcher(self.padatious_service)
            match_funcs = [self._converse, padatious_matcher.match_high, self.adapt_service.match_intent, self.fallback.high_prio, padatious_matcher.match_medium, self.fallback.medium_prio, padatious_matcher.match_low, self.fallback.low_prio]
            match = None
            with stopwatch:
                for match_func in match_funcs:
                    match = match_func(combined, lang, message)
                    if match:
                        break
            if match:
                if match.skill_id:
                    self.add_active_skill(match.skill_id)
                if match.intent_type:
                    reply = message.reply(match.intent_type, match.intent_data)
                    reply.data['utterances'] = utterances
                    self.bus.emit(reply)
            else:
                self.send_complete_intent_failure(message)
            self.send_metrics(match, message.context, stopwatch)
        except Exception as err:
            LOG.exception(err)

    def _converse(self, utterances, lang, message):
        if False:
            i = 10
            return i + 15
        'Give active skills a chance at the utterance\n\n        Args:\n            utterances (list):  list of utterances\n            lang (string):      4 letter ISO language code\n            message (Message):  message to use to generate reply\n\n        Returns:\n            IntentMatch if handled otherwise None.\n        '
        utterances = [item for tup in utterances for item in tup]
        self.active_skills = [skill for skill in self.active_skills if time.time() - skill[1] <= self.converse_timeout * 60]
        for skill in copy(self.active_skills):
            if self.do_converse(utterances, skill[0], lang, message):
                return IntentMatch('Converse', None, None, skill[0])
        return None

    def send_complete_intent_failure(self, message):
        if False:
            while True:
                i = 10
        'Send a message that no skill could handle the utterance.\n\n        Args:\n            message (Message): original message to forward from\n        '
        self.bus.emit(message.forward('complete_intent_failure'))

    def handle_register_vocab(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Register adapt vocabulary.\n\n        Args:\n            message (Message): message containing vocab info\n        '
        if _is_old_style_keyword_message(message):
            LOG.warning('Deprecated: Registering keywords with old message. This will be removed in v22.02.')
            _update_keyword_message(message)
        entity_value = message.data.get('entity_value')
        entity_type = message.data.get('entity_type')
        regex_str = message.data.get('regex')
        alias_of = message.data.get('alias_of')
        self.adapt_service.register_vocabulary(entity_value, entity_type, alias_of, regex_str)
        self.registered_vocab.append(message.data)

    def handle_register_intent(self, message):
        if False:
            print('Hello World!')
        'Register adapt intent.\n\n        Args:\n            message (Message): message containing intent info\n        '
        intent = open_intent_envelope(message)
        self.adapt_service.register_intent(intent)

    def handle_detach_intent(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Remover adapt intent.\n\n        Args:\n            message (Message): message containing intent info\n        '
        intent_name = message.data.get('intent_name')
        self.adapt_service.detach_intent(intent_name)

    def handle_detach_skill(self, message):
        if False:
            while True:
                i = 10
        'Remove all intents registered for a specific skill.\n\n        Args:\n            message (Message): message containing intent info\n        '
        skill_id = message.data.get('skill_id')
        self.adapt_service.detach_skill(skill_id)

    def handle_add_context(self, message):
        if False:
            while True:
                i = 10
        "Add context\n\n        Args:\n            message: data contains the 'context' item to add\n                     optionally can include 'word' to be injected as\n                     an alias for the context item.\n        "
        entity = {'confidence': 1.0}
        context = message.data.get('context')
        word = message.data.get('word') or ''
        origin = message.data.get('origin') or ''
        if not isinstance(word, str):
            word = str(word)
        entity['data'] = [(word, context)]
        entity['match'] = word
        entity['key'] = word
        entity['origin'] = origin
        self.adapt_service.context_manager.inject_context(entity)

    def handle_remove_context(self, message):
        if False:
            while True:
                i = 10
        "Remove specific context\n\n        Args:\n            message: data contains the 'context' item to remove\n        "
        context = message.data.get('context')
        if context:
            self.adapt_service.context_manager.remove_context(context)

    def handle_clear_context(self, _):
        if False:
            while True:
                i = 10
        'Clears all keywords from context '
        self.adapt_service.context_manager.clear_context()

    def handle_get_intent(self, message):
        if False:
            i = 10
            return i + 15
        'Get intent from either adapt or padatious.\n\n        Args:\n            message (Message): message containing utterance\n        '
        utterance = message.data['utterance']
        lang = message.data.get('lang', 'en-us')
        combined = _normalize_all_utterances([utterance])
        padatious_matcher = PadatiousMatcher(self.padatious_service)
        match_funcs = [padatious_matcher.match_high, self.adapt_service.match_intent, padatious_matcher.match_medium, padatious_matcher.match_low]
        for match_func in match_funcs:
            match = match_func(combined, lang, message)
            if match:
                if match.intent_type:
                    intent_data = match.intent_data
                    intent_data['intent_name'] = match.intent_type
                    intent_data['intent_service'] = match.intent_service
                    intent_data['skill_id'] = match.skill_id
                    intent_data['handler'] = match_func.__name__
                    self.bus.emit(message.reply('intent.service.intent.reply', {'intent': intent_data}))
                return
        self.bus.emit(message.reply('intent.service.intent.reply', {'intent': None}))

    def handle_get_skills(self, message):
        if False:
            i = 10
            return i + 15
        'Send registered skills to caller.\n\n        Argument:\n            message: query message to reply to.\n        '
        self.bus.emit(message.reply('intent.service.skills.reply', {'skills': self.skill_names}))

    def handle_get_active_skills(self, message):
        if False:
            print('Hello World!')
        'Send active skills to caller.\n\n        Argument:\n            message: query message to reply to.\n        '
        self.bus.emit(message.reply('intent.service.active_skills.reply', {'skills': self.active_skills}))

    def handle_get_adapt(self, message):
        if False:
            print('Hello World!')
        'handler getting the adapt response for an utterance.\n\n        Args:\n            message (Message): message containing utterance\n        '
        utterance = message.data['utterance']
        lang = message.data.get('lang', 'en-us')
        combined = _normalize_all_utterances([utterance])
        intent = self.adapt_service.match_intent(combined, lang)
        intent_data = intent.intent_data if intent else None
        self.bus.emit(message.reply('intent.service.adapt.reply', {'intent': intent_data}))

    def handle_adapt_manifest(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Send adapt intent manifest to caller.\n\n        Argument:\n            message: query message to reply to.\n        '
        self.bus.emit(message.reply('intent.service.adapt.manifest', {'intents': self.registered_intents}))

    def handle_vocab_manifest(self, message):
        if False:
            return 10
        'Send adapt vocabulary manifest to caller.\n\n        Argument:\n            message: query message to reply to.\n        '
        self.bus.emit(message.reply('intent.service.adapt.vocab.manifest', {'vocab': self.registered_vocab}))

    def handle_get_padatious(self, message):
        if False:
            for i in range(10):
                print('nop')
        'messagebus handler for perfoming padatious parsing.\n\n        Args:\n            message (Message): message triggering the method\n        '
        utterance = message.data['utterance']
        norm = message.data.get('norm_utt', utterance)
        intent = self.padatious_service.calc_intent(utterance)
        if not intent and norm != utterance:
            intent = self.padatious_service.calc_intent(norm)
        if intent:
            intent = intent.__dict__
        self.bus.emit(message.reply('intent.service.padatious.reply', {'intent': intent}))

    def handle_padatious_manifest(self, message):
        if False:
            print('Hello World!')
        'Messagebus handler returning the registered padatious intents.\n\n        Args:\n            message (Message): message triggering the method\n        '
        self.bus.emit(message.reply('intent.service.padatious.manifest', {'intents': self.padatious_service.registered_intents}))

    def handle_entity_manifest(self, message):
        if False:
            print('Hello World!')
        'Messagebus handler returning the registered padatious entities.\n\n        Args:\n            message (Message): message triggering the method\n        '
        self.bus.emit(message.reply('intent.service.padatious.entities.manifest', {'entities': self.padatious_service.registered_entities}))

def _is_old_style_keyword_message(message):
    if False:
        while True:
            i = 10
    'Simple check that the message is not using the updated format.\n\n    TODO: Remove in v22.02\n\n    Args:\n        message (Message): Message object to check\n\n    Returns:\n        (bool) True if this is an old messagem, else False\n    '
    return 'entity_value' not in message.data and 'start' in message.data

def _update_keyword_message(message):
    if False:
        print('Hello World!')
    'Make old style keyword registration message compatible.\n\n    Copies old keys in message data to new names.\n\n    Args:\n        message (Message): Message to update\n    '
    message.data['entity_value'] = message.data['start']
    message.data['entity_type'] = message.data['end']