"""The intent service interface offers a unified wrapper class for the
Intent Service. Including both adapt and padatious.
"""
from os.path import exists, isfile
from adapt.intent import Intent
from mycroft.messagebus.message import Message
from mycroft.messagebus.client import MessageBusClient
from mycroft.util import create_daemon
from mycroft.util.log import LOG

class IntentServiceInterface:
    """Interface to communicate with the Mycroft intent service.

    This class wraps the messagebus interface of the intent service allowing
    for easier interaction with the service. It wraps both the Adapt and
    Padatious parts of the intent services.
    """

    def __init__(self, bus=None):
        if False:
            i = 10
            return i + 15
        self.bus = bus
        self.registered_intents = []
        self.detached_intents = []

    def set_bus(self, bus):
        if False:
            print('Hello World!')
        self.bus = bus

    def register_adapt_keyword(self, vocab_type, entity, aliases=None):
        if False:
            return 10
        'Send a message to the intent service to add an Adapt keyword.\n\n            vocab_type(str): Keyword reference\n            entity (str): Primary keyword\n            aliases (list): List of alternative keywords\n        '
        aliases = aliases or []
        entity_data = {'entity_value': entity, 'entity_type': vocab_type}
        compatibility_data = {'start': entity, 'end': vocab_type}
        self.bus.emit(Message('register_vocab', {**entity_data, **compatibility_data}))
        for alias in aliases:
            alias_data = {'entity_value': alias, 'entity_type': vocab_type, 'alias_of': entity}
            compatibility_data = {'start': alias, 'end': vocab_type}
            self.bus.emit(Message('register_vocab', {**alias_data, **compatibility_data}))

    def register_adapt_regex(self, regex):
        if False:
            i = 10
            return i + 15
        'Register a regex with the intent service.\n\n        Args:\n            regex (str): Regex to be registered, (Adapt extracts keyword\n                         reference from named match group.\n        '
        self.bus.emit(Message('register_vocab', {'regex': regex}))

    def register_adapt_intent(self, name, intent_parser):
        if False:
            print('Hello World!')
        'Register an Adapt intent parser object.\n\n        Serializes the intent_parser and sends it over the messagebus to\n        registered.\n        '
        self.bus.emit(Message('register_intent', intent_parser.__dict__))
        self.registered_intents.append((name, intent_parser))
        self.detached_intents = [detached for detached in self.detached_intents if detached[0] != name]

    def detach_intent(self, intent_name):
        if False:
            print('Hello World!')
        'Remove an intent from the intent service.\n\n        The intent is saved in the list of detached intents for use when\n        re-enabling an intent.\n\n        Args:\n            intent_name(str): Intent reference\n        '
        name = intent_name.split(':')[1]
        if name in self:
            self.bus.emit(Message('detach_intent', {'intent_name': intent_name}))
            self.detached_intents.append((name, self.get_intent(name)))
            self.registered_intents = [pair for pair in self.registered_intents if pair[0] != name]

    def intent_is_detached(self, intent_name):
        if False:
            while True:
                i = 10
        'Determine if an intent is detached.\n\n        Args:\n            intent_name(str): Intent reference\n\n        Returns:\n            (bool) True if intent is found, else False.\n        '
        for (name, _) in self.detached_intents:
            if name == intent_name:
                return True
        return False

    def set_adapt_context(self, context, word, origin):
        if False:
            while True:
                i = 10
        'Set an Adapt context.\n\n        Args:\n            context (str): context keyword name\n            word (str): word to register\n            origin (str): original origin of the context (for cross context)\n        '
        self.bus.emit(Message('add_context', {'context': context, 'word': word, 'origin': origin}))

    def remove_adapt_context(self, context):
        if False:
            i = 10
            return i + 15
        'Remove an active Adapt context.\n\n        Args:\n            context(str): name of context to remove\n        '
        self.bus.emit(Message('remove_context', {'context': context}))

    def register_padatious_intent(self, intent_name, filename):
        if False:
            i = 10
            return i + 15
        'Register a padatious intent file with Padatious.\n\n        Args:\n            intent_name(str): intent identifier\n            filename(str): complete file path for entity file\n        '
        if not isinstance(filename, str):
            raise ValueError('Filename path must be a string')
        if not exists(filename):
            raise FileNotFoundError('Unable to find "{}"'.format(filename))
        data = {'file_name': filename, 'name': intent_name}
        self.bus.emit(Message('padatious:register_intent', data))
        self.registered_intents.append((intent_name.split(':')[-1], data))

    def register_padatious_entity(self, entity_name, filename):
        if False:
            i = 10
            return i + 15
        'Register a padatious entity file with Padatious.\n\n        Args:\n            entity_name(str): entity name\n            filename(str): complete file path for entity file\n        '
        if not isinstance(filename, str):
            raise ValueError('Filename path must be a string')
        if not exists(filename):
            raise FileNotFoundError('Unable to find "{}"'.format(filename))
        self.bus.emit(Message('padatious:register_entity', {'file_name': filename, 'name': entity_name}))

    def __iter__(self):
        if False:
            return 10
        'Iterator over the registered intents.\n\n        Returns an iterator returning name-handler pairs of the registered\n        intent handlers.\n        '
        return iter(self.registered_intents)

    def __contains__(self, val):
        if False:
            i = 10
            return i + 15
        'Checks if an intent name has been registered.'
        return val in [i[0] for i in self.registered_intents]

    def get_intent(self, intent_name):
        if False:
            return 10
        'Get intent from intent_name.\n\n        This will find both enabled and disabled intents.\n\n        Args:\n            intent_name (str): name to find.\n\n        Returns:\n            Found intent or None if none were found.\n        '
        for (name, intent) in self:
            if name == intent_name:
                return intent
        for (name, intent) in self.detached_intents:
            if name == intent_name:
                return intent
        return None

class IntentQueryApi:
    """
    Query Intent Service at runtime
    """

    def __init__(self, bus=None, timeout=5):
        if False:
            for i in range(10):
                print('nop')
        if bus is None:
            bus = MessageBusClient()
            create_daemon(bus.run_forever)
        self.bus = bus
        self.timeout = timeout

    def get_adapt_intent(self, utterance, lang='en-us'):
        if False:
            print('Hello World!')
        ' get best adapt intent for utterance '
        msg = Message('intent.service.adapt.get', {'utterance': utterance, 'lang': lang}, context={'destination': 'intent_service', 'source': 'intent_api'})
        resp = self.bus.wait_for_response(msg, 'intent.service.adapt.reply', timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        return data['intent']

    def get_padatious_intent(self, utterance, lang='en-us'):
        if False:
            return 10
        ' get best padatious intent for utterance '
        msg = Message('intent.service.padatious.get', {'utterance': utterance, 'lang': lang}, context={'destination': 'intent_service', 'source': 'intent_api'})
        resp = self.bus.wait_for_response(msg, 'intent.service.padatious.reply', timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        return data['intent']

    def get_intent(self, utterance, lang='en-us'):
        if False:
            return 10
        ' get best intent for utterance '
        msg = Message('intent.service.intent.get', {'utterance': utterance, 'lang': lang}, context={'destination': 'intent_service', 'source': 'intent_api'})
        resp = self.bus.wait_for_response(msg, 'intent.service.intent.reply', timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        return data['intent']

    def get_skill(self, utterance, lang='en-us'):
        if False:
            while True:
                i = 10
        ' get skill that utterance will trigger '
        intent = self.get_intent(utterance, lang)
        if not intent:
            return None
        if intent.get('skill_id'):
            return intent['skill_id']
        if intent.get('intent_name'):
            return intent['name'].split(':')[0]
        if intent.get('intent_type'):
            return intent['intent_type'].split(':')[0]
        return None

    def get_skills_manifest(self):
        if False:
            print('Hello World!')
        msg = Message('intent.service.skills.get', context={'destination': 'intent_service', 'source': 'intent_api'})
        resp = self.bus.wait_for_response(msg, 'intent.service.skills.reply', timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        return data['skills']

    def get_active_skills(self, include_timestamps=False):
        if False:
            for i in range(10):
                print('nop')
        msg = Message('intent.service.active_skills.get', context={'destination': 'intent_service', 'source': 'intent_api'})
        resp = self.bus.wait_for_response(msg, 'intent.service.active_skills.reply', timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        if include_timestamps:
            return data['skills']
        return [s[0] for s in data['skills']]

    def get_adapt_manifest(self):
        if False:
            for i in range(10):
                print('nop')
        msg = Message('intent.service.adapt.manifest.get', context={'destination': 'intent_service', 'source': 'intent_api'})
        resp = self.bus.wait_for_response(msg, 'intent.service.adapt.manifest', timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        return data['intents']

    def get_padatious_manifest(self):
        if False:
            for i in range(10):
                print('nop')
        msg = Message('intent.service.padatious.manifest.get', context={'destination': 'intent_service', 'source': 'intent_api'})
        resp = self.bus.wait_for_response(msg, 'intent.service.padatious.manifest', timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        return data['intents']

    def get_intent_manifest(self):
        if False:
            return 10
        padatious = self.get_padatious_manifest()
        adapt = self.get_adapt_manifest()
        return {'adapt': adapt, 'padatious': padatious}

    def get_vocab_manifest(self):
        if False:
            while True:
                i = 10
        msg = Message('intent.service.adapt.vocab.manifest.get', context={'destination': 'intent_service', 'source': 'intent_api'})
        reply_msg_type = 'intent.service.adapt.vocab.manifest'
        resp = self.bus.wait_for_response(msg, reply_msg_type, timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        vocab = {}
        for voc in data['vocab']:
            if voc.get('regex'):
                continue
            if voc['end'] not in vocab:
                vocab[voc['end']] = {'samples': []}
            vocab[voc['end']]['samples'].append(voc['start'])
        return [{'name': voc, 'samples': vocab[voc]['samples']} for voc in vocab]

    def get_regex_manifest(self):
        if False:
            print('Hello World!')
        msg = Message('intent.service.adapt.vocab.manifest.get', context={'destination': 'intent_service', 'source': 'intent_api'})
        reply_msg_type = 'intent.service.adapt.vocab.manifest'
        resp = self.bus.wait_for_response(msg, reply_msg_type, timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        vocab = {}
        for voc in data['vocab']:
            if not voc.get('regex'):
                continue
            name = voc['regex'].split('(?P<')[-1].split('>')[0]
            if name not in vocab:
                vocab[name] = {'samples': []}
            vocab[name]['samples'].append(voc['regex'])
        return [{'name': voc, 'regexes': vocab[voc]['samples']} for voc in vocab]

    def get_entities_manifest(self):
        if False:
            print('Hello World!')
        msg = Message('intent.service.padatious.entities.manifest.get', context={'destination': 'intent_service', 'source': 'intent_api'})
        reply_msg_type = 'intent.service.padatious.entities.manifest'
        resp = self.bus.wait_for_response(msg, reply_msg_type, timeout=self.timeout)
        data = resp.data if resp is not None else {}
        if not data:
            LOG.error('Intent Service timed out!')
            return None
        entities = []
        for ent in data['entities']:
            if isfile(ent['file_name']):
                with open(ent['file_name']) as f:
                    lines = f.read().replace('(', '').replace(')', '').split('\n')
                samples = []
                for l in lines:
                    samples += [a.strip() for a in l.split('|') if a.strip()]
                entities.append({'name': ent['name'], 'samples': samples})
        return entities

    def get_keywords_manifest(self):
        if False:
            return 10
        padatious = self.get_entities_manifest()
        adapt = self.get_vocab_manifest()
        regex = self.get_regex_manifest()
        return {'adapt': adapt, 'padatious': padatious, 'regex': regex}

def open_intent_envelope(message):
    if False:
        return 10
    'Convert dictionary received over messagebus to Intent.'
    intent_dict = message.data
    return Intent(intent_dict.get('name'), intent_dict.get('requires'), intent_dict.get('at_least_one'), intent_dict.get('optional'))