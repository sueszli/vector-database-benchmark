"""Intent service wrapping padatious."""
from functools import lru_cache
from subprocess import call
from threading import Event
from time import time as get_time, sleep
from os.path import expanduser, isfile
from mycroft.configuration import Configuration
from mycroft.messagebus.message import Message
from mycroft.util.log import LOG
from .base import IntentMatch

class PadatiousMatcher:
    """Matcher class to avoid redundancy in padatious intent matching."""

    def __init__(self, service):
        if False:
            print('Hello World!')
        self.service = service
        self.has_result = False
        self.ret = None
        self.conf = None

    def _match_level(self, utterances, limit):
        if False:
            print('Hello World!')
        'Match intent and make sure a certain level of confidence is reached.\n\n        Args:\n            utterances (list of tuples): Utterances to parse, originals paired\n                                         with optional normalized version.\n            limit (float): required confidence level.\n        '
        if not self.has_result:
            padatious_intent = None
            LOG.debug('Padatious Matching confidence > {}'.format(limit))
            for utt in utterances:
                for variant in utt:
                    intent = self.service.calc_intent(variant)
                    if intent:
                        best = padatious_intent.conf if padatious_intent else 0.0
                        if best < intent.conf:
                            padatious_intent = intent
                            padatious_intent.matches['utterance'] = utt[0]
            if padatious_intent:
                skill_id = padatious_intent.name.split(':')[0]
                self.ret = IntentMatch('Padatious', padatious_intent.name, padatious_intent.matches, skill_id)
                self.conf = padatious_intent.conf
            self.has_result = True
        if self.conf and self.conf > limit:
            return self.ret
        return None

    def match_high(self, utterances, _=None, __=None):
        if False:
            return 10
        'Intent matcher for high confidence.\n\n        Args:\n            utterances (list of tuples): Utterances to parse, originals paired\n                                         with optional normalized version.\n        '
        return self._match_level(utterances, 0.95)

    def match_medium(self, utterances, _=None, __=None):
        if False:
            while True:
                i = 10
        'Intent matcher for medium confidence.\n\n        Args:\n            utterances (list of tuples): Utterances to parse, originals paired\n                                         with optional normalized version.\n        '
        return self._match_level(utterances, 0.8)

    def match_low(self, utterances, _=None, __=None):
        if False:
            for i in range(10):
                print('nop')
        'Intent matcher for low confidence.\n\n        Args:\n            utterances (list of tuples): Utterances to parse, originals paired\n                                         with optional normalized version.\n        '
        return self._match_level(utterances, 0.5)

class PadatiousService:
    """Service class for padatious intent matching."""

    def __init__(self, bus, config):
        if False:
            return 10
        self.padatious_config = config
        self.bus = bus
        intent_cache = expanduser(self.padatious_config['intent_cache'])
        try:
            from padatious import IntentContainer
        except ImportError:
            LOG.error('Padatious not installed. Please re-run dev_setup.sh')
            try:
                call(['notify-send', 'Padatious not installed', 'Please run build_host_setup and dev_setup again'])
            except OSError:
                pass
            return
        self.container = IntentContainer(intent_cache)
        self._bus = bus
        self.bus.on('padatious:register_intent', self.register_intent)
        self.bus.on('padatious:register_entity', self.register_entity)
        self.bus.on('detach_intent', self.handle_detach_intent)
        self.bus.on('detach_skill', self.handle_detach_skill)
        self.bus.on('mycroft.skills.initialized', self.train)
        self.finished_training_event = Event()
        self.finished_initial_train = False
        self.train_delay = self.padatious_config['train_delay']
        self.train_time = get_time() + self.train_delay
        self.registered_intents = []
        self.registered_entities = []

    def train(self, message=None):
        if False:
            return 10
        'Perform padatious training.\n\n        Args:\n            message (Message): optional triggering message\n        '
        padatious_single_thread = Configuration.get()['padatious']['single_thread']
        if message is None:
            single_thread = padatious_single_thread
        else:
            single_thread = message.data.get('single_thread', padatious_single_thread)
        self.finished_training_event.clear()
        LOG.info('Training... (single_thread={})'.format(single_thread))
        self.container.train(single_thread=single_thread)
        LOG.info('Training complete.')
        self.finished_training_event.set()
        if not self.finished_initial_train:
            self.bus.emit(Message('mycroft.skills.trained'))
            self.finished_initial_train = True

    def wait_and_train(self):
        if False:
            i = 10
            return i + 15
        'Wait for minimum time between training and start training.'
        if not self.finished_initial_train:
            return
        sleep(self.train_delay)
        if self.train_time < 0.0:
            return
        if self.train_time <= get_time() + 0.01:
            self.train_time = -1.0
            self.train()

    def __detach_intent(self, intent_name):
        if False:
            return 10
        ' Remove an intent if it has been registered.\n\n        Args:\n            intent_name (str): intent identifier\n        '
        if intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            self.container.remove_intent(intent_name)

    def handle_detach_intent(self, message):
        if False:
            i = 10
            return i + 15
        'Messagebus handler for detaching padatious intent.\n\n        Args:\n            message (Message): message triggering action\n        '
        self.__detach_intent(message.data.get('intent_name'))

    def handle_detach_skill(self, message):
        if False:
            i = 10
            return i + 15
        'Messagebus handler for detaching all intents for skill.\n\n        Args:\n            message (Message): message triggering action\n        '
        skill_id = message.data['skill_id']
        remove_list = [i for i in self.registered_intents if skill_id in i]
        for i in remove_list:
            self.__detach_intent(i)

    def _register_object(self, message, object_name, register_func):
        if False:
            while True:
                i = 10
        'Generic method for registering a padatious object.\n\n        Args:\n            message (Message): trigger for action\n            object_name (str): type of entry to register\n            register_func (callable): function to call for registration\n        '
        file_name = message.data['file_name']
        name = message.data['name']
        LOG.debug('Registering Padatious ' + object_name + ': ' + name)
        if not isfile(file_name):
            LOG.warning('Could not find file ' + file_name)
            return
        register_func(name, file_name)
        self.train_time = get_time() + self.train_delay
        self.wait_and_train()

    def register_intent(self, message):
        if False:
            i = 10
            return i + 15
        'Messagebus handler for registering intents.\n\n        Args:\n            message (Message): message triggering action\n        '
        self.registered_intents.append(message.data['name'])
        self._register_object(message, 'intent', self.container.load_intent)

    def register_entity(self, message):
        if False:
            print('Hello World!')
        'Messagebus handler for registering entities.\n\n        Args:\n            message (Message): message triggering action\n        '
        self.registered_entities.append(message.data)
        self._register_object(message, 'entity', self.container.load_entity)

    def calc_intent(self, utt):
        if False:
            return 10
        'Cached version of container calc_intent.\n\n        This improves speed when called multiple times for different confidence\n        levels.\n\n        NOTE: This cache will keep a reference to this class\n        (PadatiousService), but we can live with that since it is used as a\n        singleton.\n\n        Args:\n            utt (str): utterance to calculate best intent for\n        '
        return self.container.calc_intent(utt)