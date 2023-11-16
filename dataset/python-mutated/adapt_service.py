"""An intent parsing service using the Adapt parser."""
from threading import Lock
import time
from adapt.context import ContextManagerFrame
from adapt.engine import IntentDeterminationEngine
from adapt.intent import IntentBuilder
from mycroft.util.log import LOG
from .base import IntentMatch

def _entity_skill_id(skill_id):
    if False:
        print('Hello World!')
    'Helper converting a skill id to the format used in entities.\n\n    Arguments:\n        skill_id (str): skill identifier\n\n    Returns:\n        (str) skill id on the format used by skill entities\n    '
    skill_id = skill_id[:-1]
    skill_id = skill_id.replace('.', '_')
    skill_id = skill_id.replace('-', '_')
    return skill_id

class AdaptIntent(IntentBuilder):
    """Wrapper for IntentBuilder setting a blank name.

    Args:
        name (str): Optional name of intent
    """

    def __init__(self, name=''):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name)

def _strip_result(context_features):
    if False:
        i = 10
        return i + 15
    'Keep only the latest instance of each keyword.\n\n    Arguments\n        context_features (iterable): context features to check.\n    '
    stripped = []
    processed = []
    for feature in context_features:
        keyword = feature['data'][0][1]
        if keyword not in processed:
            stripped.append(feature)
            processed.append(keyword)
    return stripped

class ContextManager:
    """Adapt Context Manager

    Use to track context throughout the course of a conversational session.
    How to manage a session's lifecycle is not captured here.
    """

    def __init__(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        self.frame_stack = []
        self.timeout = timeout * 60

    def clear_context(self):
        if False:
            while True:
                i = 10
        'Remove all contexts.'
        self.frame_stack = []

    def remove_context(self, context_id):
        if False:
            i = 10
            return i + 15
        'Remove a specific context entry.\n\n        Args:\n            context_id (str): context entry to remove\n        '
        self.frame_stack = [(f, t) for (f, t) in self.frame_stack if context_id in f.entities[0].get('data', [])]

    def inject_context(self, entity, metadata=None):
        if False:
            while True:
                i = 10
        "\n        Args:\n            entity(object): Format example...\n                               {'data': 'Entity tag as <str>',\n                                'key': 'entity proper name as <str>',\n                                'confidence': <float>'\n                               }\n            metadata(object): dict, arbitrary metadata about entity injected\n        "
        metadata = metadata or {}
        try:
            if self.frame_stack:
                top_frame = self.frame_stack[0]
            else:
                top_frame = None
            if top_frame and top_frame[0].metadata_matches(metadata):
                top_frame[0].merge_context(entity, metadata)
            else:
                frame = ContextManagerFrame(entities=[entity], metadata=metadata.copy())
                self.frame_stack.insert(0, (frame, time.time()))
        except (IndexError, KeyError):
            pass

    def get_context(self, max_frames=None, missing_entities=None):
        if False:
            while True:
                i = 10
        ' Constructs a list of entities from the context.\n\n        Args:\n            max_frames(int): maximum number of frames to look back\n            missing_entities(list of str): a list or set of tag names,\n            as strings\n\n        Returns:\n            list: a list of entities\n        '
        missing_entities = missing_entities or []
        relevant_frames = [frame[0] for frame in self.frame_stack if time.time() - frame[1] < self.timeout]
        if not max_frames or max_frames > len(relevant_frames):
            max_frames = len(relevant_frames)
        missing_entities = list(missing_entities)
        context = []
        last = ''
        depth = 0
        entity = {}
        for i in range(max_frames):
            frame_entities = [entity.copy() for entity in relevant_frames[i].entities]
            for entity in frame_entities:
                entity['confidence'] = entity.get('confidence', 1.0) / (2.0 + depth)
            context += frame_entities
            if entity['origin'] != last or entity['origin'] == '':
                depth += 1
            last = entity['origin']
        result = []
        if missing_entities:
            for entity in context:
                if entity.get('data') in missing_entities:
                    result.append(entity)
                    missing_entities.remove(entity.get('data'))
        else:
            result = context
        return _strip_result(result)

class AdaptService:
    """Intent service wrapping the Apdapt intent Parser."""

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.config = config
        self.engine = IntentDeterminationEngine()
        self.context_keywords = self.config.get('keywords', [])
        self.context_max_frames = self.config.get('max_frames', 3)
        self.context_timeout = self.config.get('timeout', 2)
        self.context_greedy = self.config.get('greedy', False)
        self.context_manager = ContextManager(self.context_timeout)
        self.lock = Lock()

    def update_context(self, intent):
        if False:
            return 10
        "Updates context with keyword from the intent.\n\n        NOTE: This method currently won't handle one_of intent keywords\n              since it's not using quite the same format as other intent\n              keywords. This is under investigation in adapt, PR pending.\n\n        Args:\n            intent: Intent to scan for keywords\n        "
        for tag in intent['__tags__']:
            if 'entities' not in tag:
                continue
            context_entity = tag['entities'][0]
            if self.context_greedy:
                self.context_manager.inject_context(context_entity)
            elif context_entity['data'][0][1] in self.context_keywords:
                self.context_manager.inject_context(context_entity)

    def match_intent(self, utterances, _=None, __=None):
        if False:
            return 10
        'Run the Adapt engine to search for an matching intent.\n\n        Args:\n            utterances (iterable): utterances for consideration in intent\n            matching. As a practical matter, a single utterance will be\n            passed in most cases.  But there are instances, such as\n            streaming STT that could pass multiple.  Each utterance\n            is represented as a tuple containing the raw, normalized, and\n            possibly other variations of the utterance.\n\n        Returns:\n            Intent structure, or None if no match was found.\n        '
        best_intent = {}

        def take_best(intent, utt):
            if False:
                print('Hello World!')
            nonlocal best_intent
            best = best_intent.get('confidence', 0.0) if best_intent else 0.0
            conf = intent.get('confidence', 0.0)
            if conf > best:
                best_intent = intent
                best_intent['utterance'] = utt
        for utt_tup in utterances:
            for utt in utt_tup:
                try:
                    intents = [i for i in self.engine.determine_intent(utt, 100, include_tags=True, context_manager=self.context_manager)]
                    if intents:
                        utt_best = max(intents, key=lambda x: x.get('confidence', 0.0))
                        take_best(utt_best, utt_tup[0])
                except Exception as err:
                    LOG.exception(err)
        if best_intent:
            self.update_context(best_intent)
            skill_id = best_intent['intent_type'].split(':')[0]
            ret = IntentMatch('Adapt', best_intent['intent_type'], best_intent, skill_id)
        else:
            ret = None
        return ret

    def register_vocab(self, start_concept, end_concept, alias_of, regex_str):
        if False:
            i = 10
            return i + 15
        'Register Vocabulary. DEPRECATED\n\n        This method should not be used, it has been replaced by\n        register_vocabulary().\n        '
        self.register_vocabulary(start_concept, end_concept, alias_of, regex_str)

    def register_vocabulary(self, entity_value, entity_type, alias_of, regex_str):
        if False:
            while True:
                i = 10
        'Register skill vocabulary as adapt entity.\n\n        This will handle both regex registration and registration of normal\n        keywords. if the "regex_str" argument is set all other arguments will\n        be ignored.\n\n        Argument:\n            entity_value: the natural langauge word\n            entity_type: the type/tag of an entity instance\n            alias_of: entity this is an alternative for\n        '
        with self.lock:
            if regex_str:
                self.engine.register_regex_entity(regex_str)
            else:
                self.engine.register_entity(entity_value, entity_type, alias_of=alias_of)

    def register_intent(self, intent):
        if False:
            print('Hello World!')
        'Register new intent with adapt engine.\n\n        Args:\n            intent (IntentParser): IntentParser to register\n        '
        with self.lock:
            self.engine.register_intent_parser(intent)

    def detach_skill(self, skill_id):
        if False:
            while True:
                i = 10
        'Remove all intents for skill.\n\n        Args:\n            skill_id (str): skill to process\n        '
        with self.lock:
            skill_parsers = [p.name for p in self.engine.intent_parsers if p.name.startswith(skill_id)]
            self.engine.drop_intent_parser(skill_parsers)
            self._detach_skill_keywords(skill_id)
            self._detach_skill_regexes(skill_id)

    def _detach_skill_keywords(self, skill_id):
        if False:
            for i in range(10):
                print('nop')
        'Detach all keywords registered with a particular skill.\n\n        Arguments:\n            skill_id (str): skill identifier\n        '
        skill_id = _entity_skill_id(skill_id)

        def match_skill_entities(data):
            if False:
                print('Hello World!')
            return data and data[1].startswith(skill_id)
        self.engine.drop_entity(match_func=match_skill_entities)

    def _detach_skill_regexes(self, skill_id):
        if False:
            print('Hello World!')
        'Detach all regexes registered with a particular skill.\n\n        Arguments:\n            skill_id (str): skill identifier\n        '
        skill_id = _entity_skill_id(skill_id)

        def match_skill_regexes(regexp):
            if False:
                i = 10
                return i + 15
            return any([r.startswith(skill_id) for r in regexp.groupindex.keys()])
        self.engine.drop_regex_entity(match_func=match_skill_regexes)

    def detach_intent(self, intent_name):
        if False:
            return 10
        'Detatch a single intent\n\n        Args:\n            intent_name (str): Identifier for intent to remove.\n        '
        new_parsers = [p for p in self.engine.intent_parsers if p.name != intent_name]
        self.engine.intent_parsers = new_parsers