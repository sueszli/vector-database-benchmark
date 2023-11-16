from enum import IntEnum
from abc import ABC, abstractmethod
from .mycroft_skill import MycroftSkill
from mycroft.util.file_utils import resolve_resource_file

class CQSMatchLevel(IntEnum):
    EXACT = 1
    CATEGORY = 2
    GENERAL = 3
CQSVisualMatchLevel = IntEnum('CQSVisualMatchLevel', [e.name for e in CQSMatchLevel])

def is_CQSVisualMatchLevel(match_level):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(match_level, type(CQSVisualMatchLevel.EXACT))
'these are for the confidence calculation'
TOPIC_MATCH_RELEVANCE = 5
RELEVANCE_MULTIPLIER = 2
MAX_ANSWER_LEN_FOR_CONFIDENCE = 50
WORD_COUNT_DIVISOR = 100

class CommonQuerySkill(MycroftSkill, ABC):
    """Question answering skills should be based on this class.

    The skill author needs to implement `CQS_match_query_phrase` returning an
    answer and can optionally implement `CQS_action` to perform additional
    actions if the skill's answer is selected.

    This class works in conjunction with skill-query which collects
    answers from several skills presenting the best one available.
    """

    def __init__(self, name=None, bus=None):
        if False:
            i = 10
            return i + 15
        super().__init__(name, bus)
        noise_words_filepath = 'text/%s/noise_words.list' % (self.lang,)
        noise_words_filename = resolve_resource_file(noise_words_filepath)
        self.translated_noise_words = []
        try:
            if noise_words_filename:
                with open(noise_words_filename) as f:
                    read_noise_words = f.read().strip()
                self.translated_noise_words = read_noise_words.split()
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            self.log.warning(f'Missing noise_words.list file in res/text/{self.lang}')
        self.level_confidence = {CQSMatchLevel.EXACT: 0.9, CQSMatchLevel.CATEGORY: 0.6, CQSMatchLevel.GENERAL: 0.5}

    def bind(self, bus):
        if False:
            print('Hello World!')
        'Overrides the default bind method of MycroftSkill.\n\n        This registers messagebus handlers for the skill during startup\n        but is nothing the skill author needs to consider.\n        '
        if bus:
            super().bind(bus)
            self.add_event('question:query', self.__handle_question_query)
            self.add_event('question:action', self.__handle_query_action)

    def __handle_question_query(self, message):
        if False:
            for i in range(10):
                print('nop')
        search_phrase = message.data['phrase']
        self.bus.emit(message.response({'phrase': search_phrase, 'skill_id': self.skill_id, 'searching': True}))
        result = self.CQS_match_query_phrase(search_phrase)
        if result:
            match = result[0]
            level = result[1]
            answer = result[2]
            callback = result[3] if len(result) > 3 else None
            confidence = self.__calc_confidence(match, search_phrase, level, answer)
            self.bus.emit(message.response({'phrase': search_phrase, 'skill_id': self.skill_id, 'answer': answer, 'callback_data': callback, 'conf': confidence}))
        else:
            self.bus.emit(message.response({'phrase': search_phrase, 'skill_id': self.skill_id, 'searching': False}))

    def remove_noise(self, phrase):
        if False:
            i = 10
            return i + 15
        'remove noise to produce essence of question'
        phrase = ' ' + phrase + ' '
        for word in self.translated_noise_words:
            mtch = ' ' + word + ' '
            if phrase.find(mtch) > -1:
                phrase = phrase.replace(mtch, ' ')
        phrase = ' '.join(phrase.split())
        return phrase.strip()

    def __calc_confidence(self, match, phrase, level, answer):
        if False:
            for i in range(10):
                print('nop')
        consumed_pct = len(match.split()) / len(phrase.split())
        if consumed_pct > 1.0:
            consumed_pct = 1.0
        consumed_pct /= 10
        num_sentences = float(float(len(answer.split('.'))) / float(10))
        bonus = 0.0
        if is_CQSVisualMatchLevel(level) and self.gui.connected:
            bonus = 0.1
        topic = self.remove_noise(match)
        answer = answer.lower()
        matches = 0
        for word in topic.split(' '):
            if answer.find(word) > -1:
                matches += TOPIC_MATCH_RELEVANCE
        answer_size = len(answer.split(' '))
        answer_size = min(MAX_ANSWER_LEN_FOR_CONFIDENCE, answer_size)
        relevance = 0.0
        if answer_size > 0:
            relevance = float(float(matches) / float(answer_size))
        relevance = relevance * RELEVANCE_MULTIPLIER
        wc_mod = float(float(answer_size) / float(WORD_COUNT_DIVISOR)) * 2
        confidence = self.level_confidence[level] + consumed_pct + bonus + num_sentences + relevance + wc_mod
        return confidence

    def __handle_query_action(self, message):
        if False:
            while True:
                i = 10
        'Message handler for question:action.\n\n        Extracts phrase and data from message forward this to the skills\n        CQS_action method.\n        '
        if message.data['skill_id'] != self.skill_id:
            return
        phrase = message.data['phrase']
        data = message.data.get('callback_data')
        self.CQS_action(phrase, data)

    @abstractmethod
    def CQS_match_query_phrase(self, phrase):
        if False:
            i = 10
            return i + 15
        'Analyze phrase to see if it is a play-able phrase with this skill.\n\n        Needs to be implemented by the skill.\n\n        Args:\n            phrase (str): User phrase, "What is an aardwark"\n\n        Returns:\n            (match, CQSMatchLevel[, callback_data]) or None: Tuple containing\n                 a string with the appropriate matching phrase, the PlayMatch\n                 type, and optionally data to return in the callback if the\n                 match is selected.\n        '
        return None

    def CQS_action(self, phrase, data):
        if False:
            i = 10
            return i + 15
        'Take additional action IF the skill is selected.\n\n        The speech is handled by the common query but if the chosen skill\n        wants to display media, set a context or prepare for sending\n        information info over e-mail this can be implemented here.\n\n        Args:\n            phrase (str): User phrase uttered after "Play", e.g. "some music"\n            data (dict): Callback data specified in match_query_phrase()\n        '
        pass