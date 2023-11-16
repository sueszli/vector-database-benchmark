"""Intent service for Mycroft's fallback system."""
from collections import namedtuple
from .base import IntentMatch
FallbackRange = namedtuple('FallbackRange', ['start', 'stop'])

class FallbackService:
    """Intent Service handling fallback skills."""

    def __init__(self, bus):
        if False:
            i = 10
            return i + 15
        self.bus = bus

    def _fallback_range(self, utterances, lang, message, fb_range):
        if False:
            print('Hello World!')
        'Send fallback request for a specified priority range.\n\n        Args:\n            utterances (list): List of tuples,\n                               utterances and normalized version\n            lang (str): Langauge code\n            message: Message for session context\n            fb_range (FallbackRange): fallback order start and stop.\n\n        Returns:\n            IntentMatch or None\n        '
        msg = message.reply('mycroft.skills.fallback', data={'utterance': utterances[0][0], 'lang': lang, 'fallback_range': (fb_range.start, fb_range.stop)})
        response = self.bus.wait_for_response(msg, timeout=10)
        if response and response.data['handled']:
            ret = IntentMatch('Fallback', None, {}, None)
        else:
            ret = None
        return ret

    def high_prio(self, utterances, lang, message):
        if False:
            while True:
                i = 10
        'Pre-padatious fallbacks.'
        return self._fallback_range(utterances, lang, message, FallbackRange(0, 5))

    def medium_prio(self, utterances, lang, message):
        if False:
            while True:
                i = 10
        'General fallbacks.'
        return self._fallback_range(utterances, lang, message, FallbackRange(5, 90))

    def low_prio(self, utterances, lang, message):
        if False:
            i = 10
            return i + 15
        'Low prio fallbacks with general matching such as chat-bot.'
        return self._fallback_range(utterances, lang, message, FallbackRange(90, 101))