import re
from mycroft.util.parse import normalize
from mycroft_bus_client.message import dig_for_message
import mycroft_bus_client

class Message(mycroft_bus_client.Message):
    """Mycroft specific Message class."""

    def utterance_remainder(self):
        if False:
            while True:
                i = 10
        '\n        For intents get the portion not consumed by Adapt.\n\n        For example: if they say \'Turn on the family room light\' and there are\n        entity matches for "turn on" and "light", then it will leave behind\n        " the family room " which is then normalized to "family room".\n\n        Returns:\n            str: Leftover words or None if not an utterance.\n        '
        utt = normalize(self.data.get('utterance', ''))
        if utt and '__tags__' in self.data:
            for token in self.data['__tags__']:
                utt = re.sub('\\b' + token.get('key', '') + '\\b', '', utt)
        return normalize(utt)