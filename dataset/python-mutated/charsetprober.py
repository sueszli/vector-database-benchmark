import logging
import re
from .enums import ProbingState

class CharSetProber(object):
    SHORTCUT_THRESHOLD = 0.95

    def __init__(self, lang_filter=None):
        if False:
            while True:
                i = 10
        self._state = None
        self.lang_filter = lang_filter
        self.logger = logging.getLogger(__name__)

    def reset(self):
        if False:
            i = 10
            return i + 15
        self._state = ProbingState.DETECTING

    @property
    def charset_name(self):
        if False:
            i = 10
            return i + 15
        return None

    def feed(self, buf):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def state(self):
        if False:
            i = 10
            return i + 15
        return self._state

    def get_confidence(self):
        if False:
            while True:
                i = 10
        return 0.0

    @staticmethod
    def filter_high_byte_only(buf):
        if False:
            i = 10
            return i + 15
        buf = re.sub(b'([\x00-\x7f])+', b' ', buf)
        return buf

    @staticmethod
    def filter_international_words(buf):
        if False:
            i = 10
            return i + 15
        '\n        We define three types of bytes:\n        alphabet: english alphabets [a-zA-Z]\n        international: international characters [\x80-ÿ]\n        marker: everything else [^a-zA-Z\x80-ÿ]\n\n        The input buffer can be thought to contain a series of words delimited\n        by markers. This function works to filter all words that contain at\n        least one international character. All contiguous sequences of markers\n        are replaced by a single space ascii character.\n\n        This filter applies to all scripts which do not use English characters.\n        '
        filtered = bytearray()
        words = re.findall(b'[a-zA-Z]*[\x80-\xff]+[a-zA-Z]*[^a-zA-Z\x80-\xff]?', buf)
        for word in words:
            filtered.extend(word[:-1])
            last_char = word[-1:]
            if not last_char.isalpha() and last_char < b'\x80':
                last_char = b' '
            filtered.extend(last_char)
        return filtered

    @staticmethod
    def filter_with_english_letters(buf):
        if False:
            return 10
        '\n        Returns a copy of ``buf`` that retains only the sequences of English\n        alphabet and high byte characters that are not between <> characters.\n        Also retains English alphabet and high byte characters immediately\n        before occurrences of >.\n\n        This filter can be applied to all scripts which contain both English\n        characters and extended ASCII characters, but is currently only used by\n        ``Latin1Prober``.\n        '
        filtered = bytearray()
        in_tag = False
        prev = 0
        for curr in range(len(buf)):
            buf_char = buf[curr:curr + 1]
            if buf_char == b'>':
                in_tag = False
            elif buf_char == b'<':
                in_tag = True
            if buf_char < b'\x80' and (not buf_char.isalpha()):
                if curr > prev and (not in_tag):
                    filtered.extend(buf[prev:curr])
                    filtered.extend(b' ')
                prev = curr + 1
        if not in_tag:
            filtered.extend(buf[prev:])
        return filtered