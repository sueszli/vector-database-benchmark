"""
Module containing the UniversalDetector detector class, which is the primary
class a user of ``chardet`` should use.

:author: Mark Pilgrim (initial port to Python)
:author: Shy Shalom (original C code)
:author: Dan Blanchard (major refactoring for 3.0)
:author: Ian Cordasco
"""
import codecs
import logging
import re
from .charsetgroupprober import CharSetGroupProber
from .enums import InputState, LanguageFilter, ProbingState
from .escprober import EscCharSetProber
from .latin1prober import Latin1Prober
from .mbcsgroupprober import MBCSGroupProber
from .sbcsgroupprober import SBCSGroupProber

class UniversalDetector(object):
    """
    The ``UniversalDetector`` class underlies the ``chardet.detect`` function
    and coordinates all of the different charset probers.

    To get a ``dict`` containing an encoding and its confidence, you can simply
    run:

    .. code::

            u = UniversalDetector()
            u.feed(some_bytes)
            u.close()
            detected = u.result

    """
    MINIMUM_THRESHOLD = 0.2
    HIGH_BYTE_DETECTOR = re.compile(b'[\x80-\xff]')
    ESC_DETECTOR = re.compile(b'(\x1b|~{)')
    WIN_BYTE_DETECTOR = re.compile(b'[\x80-\x9f]')
    ISO_WIN_MAP = {'iso-8859-1': 'Windows-1252', 'iso-8859-2': 'Windows-1250', 'iso-8859-5': 'Windows-1251', 'iso-8859-6': 'Windows-1256', 'iso-8859-7': 'Windows-1253', 'iso-8859-8': 'Windows-1255', 'iso-8859-9': 'Windows-1254', 'iso-8859-13': 'Windows-1257'}

    def __init__(self, lang_filter=LanguageFilter.ALL):
        if False:
            return 10
        self._esc_charset_prober = None
        self._charset_probers = []
        self.result = None
        self.done = None
        self._got_data = None
        self._input_state = None
        self._last_char = None
        self.lang_filter = lang_filter
        self.logger = logging.getLogger(__name__)
        self._has_win_bytes = None
        self.reset()

    def reset(self):
        if False:
            while True:
                i = 10
        '\n        Reset the UniversalDetector and all of its probers back to their\n        initial states.  This is called by ``__init__``, so you only need to\n        call this directly in between analyses of different documents.\n        '
        self.result = {'encoding': None, 'confidence': 0.0, 'language': None}
        self.done = False
        self._got_data = False
        self._has_win_bytes = False
        self._input_state = InputState.PURE_ASCII
        self._last_char = b''
        if self._esc_charset_prober:
            self._esc_charset_prober.reset()
        for prober in self._charset_probers:
            prober.reset()

    def feed(self, byte_str):
        if False:
            i = 10
            return i + 15
        "\n        Takes a chunk of a document and feeds it through all of the relevant\n        charset probers.\n\n        After calling ``feed``, you can check the value of the ``done``\n        attribute to see if you need to continue feeding the\n        ``UniversalDetector`` more data, or if it has made a prediction\n        (in the ``result`` attribute).\n\n        .. note::\n           You should always call ``close`` when you're done feeding in your\n           document if ``done`` is not already ``True``.\n        "
        if self.done:
            return
        if not len(byte_str):
            return
        if not isinstance(byte_str, bytearray):
            byte_str = bytearray(byte_str)
        if not self._got_data:
            if byte_str.startswith(codecs.BOM_UTF8):
                self.result = {'encoding': 'UTF-8-SIG', 'confidence': 1.0, 'language': ''}
            elif byte_str.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
                self.result = {'encoding': 'UTF-32', 'confidence': 1.0, 'language': ''}
            elif byte_str.startswith(b'\xfe\xff\x00\x00'):
                self.result = {'encoding': 'X-ISO-10646-UCS-4-3412', 'confidence': 1.0, 'language': ''}
            elif byte_str.startswith(b'\x00\x00\xff\xfe'):
                self.result = {'encoding': 'X-ISO-10646-UCS-4-2143', 'confidence': 1.0, 'language': ''}
            elif byte_str.startswith((codecs.BOM_LE, codecs.BOM_BE)):
                self.result = {'encoding': 'UTF-16', 'confidence': 1.0, 'language': ''}
            self._got_data = True
            if self.result['encoding'] is not None:
                self.done = True
                return
        if self._input_state == InputState.PURE_ASCII:
            if self.HIGH_BYTE_DETECTOR.search(byte_str):
                self._input_state = InputState.HIGH_BYTE
            elif self._input_state == InputState.PURE_ASCII and self.ESC_DETECTOR.search(self._last_char + byte_str):
                self._input_state = InputState.ESC_ASCII
        self._last_char = byte_str[-1:]
        if self._input_state == InputState.ESC_ASCII:
            if not self._esc_charset_prober:
                self._esc_charset_prober = EscCharSetProber(self.lang_filter)
            if self._esc_charset_prober.feed(byte_str) == ProbingState.FOUND_IT:
                self.result = {'encoding': self._esc_charset_prober.charset_name, 'confidence': self._esc_charset_prober.get_confidence(), 'language': self._esc_charset_prober.language}
                self.done = True
        elif self._input_state == InputState.HIGH_BYTE:
            if not self._charset_probers:
                self._charset_probers = [MBCSGroupProber(self.lang_filter)]
                if self.lang_filter & LanguageFilter.NON_CJK:
                    self._charset_probers.append(SBCSGroupProber())
                self._charset_probers.append(Latin1Prober())
            for prober in self._charset_probers:
                if prober.feed(byte_str) == ProbingState.FOUND_IT:
                    self.result = {'encoding': prober.charset_name, 'confidence': prober.get_confidence(), 'language': prober.language}
                    self.done = True
                    break
            if self.WIN_BYTE_DETECTOR.search(byte_str):
                self._has_win_bytes = True

    def close(self):
        if False:
            return 10
        '\n        Stop analyzing the current document and come up with a final\n        prediction.\n\n        :returns:  The ``result`` attribute, a ``dict`` with the keys\n                   `encoding`, `confidence`, and `language`.\n        '
        if self.done:
            return self.result
        self.done = True
        if not self._got_data:
            self.logger.debug('no data received!')
        elif self._input_state == InputState.PURE_ASCII:
            self.result = {'encoding': 'ascii', 'confidence': 1.0, 'language': ''}
        elif self._input_state == InputState.HIGH_BYTE:
            prober_confidence = None
            max_prober_confidence = 0.0
            max_prober = None
            for prober in self._charset_probers:
                if not prober:
                    continue
                prober_confidence = prober.get_confidence()
                if prober_confidence > max_prober_confidence:
                    max_prober_confidence = prober_confidence
                    max_prober = prober
            if max_prober and max_prober_confidence > self.MINIMUM_THRESHOLD:
                charset_name = max_prober.charset_name
                lower_charset_name = max_prober.charset_name.lower()
                confidence = max_prober.get_confidence()
                if lower_charset_name.startswith('iso-8859'):
                    if self._has_win_bytes:
                        charset_name = self.ISO_WIN_MAP.get(lower_charset_name, charset_name)
                self.result = {'encoding': charset_name, 'confidence': confidence, 'language': max_prober.language}
        if self.logger.getEffectiveLevel() == logging.DEBUG:
            if self.result['encoding'] is None:
                self.logger.debug('no probers hit minimum threshold')
                for group_prober in self._charset_probers:
                    if not group_prober:
                        continue
                    if isinstance(group_prober, CharSetGroupProber):
                        for prober in group_prober.probers:
                            self.logger.debug('%s %s confidence = %s', prober.charset_name, prober.language, prober.get_confidence())
                    else:
                        self.logger.debug('%s %s confidence = %s', prober.charset_name, prober.language, prober.get_confidence())
        return self.result