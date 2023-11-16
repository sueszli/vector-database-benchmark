import collections
import re
from enum import Enum
import six
_DEF_PUNCS = ';:,.!?¡¿—…"«»“”'
_PUNC_IDX = collections.namedtuple('_punc_index', ['punc', 'position'])

class PuncPosition(Enum):
    """Enum for the punctuations positions"""
    BEGIN = 0
    END = 1
    MIDDLE = 2
    ALONE = 3

class Punctuation:
    """Handle punctuations in text.

    Just strip punctuations from text or strip and restore them later.

    Args:
        puncs (str): The punctuations to be processed. Defaults to `_DEF_PUNCS`.

    Example:
        >>> punc = Punctuation()
        >>> punc.strip("This is. example !")
        'This is example'

        >>> text_striped, punc_map = punc.strip_to_restore("This is. example !")
        >>> ' '.join(text_striped)
        'This is example'

        >>> text_restored = punc.restore(text_striped, punc_map)
        >>> text_restored[0]
        'This is. example !'
    """

    def __init__(self, puncs: str=_DEF_PUNCS):
        if False:
            print('Hello World!')
        self.puncs = puncs

    @staticmethod
    def default_puncs():
        if False:
            for i in range(10):
                print('nop')
        'Return default set of punctuations.'
        return _DEF_PUNCS

    @property
    def puncs(self):
        if False:
            while True:
                i = 10
        return self._puncs

    @puncs.setter
    def puncs(self, value):
        if False:
            return 10
        if not isinstance(value, six.string_types):
            raise ValueError('[!] Punctuations must be of type str.')
        self._puncs = ''.join(list(dict.fromkeys(list(value))))
        self.puncs_regular_exp = re.compile(f'(\\s*[{re.escape(self._puncs)}]+\\s*)+')

    def strip(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Remove all the punctuations by replacing with `space`.\n\n        Args:\n            text (str): The text to be processed.\n\n        Example::\n\n            "This is. example !" -> "This is example "\n        '
        return re.sub(self.puncs_regular_exp, ' ', text).rstrip().lstrip()

    def strip_to_restore(self, text):
        if False:
            print('Hello World!')
        'Remove punctuations from text to restore them later.\n\n        Args:\n            text (str): The text to be processed.\n\n        Examples ::\n\n            "This is. example !" -> [["This is", "example"], [".", "!"]]\n\n        '
        (text, puncs) = self._strip_to_restore(text)
        return (text, puncs)

    def _strip_to_restore(self, text):
        if False:
            print('Hello World!')
        'Auxiliary method for Punctuation.preserve()'
        matches = list(re.finditer(self.puncs_regular_exp, text))
        if not matches:
            return ([text], [])
        if len(matches) == 1 and matches[0].group() == text:
            return ([], [_PUNC_IDX(text, PuncPosition.ALONE)])
        puncs = []
        for match in matches:
            position = PuncPosition.MIDDLE
            if match == matches[0] and text.startswith(match.group()):
                position = PuncPosition.BEGIN
            elif match == matches[-1] and text.endswith(match.group()):
                position = PuncPosition.END
            puncs.append(_PUNC_IDX(match.group(), position))
        splitted_text = []
        for (idx, punc) in enumerate(puncs):
            split = text.split(punc.punc)
            (prefix, suffix) = (split[0], punc.punc.join(split[1:]))
            splitted_text.append(prefix)
            if idx == len(puncs) - 1 and len(suffix) > 0:
                splitted_text.append(suffix)
            text = suffix
        return (splitted_text, puncs)

    @classmethod
    def restore(cls, text, puncs):
        if False:
            return 10
        'Restore punctuation in a text.\n\n        Args:\n            text (str): The text to be processed.\n            puncs (List[str]): The list of punctuations map to be used for restoring.\n\n        Examples ::\n\n            [\'This is\', \'example\'], [\'.\', \'!\'] -> "This is. example!"\n\n        '
        return cls._restore(text, puncs, 0)

    @classmethod
    def _restore(cls, text, puncs, num):
        if False:
            print('Hello World!')
        'Auxiliary method for Punctuation.restore()'
        if not puncs:
            return text
        if not text:
            return [''.join((m.punc for m in puncs))]
        current = puncs[0]
        if current.position == PuncPosition.BEGIN:
            return cls._restore([current.punc + text[0]] + text[1:], puncs[1:], num)
        if current.position == PuncPosition.END:
            return [text[0] + current.punc] + cls._restore(text[1:], puncs[1:], num + 1)
        if current.position == PuncPosition.ALONE:
            return [current.mark] + cls._restore(text, puncs[1:], num + 1)
        if len(text) == 1:
            return cls._restore([text[0] + current.punc], puncs[1:], num)
        return cls._restore([text[0] + current.punc + text[1]] + text[2:], puncs[1:], num)