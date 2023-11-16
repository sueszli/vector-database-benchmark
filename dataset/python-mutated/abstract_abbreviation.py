import re
import typing
from autokey.model.helpers import DEFAULT_WORDCHAR_REGEX, TriggerMode

class AbstractAbbreviation:
    """
    Abstract class encapsulating the common functionality of an abbreviation list
    """

    def __init__(self):
        if False:
            return 10
        self.abbreviations = []
        self.backspace = True
        self.ignoreCase = False
        self.immediate = False
        self.triggerInside = False
        self.set_word_chars(DEFAULT_WORDCHAR_REGEX)

    def get_serializable(self):
        if False:
            return 10
        d = {'abbreviations': self.abbreviations, 'backspace': self.backspace, 'ignoreCase': self.ignoreCase, 'immediate': self.immediate, 'triggerInside': self.triggerInside, 'wordChars': self.get_word_chars()}
        return d

    def load_from_serialized(self, data: dict):
        if False:
            for i in range(10):
                print('nop')
        if 'abbreviations' not in data:
            self.abbreviations = [data['abbreviation']]
        else:
            self.abbreviations = data['abbreviations']
        self.backspace = data['backspace']
        self.ignoreCase = data['ignoreCase']
        self.immediate = data['immediate']
        self.triggerInside = data['triggerInside']
        self.set_word_chars(data['wordChars'])

    def copy_abbreviation(self, abbr):
        if False:
            i = 10
            return i + 15
        self.abbreviations = abbr.abbreviations
        self.backspace = abbr.backspace
        self.ignoreCase = abbr.ignoreCase
        self.immediate = abbr.immediate
        self.triggerInside = abbr.triggerInside
        self.set_word_chars(abbr.get_word_chars())

    def set_word_chars(self, regex):
        if False:
            for i in range(10):
                print('nop')
        self.wordChars = re.compile(regex, re.UNICODE)

    def get_word_chars(self):
        if False:
            i = 10
            return i + 15
        return self.wordChars.pattern

    def add_abbreviation(self, abbr):
        if False:
            return 10
        if not isinstance(abbr, str):
            raise ValueError("Abbreviations must be strings. Cannot add abbreviation '{}', having type {}.".format(abbr, type(abbr)))
        self.abbreviations.append(abbr)
        if TriggerMode.ABBREVIATION not in self.modes:
            self.modes.append(TriggerMode.ABBREVIATION)

    def add_abbreviations(self, abbreviation_list: typing.Iterable[str]):
        if False:
            return 10
        if not isinstance(abbreviation_list, list):
            abbreviation_list = list(abbreviation_list)
        if not all((isinstance(abbr, str) for abbr in abbreviation_list)):
            raise ValueError('All added Abbreviations must be strings.')
        self.abbreviations += abbreviation_list
        if TriggerMode.ABBREVIATION not in self.modes:
            self.modes.append(TriggerMode.ABBREVIATION)

    def clear_abbreviations(self):
        if False:
            while True:
                i = 10
        self.abbreviations = []

    def get_abbreviations(self):
        if False:
            return 10
        if TriggerMode.ABBREVIATION not in self.modes:
            return ''
        elif len(self.abbreviations) == 1:
            return self.abbreviations[0]
        else:
            return '[' + ','.join(self.abbreviations) + ']'

    def _should_trigger_abbreviation(self, buffer):
        if False:
            i = 10
            return i + 15
        '\n        Checks whether, based on the settings for the abbreviation and the given input,\n        the abbreviation should trigger.\n\n        @param buffer Input buffer to be checked (as string)\n        '
        return any((self.__checkInput(buffer, abbr) for abbr in self.abbreviations))

    def _get_trigger_abbreviation(self, buffer):
        if False:
            print('Hello World!')
        for abbr in self.abbreviations:
            if self.__checkInput(buffer, abbr):
                return abbr
        return None

    def __checkInput(self, buffer, abbr):
        if False:
            print('Hello World!')
        (stringBefore, typedAbbr, stringAfter) = self._partition_input(buffer, abbr)
        if len(typedAbbr) > 0:
            if not self.immediate:
                if len(stringAfter) == 1:
                    if self.wordChars.match(stringAfter):
                        return False
                    elif len(stringAfter) > 1:
                        return False
                else:
                    return False
            elif len(stringAfter) > 0:
                return False
            if len(stringBefore) > 0 and (not re.match('(^\\s)', stringBefore[-1])) and (not self.triggerInside):
                return False
            return True
        return False

    def _partition_input(self, current_string: str, abbr: typing.Optional[str]) -> typing.Tuple[str, str, str]:
        if False:
            while True:
                i = 10
        '\n        Partition the input into text before, typed abbreviation (if it exists), and text after\n        '
        if abbr:
            if self.ignoreCase:
                (string_before, typed_abbreviation, string_after) = self._case_insensitive_rpartition(current_string, abbr)
                abbr_start_index = len(string_before)
                abbr_end_index = abbr_start_index + len(typed_abbreviation)
                typed_abbreviation = current_string[abbr_start_index:abbr_end_index]
            else:
                (string_before, typed_abbreviation, string_after) = current_string.rpartition(abbr)
            return (string_before, typed_abbreviation, string_after)
        else:
            return ('', current_string, '')

    @staticmethod
    def _case_insensitive_rpartition(input_string: str, separator: str) -> typing.Tuple[str, str, str]:
        if False:
            i = 10
            return i + 15
        'Same as str.rpartition(), except that the partitioning is done case insensitive.'
        lowered_input_string = input_string.lower()
        lowered_separator = separator.lower()
        try:
            split_index = lowered_input_string.rindex(lowered_separator)
        except ValueError:
            return ('', '', input_string)
        else:
            split_index_2 = split_index + len(separator)
            return (input_string[:split_index], input_string[split_index:split_index_2], input_string[split_index_2:])