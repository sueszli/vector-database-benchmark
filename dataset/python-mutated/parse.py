"""
The mycroft.util.parse module provides various parsing functions for things
like numbers, times, durations etc. It's intention is to convert naturally
expressed concepts into standard computer readable formats. Doing this also
enables localization.

It also provides some useful associated functions like basic fuzzy matching.

The module uses lingua-franca (https://github.com/mycroftai/lingua-franca) to
do most of the actual parsing. However methods may be wrapped specifically for
use in Mycroft Skills.
"""
from difflib import SequenceMatcher
from warnings import warn
from lingua_franca import get_default_loc
from lingua_franca.parse import extract_duration, extract_number, extract_numbers, fuzzy_match, get_gender, match_one, normalize
from lingua_franca.parse import extract_datetime as _extract_datetime
from .time import now_local
from .log import LOG

def _log_unsupported_language(language, supported_languages):
    if False:
        return 10
    '\n    Log a warning when a language is unsupported\n\n    Args:\n        language: str\n            The language that was supplied.\n        supported_languages: [str]\n            The list of supported languages.\n    '
    supported = ' '.join(supported_languages)
    LOG.warning('Language "{language}" not recognized! Please make sure your language is one of the following: {supported}.'.format(language=language, supported=supported))

def extract_datetime(text, anchorDate='DEFAULT', lang=None, default_time=None):
    if False:
        i = 10
        return i + 15
    'Extracts date and time information from a sentence.\n\n    Parses many of the common ways that humans express dates and times,\n    including relative dates like "5 days from today", "tomorrow\', and\n    "Tuesday".\n\n    Vague terminology are given arbitrary values, like:\n\n    * morning = 8 AM\n    * afternoon = 3 PM\n    * evening = 7 PM\n\n    If a time isn\'t supplied or implied, the function defaults to 12 AM\n\n    Args:\n        text (str): the text to be interpreted\n        anchorDate (:obj:`datetime`, optional): the date to be used for\n            relative dating (for example, what does "tomorrow" mean?).\n            Defaults to the current local date/time.\n        lang (str): the BCP-47 code for the language to use, None uses default\n        default_time (datetime.time): time to use if none was found in\n            the input string.\n\n    Returns:\n        [:obj:`datetime`, :obj:`str`]: \'datetime\' is the extracted date\n            as a datetime object in the user\'s local timezone.\n            \'leftover_string\' is the original phrase with all date and time\n            related keywords stripped out. See examples for further\n            clarification\n            Returns \'None\' if no date or time related text is found.\n    Examples:\n        >>> extract_datetime(\n        ... "What is the weather like the day after tomorrow?",\n        ... datetime(2017, 06, 30, 00, 00)\n        ... )\n        [datetime.datetime(2017, 7, 2, 0, 0), \'what is weather like\']\n        >>> extract_datetime(\n        ... "Set up an appointment 2 weeks from Sunday at 5 pm",\n        ... datetime(2016, 02, 19, 00, 00)\n        ... )\n        [datetime.datetime(2016, 3, 6, 17, 0), \'set up appointment\']\n        >>> extract_datetime(\n        ... "Set up an appointment",\n        ... datetime(2016, 02, 19, 00, 00)\n        ... )\n        None\n    '
    if anchorDate is None:
        warn(DeprecationWarning('extract_datetime(anchorDate=None) is deprecated. This parameter can be omitted.'))
    if anchorDate is None or anchorDate == 'DEFAULT':
        anchorDate = now_local()
    return _extract_datetime(text, anchorDate, lang or get_default_loc(), default_time)