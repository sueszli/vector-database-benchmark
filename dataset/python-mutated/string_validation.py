import unicodedata
from typing import Optional
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.models import Stream
unicode_non_chars = {chr(x) for r in [range(64976, 65008), range(65534, 1114112, 65536), range(65535, 1114112, 65536)] for x in r}

def is_character_printable(char: str) -> bool:
    if False:
        i = 10
        return i + 15
    unicode_category = unicodedata.category(char)
    if unicode_category in ['Cc', 'Cs'] or char in unicode_non_chars:
        return False
    return True

def check_string_is_printable(var: str) -> Optional[int]:
    if False:
        for i in range(10):
            print('nop')
    for (i, char) in enumerate(var):
        if not is_character_printable(char):
            return i + 1
    return None

def check_stream_name(stream_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    if stream_name.strip() == '':
        raise JsonableError(_("Stream name can't be empty!"))
    if len(stream_name) > Stream.MAX_NAME_LENGTH:
        raise JsonableError(_('Stream name too long (limit: {max_length} characters).').format(max_length=Stream.MAX_NAME_LENGTH))
    invalid_character_pos = check_string_is_printable(stream_name)
    if invalid_character_pos is not None:
        raise JsonableError(_('Invalid character in stream name, at position {position}!').format(position=invalid_character_pos))

def check_stream_topic(topic: str) -> None:
    if False:
        i = 10
        return i + 15
    if topic.strip() == '':
        raise JsonableError(_("Topic can't be empty!"))
    invalid_character_pos = check_string_is_printable(topic)
    if invalid_character_pos is not None:
        raise JsonableError(_('Invalid character in topic, at position {position}!').format(position=invalid_character_pos))