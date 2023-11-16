import re
import textwrap
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast
from streamlit.emojis import ALL_EMOJIS
from streamlit.errors import StreamlitAPIException
if TYPE_CHECKING:
    from streamlit.type_util import SupportsStr
ESCAPED_EMOJI = [re.escape(e) for e in sorted(ALL_EMOJIS, reverse=True)]
EMOJI_EXTRACTION_REGEX = re.compile(f"^({'|'.join(ESCAPED_EMOJI)})[_ -]*(.*)")

def decode_ascii(string: bytes) -> str:
    if False:
        i = 10
        return i + 15
    'Decodes a string as ascii.'
    return string.decode('ascii')

def clean_text(text: 'SupportsStr') -> str:
    if False:
        return 10
    'Convert an object to text, dedent it, and strip whitespace.'
    return textwrap.dedent(str(text)).strip()

def is_emoji(text: str) -> bool:
    if False:
        print('Hello World!')
    'Check if input string is a valid emoji.'
    return text.replace('️', '') in ALL_EMOJIS

def validate_emoji(maybe_emoji: Optional[str]) -> str:
    if False:
        i = 10
        return i + 15
    if maybe_emoji is None:
        return ''
    elif is_emoji(maybe_emoji):
        return maybe_emoji
    else:
        raise StreamlitAPIException(f'The value "{maybe_emoji}" is not a valid emoji. Shortcodes are not allowed, please use a single character instead.')

def extract_leading_emoji(text: str) -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    'Return a tuple containing the first emoji found in the given string and\n    the rest of the string (minus an optional separator between the two).\n    '
    re_match = re.search(EMOJI_EXTRACTION_REGEX, text)
    if re_match is None:
        return ('', text)
    re_match: re.Match[str] = cast(Any, re_match)
    return (re_match.group(1), re_match.group(2))

def escape_markdown(raw_string: str) -> str:
    if False:
        i = 10
        return i + 15
    'Returns a new string which escapes all markdown metacharacters.\n\n    Args\n    ----\n    raw_string : str\n        A string, possibly with markdown metacharacters, e.g. "1 * 2"\n\n    Returns\n    -------\n    A string with all metacharacters escaped.\n\n    Examples\n    --------\n    ::\n        escape_markdown("1 * 2") -> "1 \\\\* 2"\n    '
    metacharacters = ['\\', '*', '-', '=', '`', '!', '#', '|']
    result = raw_string
    for character in metacharacters:
        result = result.replace(character, '\\' + character)
    return result
TEXTCHARS = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(32, 256)) - {127})

def is_binary_string(inp):
    if False:
        for i in range(10):
            print('nop')
    'Guess if an input bytesarray can be encoded as a string.'
    return bool(inp.translate(None, TEXTCHARS))

def simplify_number(num: int) -> str:
    if False:
        print('Hello World!')
    'Simplifies number into Human readable format, returns str'
    num_converted = float('{:.2g}'.format(num))
    magnitude = 0
    while abs(num_converted) >= 1000:
        magnitude += 1
        num_converted /= 1000.0
    return '{}{}'.format('{:f}'.format(num_converted).rstrip('0').rstrip('.'), ['', 'k', 'm', 'b', 't'][magnitude])
_OBJ_MEM_ADDRESS = re.compile('^\\<[a-zA-Z_]+[a-zA-Z0-9<>._ ]* at 0x[0-9a-f]+\\>$')

def is_mem_address_str(string):
    if False:
        print('Hello World!')
    'Returns True if the string looks like <foo blarg at 0x15ee6f9a0>.'
    if _OBJ_MEM_ADDRESS.match(string):
        return True
    return False
_RE_CONTAINS_HTML = re.compile('(?:</[^<]+>)|(?:<[^<]+/>)')

def probably_contains_html_tags(s: str) -> bool:
    if False:
        return 10
    'Returns True if the given string contains what seem to be HTML tags.\n\n    Note that false positives/negatives are possible, so this function should not be\n    used in contexts where complete correctness is required.'
    return bool(_RE_CONTAINS_HTML.search(s))