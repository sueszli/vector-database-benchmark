"""This module contains various provisional APIs and strategies.

It is intended for internal use, to ease code reuse, and is not stable.
Point releases may move or break the contents at any time!

Internet strategies should conform to :rfc:`3986` or the authoritative
definitions it links to.  If not, report the bug!
"""
import string
from importlib import resources
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture import utils as cu
from hypothesis.strategies._internal.utils import defines_strategy
URL_SAFE_CHARACTERS = frozenset(string.ascii_letters + string.digits + "$-_.+!*'(),~")
FRAGMENT_SAFE_CHARACTERS = URL_SAFE_CHARACTERS | {'?', '/'}
try:
    traversable = resources.files('hypothesis.vendor') / 'tlds-alpha-by-domain.txt'
    (_comment, *_tlds) = traversable.read_text(encoding='utf-8').splitlines()
except (AttributeError, ValueError):
    (_comment, *_tlds) = resources.read_text('hypothesis.vendor', 'tlds-alpha-by-domain.txt', encoding='utf-8').splitlines()
assert _comment.startswith('#')
TOP_LEVEL_DOMAINS = ['COM', *sorted((d for d in _tlds if d != 'ARPA'), key=len)]

class DomainNameStrategy(st.SearchStrategy):

    @staticmethod
    def clean_inputs(minimum, maximum, value, variable_name):
        if False:
            return 10
        if value is None:
            value = maximum
        elif not isinstance(value, int):
            raise InvalidArgument(f'Expected integer but {variable_name} is a {type(value).__name__}')
        elif not minimum <= value <= maximum:
            raise InvalidArgument(f'Invalid value {minimum!r} < {variable_name}={value!r} < {maximum!r}')
        return value

    def __init__(self, max_length=None, max_element_length=None):
        if False:
            i = 10
            return i + 15
        '\n        A strategy for :rfc:`1035` fully qualified domain names.\n\n        The upper limit for max_length is 255 in accordance with :rfc:`1035#section-2.3.4`\n        The lower limit for max_length is 4, corresponding to a two letter domain\n        with a single letter subdomain.\n        The upper limit for max_element_length is 63 in accordance with :rfc:`1035#section-2.3.4`\n        The lower limit for max_element_length is 1 in accordance with :rfc:`1035#section-2.3.4`\n        '
        max_length = self.clean_inputs(4, 255, max_length, 'max_length')
        max_element_length = self.clean_inputs(1, 63, max_element_length, 'max_element_length')
        super().__init__()
        self.max_length = max_length
        self.max_element_length = max_element_length
        if self.max_element_length == 1:
            self.label_regex = '[a-zA-Z]'
        elif self.max_element_length == 2:
            self.label_regex = '[a-zA-Z][a-zA-Z0-9]?'
        else:
            maximum_center_character_pattern_repetitions = self.max_element_length - 2
            self.label_regex = '[a-zA-Z]([a-zA-Z0-9\\-]{0,%d}[a-zA-Z0-9])?' % (maximum_center_character_pattern_repetitions,)

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15
        domain = data.draw(st.sampled_from(TOP_LEVEL_DOMAINS).filter(lambda tld: len(tld) + 2 <= self.max_length).flatmap(lambda tld: st.tuples(*(st.sampled_from([c.lower(), c.upper()]) for c in tld)).map(''.join)))
        elem_st = st.from_regex(self.label_regex, fullmatch=True).filter(lambda label: len(label) < 4 or label[2:4] != '--')
        elements = cu.many(data, min_size=1, average_size=3, max_size=126)
        while elements.more():
            sub_domain = data.draw(elem_st)
            if len(domain) + len(sub_domain) >= self.max_length:
                data.stop_example(discard=True)
                break
            domain = sub_domain + '.' + domain
        return domain

@defines_strategy(force_reusable_values=True)
def domains(*, max_length: int=255, max_element_length: int=63) -> st.SearchStrategy[str]:
    if False:
        print('Hello World!')
    'Generate :rfc:`1035` compliant fully qualified domain names.'
    return DomainNameStrategy(max_length=max_length, max_element_length=max_element_length)
_url_fragments_strategy = st.lists(st.builds(lambda char, encode: f'%{ord(char):02X}' if encode or char not in FRAGMENT_SAFE_CHARACTERS else char, st.characters(min_codepoint=0, max_codepoint=255), st.booleans()), min_size=1).map(''.join).map('#{}'.format)

@defines_strategy(force_reusable_values=True)
def urls() -> st.SearchStrategy[str]:
    if False:
        for i in range(10):
            print('nop')
    'A strategy for :rfc:`3986`, generating http/https URLs.'

    def url_encode(s):
        if False:
            for i in range(10):
                print('nop')
        return ''.join((c if c in URL_SAFE_CHARACTERS else '%%%02X' % ord(c) for c in s))
    schemes = st.sampled_from(['http', 'https'])
    ports = st.integers(min_value=0, max_value=2 ** 16 - 1).map(':{}'.format)
    paths = st.lists(st.text(string.printable).map(url_encode)).map('/'.join)
    return st.builds('{}://{}{}/{}{}'.format, schemes, domains(), st.just('') | ports, paths, st.just('') | _url_fragments_strategy)