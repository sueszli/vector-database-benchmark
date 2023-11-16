import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
ADDITIONAL_DIACRITICS = {'œ': 'oe', 'Œ': 'OE', 'ø': 'o', 'Ø': 'O', 'æ': 'ae', 'Æ': 'AE', 'ß': 'ss', 'ẞ': 'SS', 'đ': 'd', 'Đ': 'D', 'ð': 'd', 'Ð': 'D', 'þ': 'th', 'Þ': 'th', 'ł': 'l', 'Ł': 'L'}

def remove_symbols_and_diacritics(s: str, keep=''):
    if False:
        return 10
    "\n    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some\n    manual mappings)\n    "

    def replace_character(char):
        if False:
            print('Hello World!')
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]
        elif unicodedata.category(char) == 'Mn':
            return ''
        elif unicodedata.category(char)[0] in 'MSP':
            return ' '
        return char
    return ''.join((replace_character(c) for c in unicodedata.normalize('NFKD', s)))

def remove_symbols(s: str):
    if False:
        print('Hello World!')
    '\n    Replace any other markers, symbols, punctuations with a space, keeping diacritics\n    '
    return ''.join((' ' if unicodedata.category(c)[0] in 'MSP' else c for c in unicodedata.normalize('NFKC', s)))

class BasicTextNormalizer:

    def __init__(self, remove_diacritics: bool=False, split_letters: bool=False):
        if False:
            for i in range(10):
                print('nop')
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        if False:
            print('Hello World!')
        s = s.lower()
        s = re.sub('[<\\[][^>\\]]*[>\\]]', '', s)
        s = re.sub('\\(([^)]+?)\\)', '', s)
        s = self.clean(s).lower()
        if self.split_letters:
            s = ' '.join(regex.findall('\\X', s, regex.U))
        s = re.sub('\\s+', ' ', s)
        return s

class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, while handling:

    - remove any commas
    - keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.
    - spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`
    - spell out `one` and `ones`
    - interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.zeros = {'o', 'oh', 'zero'}
        self.ones = {name: i for (i, name) in enumerate(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'], start=1)}
        self.ones_plural = {'sixes' if name == 'six' else name + 's': (value, 's') for (name, value) in self.ones.items()}
        self.ones_ordinal = {'zeroth': (0, 'th'), 'first': (1, 'st'), 'second': (2, 'nd'), 'third': (3, 'rd'), 'fifth': (5, 'th'), 'twelfth': (12, 'th'), **{name + ('h' if name.endswith('t') else 'th'): (value, 'th') for (name, value) in self.ones.items() if value > 3 and value != 5 and (value != 12)}}
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}
        self.tens = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
        self.tens_plural = {name.replace('y', 'ies'): (value, 's') for (name, value) in self.tens.items()}
        self.tens_ordinal = {name.replace('y', 'ieth'): (value, 'th') for (name, value) in self.tens.items()}
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}
        self.multipliers = {'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000, 'trillion': 1000000000000, 'quadrillion': 1000000000000000, 'quintillion': 1000000000000000000, 'sextillion': 1000000000000000000000, 'septillion': 1000000000000000000000000, 'octillion': 1000000000000000000000000000, 'nonillion': 1000000000000000000000000000000, 'decillion': 1000000000000000000000000000000000}
        self.multipliers_plural = {name + 's': (value, 's') for (name, value) in self.multipliers.items()}
        self.multipliers_ordinal = {name + 'th': (value, 'th') for (name, value) in self.multipliers.items()}
        self.multipliers_suffixed = {**self.multipliers_plural, **self.multipliers_ordinal}
        self.decimals = {*self.ones, *self.tens, *self.zeros}
        self.preceding_prefixers = {'minus': '-', 'negative': '-', 'plus': '+', 'positive': '+'}
        self.following_prefixers = {'pound': '£', 'pounds': '£', 'euro': '€', 'euros': '€', 'dollar': '$', 'dollars': '$', 'cent': '¢', 'cents': '¢'}
        self.prefixes = set(list(self.preceding_prefixers.values()) + list(self.following_prefixers.values()))
        self.suffixers = {'per': {'cent': '%'}, 'percent': '%'}
        self.specials = {'and', 'double', 'triple', 'point'}
        self.words = {key for mapping in [self.zeros, self.ones, self.ones_suffixed, self.tens, self.tens_suffixed, self.multipliers, self.multipliers_suffixed, self.preceding_prefixers, self.following_prefixers, self.suffixers, self.specials] for key in mapping}
        self.literal_words = {'one', 'ones'}

    def process_words(self, words: List[str]) -> Iterator[str]:
        if False:
            print('Hello World!')
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            if False:
                i = 10
                return i + 15
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            if False:
                return 10
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result
        if len(words) == 0:
            return
        for (i, current) in enumerate(words):
            prev = words[i - 1] if i != 0 else None
            next = words[i + 1] if i != len(words) - 1 else None
            if skip:
                skip = False
                continue
            next_is_numeric = next is not None and re.match('^\\d+(\\.\\d+)?$', next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match('^\\d+(\\.\\d+)?$', current_without_prefix):
                f = to_fraction(current_without_prefix)
                if f is None:
                    raise ValueError('Converting the fraction failed')
                if value is not None:
                    if isinstance(value, str) and value.endswith('.'):
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)
                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator
                else:
                    value = current_without_prefix
            elif current not in self.words:
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or '') + '0'
            elif current in self.ones:
                ones = self.ones[current]
                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                elif value % 100 == 0:
                    value += ones
                else:
                    value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                (ones, suffix) = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif value % 100 == 0:
                    yield output(str(value + ones) + suffix)
                else:
                    yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                elif value % 100 == 0:
                    value += tens
                else:
                    value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                (tens, suffix) = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                elif value % 100 == 0:
                    yield output(str(value + tens) + suffix)
                else:
                    yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                (multiplier, suffix) = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                if value is not None:
                    yield output(value)
                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and (not next_is_numeric):
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == 'and':
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == 'double' or current == 'triple':
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == 'double' else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or '') + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == 'point':
                    if next in self.decimals or next_is_numeric:
                        value = str(value or '') + '.'
                else:
                    raise ValueError(f'Unexpected token: {current}')
            else:
                raise ValueError(f'Unexpected token: {current}')
        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        if False:
            i = 10
            return i + 15
        results = []
        segments = re.split('\\band\\s+a\\s+half\\b', s)
        for (i, segment) in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append('point five')
                else:
                    results.append('and a half')
        s = ' '.join(results)
        s = re.sub('([a-z])([0-9])', '\\1 \\2', s)
        s = re.sub('([0-9])([a-z])', '\\1 \\2', s)
        s = re.sub('([0-9])\\s+(st|nd|rd|th|s)\\b', '\\1\\2', s)
        return s

    def postprocess(self, s: str):
        if False:
            for i in range(10):
                print('nop')

        def combine_cents(m: Match):
            if False:
                for i in range(10):
                    print('nop')
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f'{currency}{integer}.{cents:02d}'
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            if False:
                print('Hello World!')
            try:
                return f'¢{int(m.group(1))}'
            except ValueError:
                return m.string
        s = re.sub('([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\\b', combine_cents, s)
        s = re.sub('[€£$]0.([0-9]{1,2})\\b', extract_cents, s)
        s = re.sub('\\b1(s?)\\b', 'one\\1', s)
        return s

    def __call__(self, s: str):
        if False:
            return 10
        s = self.preprocess(s)
        s = ' '.join((word for word in self.process_words(s.split()) if word is not None))
        s = self.postprocess(s)
        return s

class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self, english_spelling_mapping):
        if False:
            return 10
        self.mapping = english_spelling_mapping

    def __call__(self, s: str):
        if False:
            for i in range(10):
                print('nop')
        return ' '.join((self.mapping.get(word, word) for word in s.split()))

class EnglishTextNormalizer:

    def __init__(self, english_spelling_mapping):
        if False:
            while True:
                i = 10
        self.ignore_patterns = '\\b(hmm|mm|mhm|mmm|uh|um)\\b'
        self.replacers = {"\\bwon't\\b": 'will not', "\\bcan't\\b": 'can not', "\\blet's\\b": 'let us', "\\bain't\\b": 'aint', "\\by'all\\b": 'you all', '\\bwanna\\b': 'want to', '\\bgotta\\b': 'got to', '\\bgonna\\b': 'going to', "\\bi'ma\\b": 'i am going to', '\\bimma\\b': 'i am going to', '\\bwoulda\\b': 'would have', '\\bcoulda\\b': 'could have', '\\bshoulda\\b': 'should have', "\\bma'am\\b": 'madam', '\\bmr\\b': 'mister ', '\\bmrs\\b': 'missus ', '\\bst\\b': 'saint ', '\\bdr\\b': 'doctor ', '\\bprof\\b': 'professor ', '\\bcapt\\b': 'captain ', '\\bgov\\b': 'governor ', '\\bald\\b': 'alderman ', '\\bgen\\b': 'general ', '\\bsen\\b': 'senator ', '\\brep\\b': 'representative ', '\\bpres\\b': 'president ', '\\brev\\b': 'reverend ', '\\bhon\\b': 'honorable ', '\\basst\\b': 'assistant ', '\\bassoc\\b': 'associate ', '\\blt\\b': 'lieutenant ', '\\bcol\\b': 'colonel ', '\\bjr\\b': 'junior ', '\\bsr\\b': 'senior ', '\\besq\\b': 'esquire ', "'d been\\b": ' had been', "'s been\\b": ' has been', "'d gone\\b": ' had gone', "'s gone\\b": ' has gone', "'d done\\b": ' had done', "'s got\\b": ' has got', "n't\\b": ' not', "'re\\b": ' are', "'s\\b": ' is', "'d\\b": ' would', "'ll\\b": ' will', "'t\\b": ' not', "'ve\\b": ' have', "'m\\b": ' am'}
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)

    def __call__(self, s: str):
        if False:
            for i in range(10):
                print('nop')
        s = s.lower()
        s = re.sub('[<\\[][^>\\]]*[>\\]]', '', s)
        s = re.sub('\\(([^)]+?)\\)', '', s)
        s = re.sub(self.ignore_patterns, '', s)
        s = re.sub("\\s+'", "'", s)
        for (pattern, replacement) in self.replacers.items():
            s = re.sub(pattern, replacement, s)
        s = re.sub('(\\d),(\\d)', '\\1\\2', s)
        s = re.sub('\\.([^0-9]|$)', ' \\1', s)
        s = remove_symbols_and_diacritics(s, keep='.%$¢€£')
        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)
        s = re.sub('[.$¢€£]([^0-9])', ' \\1', s)
        s = re.sub('([^0-9])%', '\\1 ', s)
        s = re.sub('\\s+', ' ', s)
        return s