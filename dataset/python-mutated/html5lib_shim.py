"""
Shim module between Bleach and html5lib. This makes it easier to upgrade the
html5lib library without having to change a lot of code.
"""
import re
import string
import warnings
warnings.filterwarnings('ignore', message="html5lib's sanitizer is deprecated", category=DeprecationWarning, module='bleach._vendor.html5lib')
from bleach._vendor.html5lib import HTMLParser, getTreeWalker
from bleach._vendor.html5lib import constants
from bleach._vendor.html5lib.constants import namespaces, prefixes
from bleach._vendor.html5lib.constants import _ReparseException as ReparseException
from bleach._vendor.html5lib.filters.base import Filter
from bleach._vendor.html5lib.filters.sanitizer import allowed_protocols, allowed_css_properties, allowed_svg_properties, attr_val_is_uri, svg_attr_val_allows_ref, svg_allow_local_href
from bleach._vendor.html5lib.filters.sanitizer import Filter as SanitizerFilter
from bleach._vendor.html5lib._inputstream import HTMLInputStream
from bleach._vendor.html5lib.serializer import escape, HTMLSerializer
from bleach._vendor.html5lib._tokenizer import attributeMap, HTMLTokenizer
from bleach._vendor.html5lib._trie import Trie
ENTITIES = constants.entities
ENTITIES_TRIE = Trie(ENTITIES)
TAG_TOKEN_TYPES = {constants.tokenTypes['StartTag'], constants.tokenTypes['EndTag'], constants.tokenTypes['EmptyTag']}
TAG_TOKEN_TYPE_START = constants.tokenTypes['StartTag']
TAG_TOKEN_TYPE_END = constants.tokenTypes['EndTag']
TAG_TOKEN_TYPE_CHARACTERS = constants.tokenTypes['Characters']
TAG_TOKEN_TYPE_PARSEERROR = constants.tokenTypes['ParseError']
HTML_TAGS = frozenset(('a', 'abbr', 'address', 'area', 'article', 'aside', 'audio', 'b', 'base', 'bdi', 'bdo', 'blockquote', 'body', 'br', 'button', 'canvas', 'caption', 'cite', 'code', 'col', 'colgroup', 'data', 'datalist', 'dd', 'del', 'details', 'dfn', 'dialog', 'div', 'dl', 'dt', 'em', 'embed', 'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'header', 'hgroup', 'hr', 'html', 'i', 'iframe', 'img', 'input', 'ins', 'kbd', 'keygen', 'label', 'legend', 'li', 'link', 'map', 'mark', 'menu', 'meta', 'meter', 'nav', 'noscript', 'object', 'ol', 'optgroup', 'option', 'output', 'p', 'param', 'picture', 'pre', 'progress', 'q', 'rp', 'rt', 'ruby', 's', 'samp', 'script', 'section', 'select', 'slot', 'small', 'source', 'span', 'strong', 'style', 'sub', 'summary', 'sup', 'table', 'tbody', 'td', 'template', 'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr', 'track', 'u', 'ul', 'var', 'video', 'wbr'))
HTML_TAGS_BLOCK_LEVEL = frozenset(('address', 'article', 'aside', 'blockquote', 'details', 'dialog', 'dd', 'div', 'dl', 'dt', 'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hgroup', 'hr', 'li', 'main', 'nav', 'ol', 'p', 'pre', 'section', 'table', 'ul'))

class InputStreamWithMemory:
    """Wraps an HTMLInputStream to remember characters since last <

    This wraps existing HTMLInputStream classes to keep track of the stream
    since the last < which marked an open tag state.

    """

    def __init__(self, inner_stream):
        if False:
            for i in range(10):
                print('nop')
        self._inner_stream = inner_stream
        self.reset = self._inner_stream.reset
        self.position = self._inner_stream.position
        self._buffer = []

    @property
    def errors(self):
        if False:
            for i in range(10):
                print('nop')
        return self._inner_stream.errors

    @property
    def charEncoding(self):
        if False:
            while True:
                i = 10
        return self._inner_stream.charEncoding

    @property
    def changeEncoding(self):
        if False:
            for i in range(10):
                print('nop')
        return self._inner_stream.changeEncoding

    def char(self):
        if False:
            i = 10
            return i + 15
        c = self._inner_stream.char()
        if c:
            self._buffer.append(c)
        return c

    def charsUntil(self, characters, opposite=False):
        if False:
            for i in range(10):
                print('nop')
        chars = self._inner_stream.charsUntil(characters, opposite=opposite)
        self._buffer.extend(list(chars))
        return chars

    def unget(self, char):
        if False:
            for i in range(10):
                print('nop')
        if self._buffer:
            self._buffer.pop(-1)
        return self._inner_stream.unget(char)

    def get_tag(self):
        if False:
            i = 10
            return i + 15
        'Returns the stream history since last \'<\'\n\n        Since the buffer starts at the last \'<\' as as seen by tagOpenState(),\n        we know that everything from that point to when this method is called\n        is the "tag" that is being tokenized.\n\n        '
        return ''.join(self._buffer)

    def start_tag(self):
        if False:
            print('Hello World!')
        "Resets stream history to just '<'\n\n        This gets called by tagOpenState() which marks a '<' that denotes an\n        open tag. Any time we see that, we reset the buffer.\n\n        "
        self._buffer = ['<']

class BleachHTMLTokenizer(HTMLTokenizer):
    """Tokenizer that doesn't consume character entities"""

    def __init__(self, consume_entities=False, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.consume_entities = consume_entities
        self.stream = InputStreamWithMemory(self.stream)
        self.emitted_last_token = None

    def __iter__(self):
        if False:
            print('Hello World!')
        last_error_token = None
        for token in super().__iter__():
            if last_error_token is not None:
                if last_error_token['data'] == 'invalid-character-in-attribute-name' and token['type'] in TAG_TOKEN_TYPES and token.get('data'):
                    token['data'] = attributeMap(((attr_name, attr_value) for (attr_name, attr_value) in token['data'].items() if '"' not in attr_name and "'" not in attr_name and ('<' not in attr_name)))
                    last_error_token = None
                    yield token
                elif last_error_token['data'] == 'expected-closing-tag-but-got-char' and self.parser.tags is not None and (token['data'].lower().strip() not in self.parser.tags):
                    token['data'] = self.stream.get_tag()
                    token['type'] = TAG_TOKEN_TYPE_CHARACTERS
                    last_error_token = None
                    yield token
                elif token['type'] == TAG_TOKEN_TYPE_PARSEERROR:
                    yield last_error_token
                    last_error_token = token
                else:
                    yield last_error_token
                    yield token
                    last_error_token = None
                continue
            if token['type'] == TAG_TOKEN_TYPE_PARSEERROR:
                last_error_token = token
                continue
            yield token
        if last_error_token:
            if last_error_token['data'] == 'eof-in-tag-name':
                yield {'type': TAG_TOKEN_TYPE_CHARACTERS, 'data': '<' + self.currentToken['name']}
            else:
                yield last_error_token

    def consumeEntity(self, allowedChar=None, fromAttribute=False):
        if False:
            print('Hello World!')
        if self.consume_entities:
            return super().consumeEntity(allowedChar, fromAttribute)
        if fromAttribute:
            self.currentToken['data'][-1][1] += '&'
        else:
            self.tokenQueue.append({'type': TAG_TOKEN_TYPE_CHARACTERS, 'data': '&'})

    def tagOpenState(self):
        if False:
            for i in range(10):
                print('nop')
        self.stream.start_tag()
        return super().tagOpenState()

    def emitCurrentToken(self):
        if False:
            print('Hello World!')
        token = self.currentToken
        if self.parser.tags is not None and token['type'] in TAG_TOKEN_TYPES and (token['name'].lower() not in self.parser.tags):
            if self.parser.strip:
                if self.emitted_last_token and token['type'] == TAG_TOKEN_TYPE_START and (token['name'].lower() in HTML_TAGS_BLOCK_LEVEL):
                    new_data = '\n'
                else:
                    new_data = ''
            else:
                new_data = self.stream.get_tag()
            new_token = {'type': TAG_TOKEN_TYPE_CHARACTERS, 'data': new_data}
            self.currentToken = self.emitted_last_token = new_token
            self.tokenQueue.append(new_token)
            self.state = self.dataState
            return
        self.emitted_last_token = self.currentToken
        super().emitCurrentToken()

class BleachHTMLParser(HTMLParser):
    """Parser that uses BleachHTMLTokenizer"""

    def __init__(self, tags, strip, consume_entities, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        :arg tags: set of allowed tags--everything else is either stripped or\n            escaped; if None, then this doesn't look at tags at all\n        :arg strip: whether to strip disallowed tags (True) or escape them (False);\n            if tags=None, then this doesn't have any effect\n        :arg consume_entities: whether to consume entities (default behavior) or\n            leave them as is when tokenizing (BleachHTMLTokenizer-added behavior)\n\n        "
        self.tags = frozenset((tag.lower() for tag in tags)) if tags is not None else None
        self.strip = strip
        self.consume_entities = consume_entities
        super().__init__(**kwargs)

    def _parse(self, stream, innerHTML=False, container='div', scripting=True, **kwargs):
        if False:
            while True:
                i = 10
        self.innerHTMLMode = innerHTML
        self.container = container
        self.scripting = scripting
        self.tokenizer = BleachHTMLTokenizer(stream=stream, consume_entities=self.consume_entities, parser=self, **kwargs)
        self.reset()
        try:
            self.mainLoop()
        except ReparseException:
            self.reset()
            self.mainLoop()

def convert_entity(value):
    if False:
        while True:
            i = 10
    "Convert an entity (minus the & and ; part) into what it represents\n\n    This handles numeric, hex, and text entities.\n\n    :arg value: the string (minus the ``&`` and ``;`` part) to convert\n\n    :returns: unicode character or None if it's an ambiguous ampersand that\n        doesn't match a character entity\n\n    "
    if value[0] == '#':
        if len(value) < 2:
            return None
        if value[1] in ('x', 'X'):
            (int_as_string, base) = (value[2:], 16)
        else:
            (int_as_string, base) = (value[1:], 10)
        if int_as_string == '':
            return None
        code_point = int(int_as_string, base)
        if 0 < code_point < 1114112:
            return chr(code_point)
        else:
            return None
    return ENTITIES.get(value, None)

def convert_entities(text):
    if False:
        while True:
            i = 10
    'Converts all found entities in the text\n\n    :arg text: the text to convert entities in\n\n    :returns: unicode text with converted entities\n\n    '
    if '&' not in text:
        return text
    new_text = []
    for part in next_possible_entity(text):
        if not part:
            continue
        if part.startswith('&'):
            entity = match_entity(part)
            if entity is not None:
                converted = convert_entity(entity)
                if converted is not None:
                    new_text.append(converted)
                    remainder = part[len(entity) + 2:]
                    if part:
                        new_text.append(remainder)
                    continue
        new_text.append(part)
    return ''.join(new_text)

def match_entity(stream):
    if False:
        return 10
    'Returns first entity in stream or None if no entity exists\n\n    Note: For Bleach purposes, entities must start with a "&" and end with a\n    ";". This ignores ambiguous character entities that have no ";" at the end.\n\n    :arg stream: the character stream\n\n    :returns: the entity string without "&" or ";" if it\'s a valid character\n        entity; ``None`` otherwise\n\n    '
    if stream[0] != '&':
        raise ValueError('Stream should begin with "&"')
    stream = stream[1:]
    stream = list(stream)
    possible_entity = ''
    end_characters = '<&=;' + string.whitespace
    if stream and stream[0] == '#':
        possible_entity = '#'
        stream.pop(0)
        if stream and stream[0] in ('x', 'X'):
            allowed = '0123456789abcdefABCDEF'
            possible_entity += stream.pop(0)
        else:
            allowed = '0123456789'
        while stream and stream[0] not in end_characters:
            c = stream.pop(0)
            if c not in allowed:
                break
            possible_entity += c
        if possible_entity and stream and (stream[0] == ';'):
            return possible_entity
        return None
    while stream and stream[0] not in end_characters:
        c = stream.pop(0)
        possible_entity += c
        if not ENTITIES_TRIE.has_keys_with_prefix(possible_entity):
            return None
    if possible_entity and stream and (stream[0] == ';'):
        return possible_entity
    return None
AMP_SPLIT_RE = re.compile('(&)')

def next_possible_entity(text):
    if False:
        return 10
    'Takes a text and generates a list of possible entities\n\n    :arg text: the text to look at\n\n    :returns: generator where each part (except the first) starts with an\n        "&"\n\n    '
    for (i, part) in enumerate(AMP_SPLIT_RE.split(text)):
        if i == 0:
            yield part
        elif i % 2 == 0:
            yield ('&' + part)

class BleachHTMLSerializer(HTMLSerializer):
    """HTMLSerializer that undoes & -> &amp; in attributes and sets
    escape_rcdata to True
    """
    escape_rcdata = True

    def escape_base_amp(self, stoken):
        if False:
            for i in range(10):
                print('nop')
        'Escapes just bare & in HTML attribute values'
        stoken = stoken.replace('&amp;', '&')
        for part in next_possible_entity(stoken):
            if not part:
                continue
            if part.startswith('&'):
                entity = match_entity(part)
                if entity is not None and convert_entity(entity) is not None:
                    yield f'&{entity};'
                    part = part[len(entity) + 2:]
                    if part:
                        yield part
                    continue
            yield part.replace('&', '&amp;')

    def serialize(self, treewalker, encoding=None):
        if False:
            for i in range(10):
                print('nop')
        "Wrap HTMLSerializer.serialize and conver & to &amp; in attribute values\n\n        Note that this converts & to &amp; in attribute values where the & isn't\n        already part of an unambiguous character entity.\n\n        "
        in_tag = False
        after_equals = False
        for stoken in super().serialize(treewalker, encoding):
            if in_tag:
                if stoken == '>':
                    in_tag = False
                elif after_equals:
                    if stoken != '"':
                        yield from self.escape_base_amp(stoken)
                        after_equals = False
                        continue
                elif stoken == '=':
                    after_equals = True
                yield stoken
            else:
                if stoken.startswith('<'):
                    in_tag = True
                yield stoken