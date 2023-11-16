import re
from urllib.parse import quote
from bleach import callbacks as linkify_callbacks
from bleach import html5lib_shim
DEFAULT_CALLBACKS = [linkify_callbacks.nofollow]
TLDS = 'ac ad ae aero af ag ai al am an ao aq ar arpa as asia at au aw ax az\n       ba bb bd be bf bg bh bi biz bj bm bn bo br bs bt bv bw by bz ca cat\n       cc cd cf cg ch ci ck cl cm cn co com coop cr cu cv cx cy cz de dj dk\n       dm do dz ec edu ee eg er es et eu fi fj fk fm fo fr ga gb gd ge gf gg\n       gh gi gl gm gn gov gp gq gr gs gt gu gw gy hk hm hn hr ht hu id ie il\n       im in info int io iq ir is it je jm jo jobs jp ke kg kh ki km kn kp\n       kr kw ky kz la lb lc li lk lr ls lt lu lv ly ma mc md me mg mh mil mk\n       ml mm mn mo mobi mp mq mr ms mt mu museum mv mw mx my mz na name nc ne\n       net nf ng ni nl no np nr nu nz om org pa pe pf pg ph pk pl pm pn post\n       pr pro ps pt pw py qa re ro rs ru rw sa sb sc sd se sg sh si sj sk sl\n       sm sn so sr ss st su sv sx sy sz tc td tel tf tg th tj tk tl tm tn to\n       tp tr travel tt tv tw tz ua ug uk us uy uz va vc ve vg vi vn vu wf ws\n       xn xxx ye yt yu za zm zw'.split()
TLDS.reverse()

def build_url_re(tlds=TLDS, protocols=html5lib_shim.allowed_protocols):
    if False:
        while True:
            i = 10
    'Builds the url regex used by linkifier\n\n    If you want a different set of tlds or allowed protocols, pass those in\n    and stomp on the existing ``url_re``::\n\n        from bleach import linkifier\n\n        my_url_re = linkifier.build_url_re(my_tlds_list, my_protocols)\n\n        linker = LinkifyFilter(url_re=my_url_re)\n\n    '
    return re.compile('\\(*  # Match any opening parentheses.\n        \\b(?<![@.])(?:(?:{0}):/{{0,3}}(?:(?:\\w+:)?\\w+@)?)?  # http://\n        ([\\w-]+\\.)+(?:{1})(?:\\:[0-9]+)?(?!\\.\\w)\\b   # xx.yy.tld(:##)?\n        (?:[/?][^\\s\\{{\\}}\\|\\\\\\^\\[\\]`<>"]*)?\n            # /path/zz (excluding "unsafe" chars from RFC 1738,\n            # except for # and ~, which happen in practice)\n        '.format('|'.join(sorted(protocols)), '|'.join(sorted(tlds))), re.IGNORECASE | re.VERBOSE | re.UNICODE)
URL_RE = build_url_re()
PROTO_RE = re.compile('^[\\w-]+:/{0,3}', re.IGNORECASE)

def build_email_re(tlds=TLDS):
    if False:
        print('Hello World!')
    'Builds the email regex used by linkifier\n\n    If you want a different set of tlds, pass those in and stomp on the existing ``email_re``::\n\n        from bleach import linkifier\n\n        my_email_re = linkifier.build_email_re(my_tlds_list)\n\n        linker = LinkifyFilter(email_re=my_url_re)\n\n    '
    return re.compile('(?<!//)\n        (([-!#$%&\'*+/=?^_`{{}}|~0-9A-Z]+\n            (\\.[-!#$%&\'*+/=?^_`{{}}|~0-9A-Z]+)*  # dot-atom\n        |^"([\\001-\\010\\013\\014\\016-\\037!#-\\[\\]-\\177]\n            |\\\\[\\001-\\011\\013\\014\\016-\\177])*"  # quoted-string\n        )@(?:[A-Z0-9](?:[A-Z0-9-]{{0,61}}[A-Z0-9])?\\.)+(?:{0}))  # domain\n        '.format('|'.join(tlds)), re.IGNORECASE | re.MULTILINE | re.VERBOSE)
EMAIL_RE = build_email_re()

class Linker:
    """Convert URL-like strings in an HTML fragment to links

    This function converts strings that look like URLs, domain names and email
    addresses in text that may be an HTML fragment to links, while preserving:

    1. links already in the string
    2. urls found in attributes
    3. email addresses

    linkify does a best-effort approach and tries to recover from bad
    situations due to crazy text.

    """

    def __init__(self, callbacks=DEFAULT_CALLBACKS, skip_tags=None, parse_email=False, url_re=URL_RE, email_re=EMAIL_RE, recognized_tags=html5lib_shim.HTML_TAGS):
        if False:
            print('Hello World!')
        "Creates a Linker instance\n\n        :arg list callbacks: list of callbacks to run when adjusting tag attributes;\n            defaults to ``bleach.linkifier.DEFAULT_CALLBACKS``\n\n        :arg set skip_tags: set of tags that you don't want to linkify the\n            contents of; for example, you could set this to ``{'pre'}`` to skip\n            linkifying contents of ``pre`` tags; ``None`` means you don't\n            want linkify to skip any tags\n\n        :arg bool parse_email: whether or not to linkify email addresses\n\n        :arg url_re: url matching regex\n\n        :arg email_re: email matching regex\n\n        :arg set recognized_tags: the set of tags that linkify knows about;\n            everything else gets escaped\n\n        :returns: linkified text as unicode\n\n        "
        self.callbacks = callbacks
        self.skip_tags = skip_tags
        self.parse_email = parse_email
        self.url_re = url_re
        self.email_re = email_re
        self.parser = html5lib_shim.BleachHTMLParser(tags=frozenset(recognized_tags), strip=False, consume_entities=False, namespaceHTMLElements=False)
        self.walker = html5lib_shim.getTreeWalker('etree')
        self.serializer = html5lib_shim.BleachHTMLSerializer(quote_attr_values='always', omit_optional_tags=False, resolve_entities=False, sanitize=False, alphabetical_attributes=False)

    def linkify(self, text):
        if False:
            return 10
        'Linkify specified text\n\n        :arg str text: the text to add links to\n\n        :returns: linkified text as unicode\n\n        :raises TypeError: if ``text`` is not a text type\n\n        '
        if not isinstance(text, str):
            raise TypeError('argument must be of text type')
        if not text:
            return ''
        dom = self.parser.parseFragment(text)
        filtered = LinkifyFilter(source=self.walker(dom), callbacks=self.callbacks, skip_tags=self.skip_tags, parse_email=self.parse_email, url_re=self.url_re, email_re=self.email_re)
        return self.serializer.render(filtered)

class LinkifyFilter(html5lib_shim.Filter):
    """html5lib filter that linkifies text

    This will do the following:

    * convert email addresses into links
    * convert urls into links
    * edit existing links by running them through callbacks--the default is to
      add a ``rel="nofollow"``

    This filter can be used anywhere html5lib filters can be used.

    """

    def __init__(self, source, callbacks=DEFAULT_CALLBACKS, skip_tags=None, parse_email=False, url_re=URL_RE, email_re=EMAIL_RE):
        if False:
            for i in range(10):
                print('nop')
        "Creates a LinkifyFilter instance\n\n        :arg source: stream as an html5lib TreeWalker\n\n        :arg list callbacks: list of callbacks to run when adjusting tag attributes;\n            defaults to ``bleach.linkifier.DEFAULT_CALLBACKS``\n\n        :arg set skip_tags: set of tags that you don't want to linkify the\n            contents of; for example, you could set this to ``{'pre'}`` to skip\n            linkifying contents of ``pre`` tags\n\n        :arg bool parse_email: whether or not to linkify email addresses\n\n        :arg url_re: url matching regex\n\n        :arg email_re: email matching regex\n\n        "
        super().__init__(source)
        self.callbacks = callbacks or []
        self.skip_tags = skip_tags or {}
        self.parse_email = parse_email
        self.url_re = url_re
        self.email_re = email_re

    def apply_callbacks(self, attrs, is_new):
        if False:
            return 10
        'Given an attrs dict and an is_new bool, runs through callbacks\n\n        Callbacks can return an adjusted attrs dict or ``None``. In the case of\n        ``None``, we stop going through callbacks and return that and the link\n        gets dropped.\n\n        :arg dict attrs: map of ``(namespace, name)`` -> ``value``\n\n        :arg bool is_new: whether or not this link was added by linkify\n\n        :returns: adjusted attrs dict or ``None``\n\n        '
        for cb in self.callbacks:
            attrs = cb(attrs, is_new)
            if attrs is None:
                return None
        return attrs

    def extract_character_data(self, token_list):
        if False:
            for i in range(10):
                print('nop')
        'Extracts and squashes character sequences in a token stream'
        out = []
        for token in token_list:
            token_type = token['type']
            if token_type in ['Characters', 'SpaceCharacters']:
                out.append(token['data'])
        return ''.join(out)

    def handle_email_addresses(self, src_iter):
        if False:
            return 10
        'Handle email addresses in character tokens'
        for token in src_iter:
            if token['type'] == 'Characters':
                text = token['data']
                new_tokens = []
                end = 0
                for match in self.email_re.finditer(text):
                    if match.start() > end:
                        new_tokens.append({'type': 'Characters', 'data': text[end:match.start()]})
                    parts = match.group(0).split('@')
                    parts[0] = quote(parts[0])
                    address = '@'.join(parts)
                    attrs = {(None, 'href'): 'mailto:%s' % address, '_text': match.group(0)}
                    attrs = self.apply_callbacks(attrs, True)
                    if attrs is None:
                        new_tokens.append({'type': 'Characters', 'data': match.group(0)})
                    else:
                        _text = attrs.pop('_text', '')
                        new_tokens.extend([{'type': 'StartTag', 'name': 'a', 'data': attrs}, {'type': 'Characters', 'data': str(_text)}, {'type': 'EndTag', 'name': 'a'}])
                    end = match.end()
                if new_tokens:
                    if end < len(text):
                        new_tokens.append({'type': 'Characters', 'data': text[end:]})
                    yield from new_tokens
                    continue
            yield token

    def strip_non_url_bits(self, fragment):
        if False:
            return 10
        'Strips non-url bits from the url\n\n        This accounts for over-eager matching by the regex.\n\n        '
        prefix = suffix = ''
        while fragment:
            if fragment.startswith('('):
                prefix = prefix + '('
                fragment = fragment[1:]
                if fragment.endswith(')'):
                    suffix = ')' + suffix
                    fragment = fragment[:-1]
                continue
            if fragment.endswith(')') and '(' not in fragment:
                fragment = fragment[:-1]
                suffix = ')' + suffix
                continue
            if fragment.endswith(','):
                fragment = fragment[:-1]
                suffix = ',' + suffix
                continue
            if fragment.endswith('.'):
                fragment = fragment[:-1]
                suffix = '.' + suffix
                continue
            break
        return (fragment, prefix, suffix)

    def handle_links(self, src_iter):
        if False:
            for i in range(10):
                print('nop')
        'Handle links in character tokens'
        in_a = False
        for token in src_iter:
            if in_a:
                if token['type'] == 'EndTag' and token['name'] == 'a':
                    in_a = False
                yield token
                continue
            elif token['type'] == 'StartTag' and token['name'] == 'a':
                in_a = True
                yield token
                continue
            if token['type'] == 'Characters':
                text = token['data']
                new_tokens = []
                end = 0
                for match in self.url_re.finditer(text):
                    if match.start() > end:
                        new_tokens.append({'type': 'Characters', 'data': text[end:match.start()]})
                    url = match.group(0)
                    prefix = suffix = ''
                    (url, prefix, suffix) = self.strip_non_url_bits(url)
                    if PROTO_RE.search(url):
                        href = url
                    else:
                        href = 'http://%s' % url
                    attrs = {(None, 'href'): href, '_text': url}
                    attrs = self.apply_callbacks(attrs, True)
                    if attrs is None:
                        new_tokens.append({'type': 'Characters', 'data': prefix + url + suffix})
                    else:
                        if prefix:
                            new_tokens.append({'type': 'Characters', 'data': prefix})
                        _text = attrs.pop('_text', '')
                        new_tokens.extend([{'type': 'StartTag', 'name': 'a', 'data': attrs}, {'type': 'Characters', 'data': str(_text)}, {'type': 'EndTag', 'name': 'a'}])
                        if suffix:
                            new_tokens.append({'type': 'Characters', 'data': suffix})
                    end = match.end()
                if new_tokens:
                    if end < len(text):
                        new_tokens.append({'type': 'Characters', 'data': text[end:]})
                    yield from new_tokens
                    continue
            yield token

    def handle_a_tag(self, token_buffer):
        if False:
            while True:
                i = 10
        'Handle the "a" tag\n\n        This could adjust the link or drop it altogether depending on what the\n        callbacks return.\n\n        This yields the new set of tokens.\n\n        '
        a_token = token_buffer[0]
        if a_token['data']:
            attrs = a_token['data']
        else:
            attrs = {}
        text = self.extract_character_data(token_buffer)
        attrs['_text'] = text
        attrs = self.apply_callbacks(attrs, False)
        if attrs is None:
            yield {'type': 'Characters', 'data': text}
        else:
            new_text = attrs.pop('_text', '')
            a_token['data'] = attrs
            if text == new_text:
                yield a_token
                yield from token_buffer[1:]
            else:
                yield a_token
                yield {'type': 'Characters', 'data': str(new_text)}
                yield token_buffer[-1]

    def extract_entities(self, token):
        if False:
            for i in range(10):
                print('nop')
        "Handles Characters tokens with entities\n\n        Our overridden tokenizer doesn't do anything with entities. However,\n        that means that the serializer will convert all ``&`` in Characters\n        tokens to ``&amp;``.\n\n        Since we don't want that, we extract entities here and convert them to\n        Entity tokens so the serializer will let them be.\n\n        :arg token: the Characters token to work on\n\n        :returns: generator of tokens\n\n        "
        data = token.get('data', '')
        if '&' not in data:
            yield token
            return
        new_tokens = []
        for part in html5lib_shim.next_possible_entity(data):
            if not part:
                continue
            if part.startswith('&'):
                entity = html5lib_shim.match_entity(part)
                if entity is not None:
                    if entity == 'amp':
                        new_tokens.append({'type': 'Characters', 'data': '&'})
                    else:
                        new_tokens.append({'type': 'Entity', 'name': entity})
                    remainder = part[len(entity) + 2:]
                    if remainder:
                        new_tokens.append({'type': 'Characters', 'data': remainder})
                    continue
            new_tokens.append({'type': 'Characters', 'data': part})
        yield from new_tokens

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        in_a = False
        in_skip_tag = None
        token_buffer = []
        for token in super().__iter__():
            if in_a:
                if token['type'] == 'EndTag' and token['name'] == 'a':
                    token_buffer.append(token)
                    yield from self.handle_a_tag(token_buffer)
                    in_a = False
                    token_buffer = []
                else:
                    token_buffer.append(token)
                continue
            if token['type'] in ['StartTag', 'EmptyTag']:
                if token['name'] in self.skip_tags:
                    in_skip_tag = token['name']
                elif token['name'] == 'a':
                    in_a = True
                    token_buffer.append(token)
                    continue
            elif in_skip_tag and self.skip_tags:
                if token['type'] == 'EndTag' and token['name'] == in_skip_tag:
                    in_skip_tag = None
            elif not in_a and (not in_skip_tag) and (token['type'] == 'Characters'):
                new_stream = iter([token])
                if self.parse_email:
                    new_stream = self.handle_email_addresses(new_stream)
                new_stream = self.handle_links(new_stream)
                for new_token in new_stream:
                    yield from self.extract_entities(new_token)
                continue
            yield token