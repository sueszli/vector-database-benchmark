"""
    tinycss.css21
    -------------

    Parser for CSS 2.1
    http://www.w3.org/TR/CSS21/syndata.html

    :copyright: (c) 2012 by Simon Sapin.
    :license: BSD, see LICENSE for more details.
"""
from itertools import chain, islice
from tinycss.decoding import decode
from tinycss.token_data import TokenList
from tinycss.tokenizer import tokenize_grouped
from tinycss.parsing import strip_whitespace, remove_whitespace, split_on_comma, validate_value, validate_any, ParseError

class Stylesheet:
    """
    A parsed CSS stylesheet.

    .. attribute:: rules

        A mixed list, in source order, of :class:`RuleSet` and various
        at-rules such as :class:`ImportRule`, :class:`MediaRule`
        and :class:`PageRule`.
        Use their :obj:`at_keyword` attribute to distinguish them.

    .. attribute:: errors

        A list of :class:`~.parsing.ParseError`. Invalid rules and declarations
        are ignored, with the details logged in this list.

    .. attribute:: encoding

        The character encoding that was used to decode the stylesheet
        from bytes, or ``None`` for Unicode stylesheets.

    """

    def __init__(self, rules, errors, encoding):
        if False:
            return 10
        self.rules = rules
        self.errors = errors
        self.encoding = encoding

    def __repr__(self):
        if False:
            return 10
        return '<{0.__class__.__name__} {1} rules {2} errors>'.format(self, len(self.rules), len(self.errors))

class AtRule:
    """
    An unparsed at-rule.

    .. attribute:: at_keyword

        The normalized (lower-case) at-keyword as a string. Eg: ``'@page'``

    .. attribute:: head

        The part of the at-rule between the at-keyword and the ``{``
        marking the body, or the ``;`` marking the end of an at-rule without
        a body.  A :class:`~.token_data.TokenList`.

    .. attribute:: body

        The content of the body between ``{`` and ``}`` as a
        :class:`~.token_data.TokenList`, or ``None`` if there is no body
        (ie. if the rule ends with ``;``).

    The head was validated against the core grammar but **not** the body,
    as the body might contain declarations. In case of an error in a
    declaration, parsing should continue from the next declaration.
    The whole rule should not be ignored as it would be for an error
    in the head.

    These at-rules are expected to be parsed further before reaching
    the user API.

    """
    __slots__ = ('at_keyword', 'head', 'body', 'line', 'column')

    def __init__(self, at_keyword, head, body, line, column):
        if False:
            while True:
                i = 10
        self.at_keyword = at_keyword
        self.head = TokenList(head)
        self.body = TokenList(body) if body is not None else body
        self.line = line
        self.column = column

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<{0.__class__.__name__} {0.line}:{0.column} {0.at_keyword}>'.format(self)

class RuleSet:
    """A ruleset.

    .. attribute:: at_keyword

        Always ``None``. Helps to tell rulesets apart from at-rules.

    .. attribute:: selector

        The selector as a :class:`~.token_data.TokenList`.
        In CSS 3, this is actually called a selector group.

        ``rule.selector.as_css()`` gives the selector as a string.
        This string can be used with *cssselect*, see :ref:`selectors3`.

    .. attribute:: declarations

        The list of :class:`Declaration`, in source order.

    """
    at_keyword = None
    __slots__ = ('selector', 'declarations', 'line', 'column')

    def __init__(self, selector, declarations, line, column):
        if False:
            i = 10
            return i + 15
        self.selector = TokenList(selector)
        self.declarations = declarations
        self.line = line
        self.column = column

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{0.__class__.__name__} at {0.line}:{0.column} {1}>'.format(self, self.selector.as_css())

class Declaration:
    """A property declaration.

    .. attribute:: name

        The property name as a normalized (lower-case) string.

    .. attribute:: value

        The property value as a :class:`~.token_data.TokenList`.

        The value is not parsed. UAs using tinycss may only support
        some properties or some values and tinycss does not know which.
        They need to parse values themselves and ignore declarations with
        unknown or unsupported properties or values, and fall back
        on any previous declaration.

        :mod:`tinycss.color3` parses color values, but other values
        will need specific parsing/validation code.

    .. attribute:: priority

        Either the string ``'important'`` or ``None``.

    """
    __slots__ = ('name', 'value', 'priority', 'line', 'column')

    def __init__(self, name, value, priority, line, column):
        if False:
            return 10
        self.name = name
        self.value = TokenList(value)
        self.priority = priority
        self.line = line
        self.column = column

    def __repr__(self):
        if False:
            return 10
        priority = ' !' + self.priority if self.priority else ''
        return '<{0.__class__.__name__} {0.line}:{0.column} {0.name}: {1}{2}>'.format(self, self.value.as_css(), priority)

class PageRule:
    """A parsed CSS 2.1 @page rule.

    .. attribute:: at_keyword

        Always ``'@page'``

    .. attribute:: selector

        The page selector.
        In CSS 2.1 this is either ``None`` (no selector), or the string
        ``'first'``, ``'left'`` or ``'right'`` for the pseudo class
        of the same name.

    .. attribute:: specificity

        Specificity of the page selector. This is a tuple of four integers,
        but these tuples are mostly meant to be compared to each other.

    .. attribute:: declarations

        A list of :class:`Declaration`, in source order.

    .. attribute:: at_rules

        The list of parsed at-rules inside the @page block, in source order.
        Always empty for CSS 2.1.

    """
    at_keyword = '@page'
    __slots__ = ('selector', 'specificity', 'declarations', 'at_rules', 'line', 'column')

    def __init__(self, selector, specificity, declarations, at_rules, line, column):
        if False:
            print('Hello World!')
        self.selector = selector
        self.specificity = specificity
        self.declarations = declarations
        self.at_rules = at_rules
        self.line = line
        self.column = column

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{0.__class__.__name__} {0.line}:{0.column} {0.selector}>'.format(self)

class MediaRule:
    """A parsed @media rule.

    .. attribute:: at_keyword

        Always ``'@media'``

    .. attribute:: media

        For CSS 2.1 without media queries: the media types
        as a list of strings.

    .. attribute:: rules

        The list :class:`RuleSet` and various at-rules inside the @media
        block, in source order.

    """
    at_keyword = '@media'
    __slots__ = ('media', 'rules', 'line', 'column')

    def __init__(self, media, rules, line, column):
        if False:
            return 10
        self.media = media
        self.rules = rules
        self.line = line
        self.column = column

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<{0.__class__.__name__} {0.line}:{0.column} {0.media}>'.format(self)

class ImportRule:
    """A parsed @import rule.

    .. attribute:: at_keyword

        Always ``'@import'``

    .. attribute:: uri

        The URI to be imported, as read from the stylesheet.
        (URIs are not made absolute.)

    .. attribute:: media

        For CSS 2.1 without media queries: the media types
        as a list of strings.
        This attribute is explicitly ``['all']`` if the media was omitted
        in the source.

    """
    at_keyword = '@import'
    __slots__ = ('uri', 'media', 'line', 'column')

    def __init__(self, uri, media, line, column):
        if False:
            print('Hello World!')
        self.uri = uri
        self.media = media
        self.line = line
        self.column = column

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{0.__class__.__name__} {0.line}:{0.column} {0.uri}>'.format(self)

def _remove_at_charset(tokens):
    if False:
        print('Hello World!')
    'Remove any valid @charset at the beginning of a token stream.\n\n    :param tokens:\n        An iterable of tokens\n    :returns:\n        A possibly truncated iterable of tokens\n\n    '
    tokens = iter(tokens)
    header = list(islice(tokens, 4))
    if [t.type for t in header] == ['ATKEYWORD', 'S', 'STRING', ';']:
        (atkw, space, string, semicolon) = header
        if (atkw.value, space.value) == ('@charset', ' ') and string.as_css()[0] == '"':
            return tokens
    return chain(header, tokens)

class CSS21Parser:
    """Parser for CSS 2.1

    This parser supports the core CSS syntax as well as @import, @media,
    @page and !important.

    Note that property values are still not parsed, as UAs using this
    parser may only support some properties or some values.

    Currently the parser holds no state. It being a class only allows
    subclassing and overriding its methods.

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.at_parsers = {'@' + x: getattr(self, 'parse_%s_rule' % x) for x in ('media', 'page', 'import', 'charset')}

    def parse_stylesheet_file(self, css_file, protocol_encoding=None, linking_encoding=None, document_encoding=None):
        if False:
            return 10
        'Parse a stylesheet from a file or filename.\n\n        Character encoding-related parameters and behavior are the same\n        as in :meth:`parse_stylesheet_bytes`.\n\n        :param css_file:\n            Either a file (any object with a :meth:`~file.read` method)\n            or a filename.\n        :return:\n            A :class:`Stylesheet`.\n\n        '
        if hasattr(css_file, 'read'):
            css_bytes = css_file.read()
        else:
            with open(css_file, 'rb') as fd:
                css_bytes = fd.read()
        return self.parse_stylesheet_bytes(css_bytes, protocol_encoding, linking_encoding, document_encoding)

    def parse_stylesheet_bytes(self, css_bytes, protocol_encoding=None, linking_encoding=None, document_encoding=None):
        if False:
            for i in range(10):
                print('nop')
        'Parse a stylesheet from a byte string.\n\n        The character encoding is determined from the passed metadata and the\n        ``@charset`` rule in the stylesheet (if any).\n        If no encoding information is available or decoding fails,\n        decoding defaults to UTF-8 and then fall back on ISO-8859-1.\n\n        :param css_bytes:\n            A CSS stylesheet as a byte string.\n        :param protocol_encoding:\n            The "charset" parameter of a "Content-Type" HTTP header (if any),\n            or similar metadata for other protocols.\n        :param linking_encoding:\n            ``<link charset="">`` or other metadata from the linking mechanism\n            (if any)\n        :param document_encoding:\n            Encoding of the referring style sheet or document (if any)\n        :return:\n            A :class:`Stylesheet`.\n\n        '
        (css_unicode, encoding) = decode(css_bytes, protocol_encoding, linking_encoding, document_encoding)
        return self.parse_stylesheet(css_unicode, encoding=encoding)

    def parse_stylesheet(self, css_unicode, encoding=None):
        if False:
            while True:
                i = 10
        'Parse a stylesheet from an Unicode string.\n\n        :param css_unicode:\n            A CSS stylesheet as an unicode string.\n        :param encoding:\n            The character encoding used to decode the stylesheet from bytes,\n            if any.\n        :return:\n            A :class:`Stylesheet`.\n\n        '
        tokens = tokenize_grouped(css_unicode)
        if encoding:
            tokens = _remove_at_charset(tokens)
        (rules, errors) = self.parse_rules(tokens, context='stylesheet')
        return Stylesheet(rules, errors, encoding)

    def parse_style_attr(self, css_source):
        if False:
            while True:
                i = 10
        'Parse a "style" attribute (eg. of an HTML element).\n\n        This method only accepts Unicode as the source (HTML) document\n        is supposed to handle the character encoding.\n\n        :param css_source:\n            The attribute value, as an unicode string.\n        :return:\n            A tuple of the list of valid :class:`Declaration` and\n            a list of :class:`~.parsing.ParseError`.\n        '
        return self.parse_declaration_list(tokenize_grouped(css_source))

    def parse_rules(self, tokens, context):
        if False:
            for i in range(10):
                print('nop')
        "Parse a sequence of rules (rulesets and at-rules).\n\n        :param tokens:\n            An iterable of tokens.\n        :param context:\n            Either ``'stylesheet'`` or an at-keyword such as ``'@media'``.\n            (Most at-rules are only allowed in some contexts.)\n        :return:\n            A tuple of a list of parsed rules and a list of\n            :class:`~.parsing.ParseError`.\n\n        "
        rules = []
        errors = []
        tokens = iter(tokens)
        for token in tokens:
            if token.type not in ('S', 'CDO', 'CDC'):
                try:
                    if token.type == 'ATKEYWORD':
                        rule = self.read_at_rule(token, tokens)
                        result = self.parse_at_rule(rule, rules, errors, context)
                        rules.append(result)
                    else:
                        (rule, rule_errors) = self.parse_ruleset(token, tokens)
                        rules.append(rule)
                        errors.extend(rule_errors)
                except ParseError as exc:
                    errors.append(exc)
        return (rules, errors)

    def read_at_rule(self, at_keyword_token, tokens):
        if False:
            return 10
        'Read an at-rule from a token stream.\n\n        :param at_keyword_token:\n            The ATKEYWORD token that starts this at-rule\n            You may have read it already to distinguish the rule\n            from a ruleset.\n        :param tokens:\n            An iterator of subsequent tokens. Will be consumed just enough\n            for one at-rule.\n        :return:\n            An unparsed :class:`AtRule`.\n        :raises:\n            :class:`~.parsing.ParseError` if the head is invalid for the core\n            grammar. The body is **not** validated. See :class:`AtRule`.\n\n        '
        at_keyword = at_keyword_token.value.lower()
        head = []
        token = at_keyword_token
        for token in tokens:
            if token.type in '{;':
                break
            else:
                head.append(token)
        head = strip_whitespace(head)
        for head_token in head:
            validate_any(head_token, 'at-rule head')
        body = token.content if token.type == '{' else None
        return AtRule(at_keyword, head, body, at_keyword_token.line, at_keyword_token.column)

    def parse_at_rule(self, rule, previous_rules, errors, context):
        if False:
            for i in range(10):
                print('nop')
        "Parse an at-rule.\n\n        Subclasses that override this method must use ``super()`` and\n        pass its return value for at-rules they do not know.\n\n        In CSS 2.1, this method handles @charset, @import, @media and @page\n        rules.\n\n        :param rule:\n            An unparsed :class:`AtRule`.\n        :param previous_rules:\n            The list of at-rules and rulesets that have been parsed so far\n            in this context. This list can be used to decide if the current\n            rule is valid. (For example, @import rules are only allowed\n            before anything but a @charset rule.)\n        :param context:\n            Either ``'stylesheet'`` or an at-keyword such as ``'@media'``.\n            (Most at-rules are only allowed in some contexts.)\n        :raises:\n            :class:`~.parsing.ParseError` if the rule is invalid.\n        :return:\n            A parsed at-rule\n\n        "
        try:
            parser = self.at_parsers[rule.at_keyword]
        except KeyError:
            raise ParseError(rule, 'unknown at-rule in {0} context: {1}'.format(context, rule.at_keyword))
        else:
            return parser(rule, previous_rules, errors, context)

    def parse_page_rule(self, rule, previous_rules, errors, context):
        if False:
            return 10
        if context != 'stylesheet':
            raise ParseError(rule, '@page rule not allowed in ' + context)
        (selector, specificity) = self.parse_page_selector(rule.head)
        if rule.body is None:
            raise ParseError(rule, 'invalid {0} rule: missing block'.format(rule.at_keyword))
        (declarations, at_rules, rule_errors) = self.parse_declarations_and_at_rules(rule.body, '@page')
        errors.extend(rule_errors)
        return PageRule(selector, specificity, declarations, at_rules, rule.line, rule.column)

    def parse_media_rule(self, rule, previous_rules, errors, context):
        if False:
            i = 10
            return i + 15
        if context != 'stylesheet':
            raise ParseError(rule, '@media rule not allowed in ' + context)
        media = self.parse_media(rule.head, errors)
        if rule.body is None:
            raise ParseError(rule, 'invalid {0} rule: missing block'.format(rule.at_keyword))
        (rules, rule_errors) = self.parse_rules(rule.body, '@media')
        errors.extend(rule_errors)
        return MediaRule(media, rules, rule.line, rule.column)

    def parse_import_rule(self, rule, previous_rules, errors, context):
        if False:
            print('Hello World!')
        if context != 'stylesheet':
            raise ParseError(rule, '@import rule not allowed in ' + context)
        for previous_rule in previous_rules:
            if previous_rule.at_keyword not in ('@charset', '@import'):
                if previous_rule.at_keyword:
                    type_ = 'an {0} rule'.format(previous_rule.at_keyword)
                else:
                    type_ = 'a ruleset'
                raise ParseError(previous_rule, '@import rule not allowed after ' + type_)
        head = rule.head
        if not head:
            raise ParseError(rule, 'expected URI or STRING for @import rule')
        if head[0].type not in ('URI', 'STRING'):
            raise ParseError(rule, 'expected URI or STRING for @import rule, got ' + head[0].type)
        uri = head[0].value
        media = self.parse_media(strip_whitespace(head[1:]), errors)
        if rule.body is not None:
            raise ParseError(head[-1], "expected ';', got a block")
        return ImportRule(uri, media, rule.line, rule.column)

    def parse_charset_rule(self, rule, previous_rules, errors, context):
        if False:
            i = 10
            return i + 15
        raise ParseError(rule, 'mis-placed or malformed @charset rule')

    def parse_media(self, tokens, errors):
        if False:
            print('Hello World!')
        'For CSS 2.1, parse a list of media types.\n\n        Media Queries are expected to override this.\n\n        :param tokens:\n            A list of tokens\n        :raises:\n            :class:`~.parsing.ParseError` on invalid media types/queries\n        :returns:\n            For CSS 2.1, a list of media types as strings\n        '
        if not tokens:
            return ['all']
        media_types = []
        for part in split_on_comma(remove_whitespace(tokens)):
            types = [token.type for token in part]
            if types == ['IDENT']:
                media_types.append(part[0].value)
            else:
                raise ParseError(tokens[0], 'expected a media type' + (', got ' + ', '.join(types) if types else ''))
        return media_types

    def parse_page_selector(self, tokens):
        if False:
            while True:
                i = 10
        "Parse an @page selector.\n\n        :param tokens:\n            An iterable of token, typically from the  ``head`` attribute of\n            an unparsed :class:`AtRule`.\n        :returns:\n            A page selector. For CSS 2.1, this is ``'first'``, ``'left'``,\n            ``'right'`` or ``None``.\n        :raises:\n            :class:`~.parsing.ParseError` on invalid selectors\n\n        "
        if not tokens:
            return (None, (0, 0))
        if len(tokens) == 2 and tokens[0].type == ':' and (tokens[1].type == 'IDENT'):
            pseudo_class = tokens[1].value
            specificity = {'first': (1, 0), 'left': (0, 1), 'right': (0, 1)}.get(pseudo_class)
            if specificity:
                return (pseudo_class, specificity)
        raise ParseError(tokens[0], 'invalid @page selector')

    def parse_declarations_and_at_rules(self, tokens, context):
        if False:
            for i in range(10):
                print('nop')
        "Parse a mixed list of declarations and at rules, as found eg.\n        in the body of an @page rule.\n\n        Note that to add supported at-rules inside @page,\n        :class:`~.page3.CSSPage3Parser` extends :meth:`parse_at_rule`,\n        not this method.\n\n        :param tokens:\n            An iterable of token, typically from the  ``body`` attribute of\n            an unparsed :class:`AtRule`.\n        :param context:\n            An at-keyword such as ``'@page'``.\n            (Most at-rules are only allowed in some contexts.)\n        :returns:\n            A tuple of:\n\n            * A list of :class:`Declaration`\n            * A list of parsed at-rules (empty for CSS 2.1)\n            * A list of :class:`~.parsing.ParseError`\n\n        "
        at_rules = []
        declarations = []
        errors = []
        tokens = iter(tokens)
        for token in tokens:
            if token.type == 'ATKEYWORD':
                try:
                    rule = self.read_at_rule(token, tokens)
                    result = self.parse_at_rule(rule, at_rules, errors, context)
                    at_rules.append(result)
                except ParseError as err:
                    errors.append(err)
            elif token.type != 'S':
                declaration_tokens = []
                while token and token.type != ';':
                    declaration_tokens.append(token)
                    token = next(tokens, None)
                if declaration_tokens:
                    try:
                        declarations.append(self.parse_declaration(declaration_tokens))
                    except ParseError as err:
                        errors.append(err)
        return (declarations, at_rules, errors)

    def parse_ruleset(self, first_token, tokens):
        if False:
            print('Hello World!')
        'Parse a ruleset: a selector followed by declaration block.\n\n        :param first_token:\n            The first token of the ruleset (probably of the selector).\n            You may have read it already to distinguish the rule\n            from an at-rule.\n        :param tokens:\n            an iterator of subsequent tokens. Will be consumed just enough\n            for one ruleset.\n        :return:\n            a tuple of a :class:`RuleSet` and an error list.\n            The errors are recovered :class:`~.parsing.ParseError` in declarations.\n            (Parsing continues from the next declaration on such errors.)\n        :raises:\n            :class:`~.parsing.ParseError` if the selector is invalid for the\n            core grammar.\n            Note a that a selector can be valid for the core grammar but\n            not for CSS 2.1 or another level.\n\n        '
        selector = []
        for token in chain([first_token], tokens):
            if token.type == '{':
                selector = strip_whitespace(selector)
                if not selector:
                    raise ParseError(first_token, 'empty selector')
                for selector_token in selector:
                    validate_any(selector_token, 'selector')
                (declarations, errors) = self.parse_declaration_list(token.content)
                ruleset = RuleSet(selector, declarations, first_token.line, first_token.column)
                return (ruleset, errors)
            else:
                selector.append(token)
        raise ParseError(token, 'no declaration block found for ruleset')

    def parse_declaration_list(self, tokens):
        if False:
            while True:
                i = 10
        'Parse a ``;`` separated declaration list.\n\n        You may want to use :meth:`parse_declarations_and_at_rules` (or\n        some other method that uses :func:`parse_declaration` directly)\n        instead if you have not just declarations in the same context.\n\n        :param tokens:\n            an iterable of tokens. Should stop at (before) the end\n            of the block, as marked by ``}``.\n        :return:\n            a tuple of the list of valid :class:`Declaration` and a list\n            of :class:`~.parsing.ParseError`\n\n        '
        parts = []
        this_part = []
        for token in tokens:
            if token.type == ';':
                parts.append(this_part)
                this_part = []
            else:
                this_part.append(token)
        parts.append(this_part)
        declarations = []
        errors = []
        for tokens in parts:
            tokens = strip_whitespace(tokens)
            if tokens:
                try:
                    declarations.append(self.parse_declaration(tokens))
                except ParseError as exc:
                    errors.append(exc)
        return (declarations, errors)

    def parse_declaration(self, tokens):
        if False:
            return 10
        "Parse a single declaration.\n\n        :param tokens:\n            an iterable of at least one token. Should stop at (before)\n            the end of the declaration, as marked by a ``;`` or ``}``.\n            Empty declarations (ie. consecutive ``;`` with only white space\n            in-between) should be skipped earlier and not passed to\n            this method.\n        :returns:\n            a :class:`Declaration`\n        :raises:\n            :class:`~.parsing.ParseError` if the tokens do not match the\n            'declaration' production of the core grammar.\n\n        "
        tokens = iter(tokens)
        name_token = next(tokens)
        if name_token.type == 'IDENT':
            property_name = name_token.value.lower()
        else:
            raise ParseError(name_token, 'expected a property name, got {0}'.format(name_token.type))
        token = name_token
        for token in tokens:
            if token.type == ':':
                break
            elif token.type != 'S':
                raise ParseError(token, "expected ':', got {0}".format(token.type))
        else:
            raise ParseError(token, "expected ':'")
        value = strip_whitespace(list(tokens))
        if not value:
            raise ParseError(token, 'expected a property value')
        validate_value(value)
        (value, priority) = self.parse_value_priority(value)
        return Declaration(property_name, value, priority, name_token.line, name_token.column)

    def parse_value_priority(self, tokens):
        if False:
            for i in range(10):
                print('nop')
        'Separate any ``!important`` marker at the end of a property value.\n\n        :param tokens:\n            A list of tokens for the property value.\n        :returns:\n            A tuple of the actual property value (a list of tokens)\n            and the :attr:`~Declaration.priority`.\n        '
        value = list(tokens)
        token = value.pop()
        if token.type == 'IDENT' and token.value.lower() == 'important':
            while value:
                token = value.pop()
                if token.type == 'DELIM' and token.value == '!':
                    while value and value[-1].type == 'S':
                        value.pop()
                    if not value:
                        raise ParseError(token, 'expected a value before !important')
                    return (value, 'important')
                elif token.type != 'S':
                    break
        return (tokens, None)