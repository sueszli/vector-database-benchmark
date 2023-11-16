"""Use the HTMLParser library to parse HTML files that aren't too bad."""
__license__ = 'MIT'
__all__ = ['HTMLParserTreeBuilder']
from html.parser import HTMLParser
import sys
import warnings
from bs4.element import CData, Comment, Declaration, Doctype, ProcessingInstruction
from bs4.dammit import EntitySubstitution, UnicodeDammit
from bs4.builder import DetectsXMLParsedAsHTML, ParserRejectedMarkup, HTML, HTMLTreeBuilder, STRICT
HTMLPARSER = 'html.parser'

class BeautifulSoupHTMLParser(HTMLParser, DetectsXMLParsedAsHTML):
    """A subclass of the Python standard library's HTMLParser class, which
    listens for HTMLParser events and translates them into calls
    to Beautiful Soup's tree construction API.
    """
    IGNORE = 'ignore'
    REPLACE = 'replace'

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n        :param on_duplicate_attribute: A strategy for what to do if a\n            tag includes the same attribute more than once. Accepted\n            values are: REPLACE (replace earlier values with later\n            ones, the default), IGNORE (keep the earliest value\n            encountered), or a callable. A callable must take three\n            arguments: the dictionary of attributes already processed,\n            the name of the duplicate attribute, and the most recent value\n            encountered.           \n        '
        self.on_duplicate_attribute = kwargs.pop('on_duplicate_attribute', self.REPLACE)
        HTMLParser.__init__(self, *args, **kwargs)
        self.already_closed_empty_element = []
        self._initialize_xml_detector()

    def error(self, message):
        if False:
            while True:
                i = 10
        raise ParserRejectedMarkup(message)

    def handle_startendtag(self, name, attrs):
        if False:
            while True:
                i = 10
        "Handle an incoming empty-element tag.\n\n        This is only called when the markup looks like <tag/>.\n\n        :param name: Name of the tag.\n        :param attrs: Dictionary of the tag's attributes.\n        "
        tag = self.handle_starttag(name, attrs, handle_empty_element=False)
        self.handle_endtag(name)

    def handle_starttag(self, name, attrs, handle_empty_element=True):
        if False:
            i = 10
            return i + 15
        "Handle an opening tag, e.g. '<tag>'\n\n        :param name: Name of the tag.\n        :param attrs: Dictionary of the tag's attributes.\n        :param handle_empty_element: True if this tag is known to be\n            an empty-element tag (i.e. there is not expected to be any\n            closing tag).\n        "
        attr_dict = {}
        for (key, value) in attrs:
            if value is None:
                value = ''
            if key in attr_dict:
                on_dupe = self.on_duplicate_attribute
                if on_dupe == self.IGNORE:
                    pass
                elif on_dupe in (None, self.REPLACE):
                    attr_dict[key] = value
                else:
                    on_dupe(attr_dict, key, value)
            else:
                attr_dict[key] = value
            attrvalue = '""'
        (sourceline, sourcepos) = self.getpos()
        tag = self.soup.handle_starttag(name, None, None, attr_dict, sourceline=sourceline, sourcepos=sourcepos)
        if tag and tag.is_empty_element and handle_empty_element:
            self.handle_endtag(name, check_already_closed=False)
            self.already_closed_empty_element.append(name)
        if self._root_tag is None:
            self._root_tag_encountered(name)

    def handle_endtag(self, name, check_already_closed=True):
        if False:
            return 10
        "Handle a closing tag, e.g. '</tag>'\n        \n        :param name: A tag name.\n        :param check_already_closed: True if this tag is expected to\n           be the closing portion of an empty-element tag,\n           e.g. '<tag></tag>'.\n        "
        if check_already_closed and name in self.already_closed_empty_element:
            self.already_closed_empty_element.remove(name)
        else:
            self.soup.handle_endtag(name)

    def handle_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Handle some textual data that shows up between tags.'
        self.soup.handle_data(data)

    def handle_charref(self, name):
        if False:
            print('Hello World!')
        'Handle a numeric character reference by converting it to the\n        corresponding Unicode character and treating it as textual\n        data.\n\n        :param name: Character number, possibly in hexadecimal.\n        '
        if name.startswith('x'):
            real_name = int(name.lstrip('x'), 16)
        elif name.startswith('X'):
            real_name = int(name.lstrip('X'), 16)
        else:
            real_name = int(name)
        data = None
        if real_name < 256:
            for encoding in (self.soup.original_encoding, 'windows-1252'):
                if not encoding:
                    continue
                try:
                    data = bytearray([real_name]).decode(encoding)
                except UnicodeDecodeError as e:
                    pass
        if not data:
            try:
                data = chr(real_name)
            except (ValueError, OverflowError) as e:
                pass
        data = data or '�'
        self.handle_data(data)

    def handle_entityref(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Handle a named entity reference by converting it to the\n        corresponding Unicode character(s) and treating it as textual\n        data.\n\n        :param name: Name of the entity reference.\n        '
        character = EntitySubstitution.HTML_ENTITY_TO_CHARACTER.get(name)
        if character is not None:
            data = character
        else:
            data = '&%s' % name
        self.handle_data(data)

    def handle_comment(self, data):
        if False:
            while True:
                i = 10
        'Handle an HTML comment.\n\n        :param data: The text of the comment.\n        '
        self.soup.endData()
        self.soup.handle_data(data)
        self.soup.endData(Comment)

    def handle_decl(self, data):
        if False:
            while True:
                i = 10
        'Handle a DOCTYPE declaration.\n\n        :param data: The text of the declaration.\n        '
        self.soup.endData()
        data = data[len('DOCTYPE '):]
        self.soup.handle_data(data)
        self.soup.endData(Doctype)

    def unknown_decl(self, data):
        if False:
            print('Hello World!')
        'Handle a declaration of unknown type -- probably a CDATA block.\n\n        :param data: The text of the declaration.\n        '
        if data.upper().startswith('CDATA['):
            cls = CData
            data = data[len('CDATA['):]
        else:
            cls = Declaration
        self.soup.endData()
        self.soup.handle_data(data)
        self.soup.endData(cls)

    def handle_pi(self, data):
        if False:
            print('Hello World!')
        'Handle a processing instruction.\n\n        :param data: The text of the instruction.\n        '
        self.soup.endData()
        self.soup.handle_data(data)
        self._document_might_be_xml(data)
        self.soup.endData(ProcessingInstruction)

class HTMLParserTreeBuilder(HTMLTreeBuilder):
    """A Beautiful soup `TreeBuilder` that uses the `HTMLParser` parser,
    found in the Python standard library.
    """
    is_xml = False
    picklable = True
    NAME = HTMLPARSER
    features = [NAME, HTML, STRICT]
    TRACKS_LINE_NUMBERS = True

    def __init__(self, parser_args=None, parser_kwargs=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Constructor.\n\n        :param parser_args: Positional arguments to pass into \n            the BeautifulSoupHTMLParser constructor, once it's\n            invoked.\n        :param parser_kwargs: Keyword arguments to pass into \n            the BeautifulSoupHTMLParser constructor, once it's\n            invoked.\n        :param kwargs: Keyword arguments for the superclass constructor.\n        "
        extra_parser_kwargs = dict()
        for arg in ('on_duplicate_attribute',):
            if arg in kwargs:
                value = kwargs.pop(arg)
                extra_parser_kwargs[arg] = value
        super(HTMLParserTreeBuilder, self).__init__(**kwargs)
        parser_args = parser_args or []
        parser_kwargs = parser_kwargs or {}
        parser_kwargs.update(extra_parser_kwargs)
        parser_kwargs['convert_charrefs'] = False
        self.parser_args = (parser_args, parser_kwargs)

    def prepare_markup(self, markup, user_specified_encoding=None, document_declared_encoding=None, exclude_encodings=None):
        if False:
            return 10
        'Run any preliminary steps necessary to make incoming markup\n        acceptable to the parser.\n\n        :param markup: Some markup -- probably a bytestring.\n        :param user_specified_encoding: The user asked to try this encoding.\n        :param document_declared_encoding: The markup itself claims to be\n            in this encoding.\n        :param exclude_encodings: The user asked _not_ to try any of\n            these encodings.\n\n        :yield: A series of 4-tuples:\n         (markup, encoding, declared encoding,\n          has undergone character replacement)\n\n         Each 4-tuple represents a strategy for converting the\n         document to Unicode and parsing it. Each strategy will be tried \n         in turn.\n        '
        if isinstance(markup, str):
            yield (markup, None, None, False)
            return
        known_definite_encodings = [user_specified_encoding]
        user_encodings = [document_declared_encoding]
        try_encodings = [user_specified_encoding, document_declared_encoding]
        dammit = UnicodeDammit(markup, known_definite_encodings=known_definite_encodings, user_encodings=user_encodings, is_html=True, exclude_encodings=exclude_encodings)
        yield (dammit.markup, dammit.original_encoding, dammit.declared_html_encoding, dammit.contains_replacement_characters)

    def feed(self, markup):
        if False:
            return 10
        'Run some incoming markup through some parsing process,\n        populating the `BeautifulSoup` object in self.soup.\n        '
        (args, kwargs) = self.parser_args
        parser = BeautifulSoupHTMLParser(*args, **kwargs)
        parser.soup = self.soup
        try:
            parser.feed(markup)
        except AssertionError as e:
            raise ParserRejectedMarkup(e)
        parser.close()
        parser.already_closed_empty_element = []