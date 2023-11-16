__license__ = 'MIT'
__all__ = ['LXMLTreeBuilderForXML', 'LXMLTreeBuilder']
try:
    from collections.abc import Callable
except ImportError as e:
    from collections import Callable
from io import BytesIO
from io import StringIO
from lxml import etree
from bs4.element import Comment, Doctype, NamespacedAttribute, ProcessingInstruction, XMLProcessingInstruction
from bs4.builder import DetectsXMLParsedAsHTML, FAST, HTML, HTMLTreeBuilder, PERMISSIVE, ParserRejectedMarkup, TreeBuilder, XML
from bs4.dammit import EncodingDetector
LXML = 'lxml'

def _invert(d):
    if False:
        while True:
            i = 10
    'Invert a dictionary.'
    return dict(((v, k) for (k, v) in list(d.items())))

class LXMLTreeBuilderForXML(TreeBuilder):
    DEFAULT_PARSER_CLASS = etree.XMLParser
    is_xml = True
    processing_instruction_class = XMLProcessingInstruction
    NAME = 'lxml-xml'
    ALTERNATE_NAMES = ['xml']
    features = [NAME, LXML, XML, FAST, PERMISSIVE]
    CHUNK_SIZE = 512
    DEFAULT_NSMAPS = dict(xml='http://www.w3.org/XML/1998/namespace')
    DEFAULT_NSMAPS_INVERTED = _invert(DEFAULT_NSMAPS)

    def initialize_soup(self, soup):
        if False:
            return 10
        'Let the BeautifulSoup object know about the standard namespace\n        mapping.\n\n        :param soup: A `BeautifulSoup`.\n        '
        super(LXMLTreeBuilderForXML, self).initialize_soup(soup)
        self._register_namespaces(self.DEFAULT_NSMAPS)

    def _register_namespaces(self, mapping):
        if False:
            return 10
        'Let the BeautifulSoup object know about namespaces encountered\n        while parsing the document.\n\n        This might be useful later on when creating CSS selectors.\n\n        This will track (almost) all namespaces, even ones that were\n        only in scope for part of the document. If two namespaces have\n        the same prefix, only the first one encountered will be\n        tracked. Un-prefixed namespaces are not tracked.\n\n        :param mapping: A dictionary mapping namespace prefixes to URIs.\n        '
        for (key, value) in list(mapping.items()):
            if key and key not in self.soup._namespaces:
                self.soup._namespaces[key] = value

    def default_parser(self, encoding):
        if False:
            for i in range(10):
                print('nop')
        'Find the default parser for the given encoding.\n\n        :param encoding: A string.\n        :return: Either a parser object or a class, which\n          will be instantiated with default arguments.\n        '
        if self._default_parser is not None:
            return self._default_parser
        return etree.XMLParser(target=self, strip_cdata=False, recover=True, encoding=encoding)

    def parser_for(self, encoding):
        if False:
            print('Hello World!')
        'Instantiate an appropriate parser for the given encoding.\n\n        :param encoding: A string.\n        :return: A parser object such as an `etree.XMLParser`.\n        '
        parser = self.default_parser(encoding)
        if isinstance(parser, Callable):
            parser = parser(target=self, strip_cdata=False, recover=True, encoding=encoding)
        return parser

    def __init__(self, parser=None, empty_element_tags=None, **kwargs):
        if False:
            return 10
        self._default_parser = parser
        if empty_element_tags is not None:
            self.empty_element_tags = set(empty_element_tags)
        self.soup = None
        self.nsmaps = [self.DEFAULT_NSMAPS_INVERTED]
        self.active_namespace_prefixes = [dict(self.DEFAULT_NSMAPS)]
        super(LXMLTreeBuilderForXML, self).__init__(**kwargs)

    def _getNsTag(self, tag):
        if False:
            for i in range(10):
                print('nop')
        if tag[0] == '{':
            return tuple(tag[1:].split('}', 1))
        else:
            return (None, tag)

    def prepare_markup(self, markup, user_specified_encoding=None, exclude_encodings=None, document_declared_encoding=None):
        if False:
            while True:
                i = 10
        'Run any preliminary steps necessary to make incoming markup\n        acceptable to the parser.\n\n        lxml really wants to get a bytestring and convert it to\n        Unicode itself. So instead of using UnicodeDammit to convert\n        the bytestring to Unicode using different encodings, this\n        implementation uses EncodingDetector to iterate over the\n        encodings, and tell lxml to try to parse the document as each\n        one in turn.\n\n        :param markup: Some markup -- hopefully a bytestring.\n        :param user_specified_encoding: The user asked to try this encoding.\n        :param document_declared_encoding: The markup itself claims to be\n            in this encoding.\n        :param exclude_encodings: The user asked _not_ to try any of\n            these encodings.\n\n        :yield: A series of 4-tuples:\n         (markup, encoding, declared encoding,\n          has undergone character replacement)\n\n         Each 4-tuple represents a strategy for converting the\n         document to Unicode and parsing it. Each strategy will be tried \n         in turn.\n        '
        is_html = not self.is_xml
        if is_html:
            self.processing_instruction_class = ProcessingInstruction
            DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(markup)
        else:
            self.processing_instruction_class = XMLProcessingInstruction
        if isinstance(markup, str):
            if len(markup) > 0 and markup[0] == u'\ufeff':
                markup = markup[1:]
            yield (markup, None, document_declared_encoding, False)
        if isinstance(markup, str):
            yield (markup.encode('utf8'), 'utf8', document_declared_encoding, False)
        known_definite_encodings = [user_specified_encoding]
        user_encodings = [document_declared_encoding]
        detector = EncodingDetector(markup, known_definite_encodings=known_definite_encodings, user_encodings=user_encodings, is_html=is_html, exclude_encodings=exclude_encodings)
        for encoding in detector.encodings:
            yield (detector.markup, encoding, document_declared_encoding, False)

    def feed(self, markup):
        if False:
            return 10
        if isinstance(markup, bytes):
            markup = BytesIO(markup)
        elif isinstance(markup, str):
            markup = StringIO(markup)
        data = markup.read(self.CHUNK_SIZE)
        try:
            self.parser = self.parser_for(self.soup.original_encoding)
            self.parser.feed(data)
            while len(data) != 0:
                data = markup.read(self.CHUNK_SIZE)
                if len(data) != 0:
                    self.parser.feed(data)
            self.parser.close()
        except (UnicodeDecodeError, LookupError, etree.ParserError) as e:
            raise ParserRejectedMarkup(e)

    def close(self):
        if False:
            return 10
        self.nsmaps = [self.DEFAULT_NSMAPS_INVERTED]

    def start(self, name, attrs, nsmap={}):
        if False:
            i = 10
            return i + 15
        attrs = dict(attrs)
        nsprefix = None
        if len(nsmap) == 0 and len(self.nsmaps) > 1:
            self.nsmaps.append(None)
        elif len(nsmap) > 0:
            self._register_namespaces(nsmap)
            self.nsmaps.append(_invert(nsmap))
            current_mapping = dict(self.active_namespace_prefixes[-1])
            current_mapping.update(nsmap)
            if '' in current_mapping:
                del current_mapping['']
            self.active_namespace_prefixes.append(current_mapping)
            attrs = attrs.copy()
            for (prefix, namespace) in list(nsmap.items()):
                attribute = NamespacedAttribute('xmlns', prefix, 'http://www.w3.org/2000/xmlns/')
                attrs[attribute] = namespace
        new_attrs = {}
        for (attr, value) in list(attrs.items()):
            (namespace, attr) = self._getNsTag(attr)
            if namespace is None:
                new_attrs[attr] = value
            else:
                nsprefix = self._prefix_for_namespace(namespace)
                attr = NamespacedAttribute(nsprefix, attr, namespace)
                new_attrs[attr] = value
        attrs = new_attrs
        (namespace, name) = self._getNsTag(name)
        nsprefix = self._prefix_for_namespace(namespace)
        self.soup.handle_starttag(name, namespace, nsprefix, attrs, namespaces=self.active_namespace_prefixes[-1])

    def _prefix_for_namespace(self, namespace):
        if False:
            while True:
                i = 10
        'Find the currently active prefix for the given namespace.'
        if namespace is None:
            return None
        for inverted_nsmap in reversed(self.nsmaps):
            if inverted_nsmap is not None and namespace in inverted_nsmap:
                return inverted_nsmap[namespace]
        return None

    def end(self, name):
        if False:
            print('Hello World!')
        self.soup.endData()
        completed_tag = self.soup.tagStack[-1]
        (namespace, name) = self._getNsTag(name)
        nsprefix = None
        if namespace is not None:
            for inverted_nsmap in reversed(self.nsmaps):
                if inverted_nsmap is not None and namespace in inverted_nsmap:
                    nsprefix = inverted_nsmap[namespace]
                    break
        self.soup.handle_endtag(name, nsprefix)
        if len(self.nsmaps) > 1:
            out_of_scope_nsmap = self.nsmaps.pop()
            if out_of_scope_nsmap is not None:
                self.active_namespace_prefixes.pop()

    def pi(self, target, data):
        if False:
            for i in range(10):
                print('nop')
        self.soup.endData()
        data = target + ' ' + data
        self.soup.handle_data(data)
        self.soup.endData(self.processing_instruction_class)

    def data(self, content):
        if False:
            for i in range(10):
                print('nop')
        self.soup.handle_data(content)

    def doctype(self, name, pubid, system):
        if False:
            print('Hello World!')
        self.soup.endData()
        doctype = Doctype.for_name_and_ids(name, pubid, system)
        self.soup.object_was_parsed(doctype)

    def comment(self, content):
        if False:
            return 10
        'Handle comments as Comment objects.'
        self.soup.endData()
        self.soup.handle_data(content)
        self.soup.endData(Comment)

    def test_fragment_to_document(self, fragment):
        if False:
            while True:
                i = 10
        'See `TreeBuilder`.'
        return '<?xml version="1.0" encoding="utf-8"?>\n%s' % fragment

class LXMLTreeBuilder(HTMLTreeBuilder, LXMLTreeBuilderForXML):
    NAME = LXML
    ALTERNATE_NAMES = ['lxml-html']
    features = ALTERNATE_NAMES + [NAME, HTML, FAST, PERMISSIVE]
    is_xml = False
    processing_instruction_class = ProcessingInstruction

    def default_parser(self, encoding):
        if False:
            while True:
                i = 10
        return etree.HTMLParser

    def feed(self, markup):
        if False:
            print('Hello World!')
        encoding = self.soup.original_encoding
        try:
            self.parser = self.parser_for(encoding)
            self.parser.feed(markup)
            self.parser.close()
        except (UnicodeDecodeError, LookupError, etree.ParserError) as e:
            raise ParserRejectedMarkup(e)

    def test_fragment_to_document(self, fragment):
        if False:
            for i in range(10):
                print('nop')
        'See `TreeBuilder`.'
        return '<html><body>%s</body></html>' % fragment