"""
Corpus reader for corpora whose documents are xml files.

(note -- not named 'xml' to avoid conflicting w/ standard xml package)
"""
import codecs
from xml.etree import ElementTree
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import *
from nltk.data import SeekableUnicodeStreamReader
from nltk.internals import ElementWrapper
from nltk.tokenize import WordPunctTokenizer

class XMLCorpusReader(CorpusReader):
    """
    Corpus reader for corpora whose documents are xml files.

    Note that the ``XMLCorpusReader`` constructor does not take an
    ``encoding`` argument, because the unicode encoding is specified by
    the XML files themselves.  See the XML specs for more info.
    """

    def __init__(self, root, fileids, wrap_etree=False):
        if False:
            print('Hello World!')
        self._wrap_etree = wrap_etree
        CorpusReader.__init__(self, root, fileids)

    def xml(self, fileid=None):
        if False:
            print('Hello World!')
        if fileid is None and len(self._fileids) == 1:
            fileid = self._fileids[0]
        if not isinstance(fileid, str):
            raise TypeError('Expected a single file identifier string')
        with self.abspath(fileid).open() as fp:
            elt = ElementTree.parse(fp).getroot()
        if self._wrap_etree:
            elt = ElementWrapper(elt)
        return elt

    def words(self, fileid=None):
        if False:
            while True:
                i = 10
        "\n        Returns all of the words and punctuation symbols in the specified file\n        that were in text nodes -- ie, tags are ignored. Like the xml() method,\n        fileid can only specify one file.\n\n        :return: the given file's text nodes as a list of words and punctuation symbols\n        :rtype: list(str)\n        "
        elt = self.xml(fileid)
        encoding = self.encoding(fileid)
        word_tokenizer = WordPunctTokenizer()
        try:
            iterator = elt.getiterator()
        except:
            iterator = elt.iter()
        out = []
        for node in iterator:
            text = node.text
            if text is not None:
                if isinstance(text, bytes):
                    text = text.decode(encoding)
                toks = word_tokenizer.tokenize(text)
                out.extend(toks)
        return out

class XMLCorpusView(StreamBackedCorpusView):
    """
    A corpus view that selects out specified elements from an XML
    file, and provides a flat list-like interface for accessing them.
    (Note: ``XMLCorpusView`` is not used by ``XMLCorpusReader`` itself,
    but may be used by subclasses of ``XMLCorpusReader``.)

    Every XML corpus view has a "tag specification", indicating what
    XML elements should be included in the view; and each (non-nested)
    element that matches this specification corresponds to one item in
    the view.  Tag specifications are regular expressions over tag
    paths, where a tag path is a list of element tag names, separated
    by '/', indicating the ancestry of the element.  Some examples:

      - ``'foo'``: A top-level element whose tag is ``foo``.
      - ``'foo/bar'``: An element whose tag is ``bar`` and whose parent
        is a top-level element whose tag is ``foo``.
      - ``'.*/foo'``: An element whose tag is ``foo``, appearing anywhere
        in the xml tree.
      - ``'.*/(foo|bar)'``: An wlement whose tag is ``foo`` or ``bar``,
        appearing anywhere in the xml tree.

    The view items are generated from the selected XML elements via
    the method ``handle_elt()``.  By default, this method returns the
    element as-is (i.e., as an ElementTree object); but it can be
    overridden, either via subclassing or via the ``elt_handler``
    constructor parameter.
    """
    _DEBUG = False
    _BLOCK_SIZE = 1024

    def __init__(self, fileid, tagspec, elt_handler=None):
        if False:
            print('Hello World!')
        '\n        Create a new corpus view based on a specified XML file.\n\n        Note that the ``XMLCorpusView`` constructor does not take an\n        ``encoding`` argument, because the unicode encoding is\n        specified by the XML files themselves.\n\n        :type tagspec: str\n        :param tagspec: A tag specification, indicating what XML\n            elements should be included in the view.  Each non-nested\n            element that matches this specification corresponds to one\n            item in the view.\n\n        :param elt_handler: A function used to transform each element\n            to a value for the view.  If no handler is specified, then\n            ``self.handle_elt()`` is called, which returns the element\n            as an ElementTree object.  The signature of elt_handler is::\n\n                elt_handler(elt, tagspec) -> value\n        '
        if elt_handler:
            self.handle_elt = elt_handler
        self._tagspec = re.compile(tagspec + '\\Z')
        'The tag specification for this corpus view.'
        self._tag_context = {0: ()}
        'A dictionary mapping from file positions (as returned by\n           ``stream.seek()`` to XML contexts.  An XML context is a\n           tuple of XML tag names, indicating which tags have not yet\n           been closed.'
        encoding = self._detect_encoding(fileid)
        StreamBackedCorpusView.__init__(self, fileid, encoding=encoding)

    def _detect_encoding(self, fileid):
        if False:
            i = 10
            return i + 15
        if isinstance(fileid, PathPointer):
            try:
                infile = fileid.open()
                s = infile.readline()
            finally:
                infile.close()
        else:
            with open(fileid, 'rb') as infile:
                s = infile.readline()
        if s.startswith(codecs.BOM_UTF16_BE):
            return 'utf-16-be'
        if s.startswith(codecs.BOM_UTF16_LE):
            return 'utf-16-le'
        if s.startswith(codecs.BOM_UTF32_BE):
            return 'utf-32-be'
        if s.startswith(codecs.BOM_UTF32_LE):
            return 'utf-32-le'
        if s.startswith(codecs.BOM_UTF8):
            return 'utf-8'
        m = re.match(b'\\s*<\\?xml\\b.*\\bencoding="([^"]+)"', s)
        if m:
            return m.group(1).decode()
        m = re.match(b"\\s*<\\?xml\\b.*\\bencoding='([^']+)'", s)
        if m:
            return m.group(1).decode()
        return 'utf-8'

    def handle_elt(self, elt, context):
        if False:
            i = 10
            return i + 15
        "\n        Convert an element into an appropriate value for inclusion in\n        the view.  Unless overridden by a subclass or by the\n        ``elt_handler`` constructor argument, this method simply\n        returns ``elt``.\n\n        :return: The view value corresponding to ``elt``.\n\n        :type elt: ElementTree\n        :param elt: The element that should be converted.\n\n        :type context: str\n        :param context: A string composed of element tags separated by\n            forward slashes, indicating the XML context of the given\n            element.  For example, the string ``'foo/bar/baz'``\n            indicates that the element is a ``baz`` element whose\n            parent is a ``bar`` element and whose grandparent is a\n            top-level ``foo`` element.\n        "
        return elt
    _VALID_XML_RE = re.compile('\n        [^<]*\n        (\n          ((<!--.*?-->)                         |  # comment\n           (<![CDATA[.*?]])                     |  # raw character data\n           (<!DOCTYPE\\s+[^\\[]*(\\[[^\\]]*])?\\s*>) |  # doctype decl\n           (<[^!>][^>]*>))                         # tag or PI\n          [^<]*)*\n        \\Z', re.DOTALL | re.VERBOSE)
    _XML_TAG_NAME = re.compile('<\\s*(?:/\\s*)?([^\\s>]+)')
    _XML_PIECE = re.compile('\n        # Include these so we can skip them:\n        (?P<COMMENT>        <!--.*?-->                          )|\n        (?P<CDATA>          <![CDATA[.*?]]>                     )|\n        (?P<PI>             <\\?.*?\\?>                           )|\n        (?P<DOCTYPE>        <!DOCTYPE\\s+[^\\[^>]*(\\[[^\\]]*])?\\s*>)|\n        # These are the ones we actually care about:\n        (?P<EMPTY_ELT_TAG>  <\\s*[^>/\\?!\\s][^>]*/\\s*>            )|\n        (?P<START_TAG>      <\\s*[^>/\\?!\\s][^>]*>                )|\n        (?P<END_TAG>        <\\s*/[^>/\\?!\\s][^>]*>               )', re.DOTALL | re.VERBOSE)

    def _read_xml_fragment(self, stream):
        if False:
            i = 10
            return i + 15
        "\n        Read a string from the given stream that does not contain any\n        un-closed tags.  In particular, this function first reads a\n        block from the stream of size ``self._BLOCK_SIZE``.  It then\n        checks if that block contains an un-closed tag.  If it does,\n        then this function either backtracks to the last '<', or reads\n        another block.\n        "
        fragment = ''
        if isinstance(stream, SeekableUnicodeStreamReader):
            startpos = stream.tell()
        while True:
            xml_block = stream.read(self._BLOCK_SIZE)
            fragment += xml_block
            if self._VALID_XML_RE.match(fragment):
                return fragment
            if re.search('[<>]', fragment).group(0) == '>':
                pos = stream.tell() - (len(fragment) - re.search('[<>]', fragment).end())
                raise ValueError('Unexpected ">" near char %s' % pos)
            if not xml_block:
                raise ValueError('Unexpected end of file: tag not closed')
            last_open_bracket = fragment.rfind('<')
            if last_open_bracket > 0:
                if self._VALID_XML_RE.match(fragment[:last_open_bracket]):
                    if isinstance(stream, SeekableUnicodeStreamReader):
                        stream.seek(startpos)
                        stream.char_seek_forward(last_open_bracket)
                    else:
                        stream.seek(-(len(fragment) - last_open_bracket), 1)
                    return fragment[:last_open_bracket]

    def read_block(self, stream, tagspec=None, elt_handler=None):
        if False:
            i = 10
            return i + 15
        '\n        Read from ``stream`` until we find at least one element that\n        matches ``tagspec``, and return the result of applying\n        ``elt_handler`` to each element found.\n        '
        if tagspec is None:
            tagspec = self._tagspec
        if elt_handler is None:
            elt_handler = self.handle_elt
        context = list(self._tag_context.get(stream.tell()))
        assert context is not None
        elts = []
        elt_start = None
        elt_depth = None
        elt_text = ''
        while elts == [] or elt_start is not None:
            if isinstance(stream, SeekableUnicodeStreamReader):
                startpos = stream.tell()
            xml_fragment = self._read_xml_fragment(stream)
            if not xml_fragment:
                if elt_start is None:
                    break
                else:
                    raise ValueError('Unexpected end of file')
            for piece in self._XML_PIECE.finditer(xml_fragment):
                if self._DEBUG:
                    print('{:>25} {}'.format('/'.join(context)[-20:], piece.group()))
                if piece.group('START_TAG'):
                    name = self._XML_TAG_NAME.match(piece.group()).group(1)
                    context.append(name)
                    if elt_start is None:
                        if re.match(tagspec, '/'.join(context)):
                            elt_start = piece.start()
                            elt_depth = len(context)
                elif piece.group('END_TAG'):
                    name = self._XML_TAG_NAME.match(piece.group()).group(1)
                    if not context:
                        raise ValueError('Unmatched tag </%s>' % name)
                    if name != context[-1]:
                        raise ValueError(f'Unmatched tag <{context[-1]}>...</{name}>')
                    if elt_start is not None and elt_depth == len(context):
                        elt_text += xml_fragment[elt_start:piece.end()]
                        elts.append((elt_text, '/'.join(context)))
                        elt_start = elt_depth = None
                        elt_text = ''
                    context.pop()
                elif piece.group('EMPTY_ELT_TAG'):
                    name = self._XML_TAG_NAME.match(piece.group()).group(1)
                    if elt_start is None:
                        if re.match(tagspec, '/'.join(context) + '/' + name):
                            elts.append((piece.group(), '/'.join(context) + '/' + name))
            if elt_start is not None:
                if elts == []:
                    elt_text += xml_fragment[elt_start:]
                    elt_start = 0
                else:
                    if self._DEBUG:
                        print(' ' * 36 + '(backtrack)')
                    if isinstance(stream, SeekableUnicodeStreamReader):
                        stream.seek(startpos)
                        stream.char_seek_forward(elt_start)
                    else:
                        stream.seek(-(len(xml_fragment) - elt_start), 1)
                    context = context[:elt_depth - 1]
                    elt_start = elt_depth = None
                    elt_text = ''
        pos = stream.tell()
        if pos in self._tag_context:
            assert tuple(context) == self._tag_context[pos]
        else:
            self._tag_context[pos] = tuple(context)
        return [elt_handler(ElementTree.fromstring(elt.encode('ascii', 'xmlcharrefreplace')), context) for (elt, context) in elts]