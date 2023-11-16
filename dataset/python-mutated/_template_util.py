"""
twisted.web.util and twisted.web.template merged to avoid cyclic deps
"""
import io
import linecache
import warnings
from collections import OrderedDict
from html import escape
from typing import IO, Any, AnyStr, Callable, Dict, List, Mapping, Optional, Tuple, Union, cast
from xml.sax import handler, make_parser
from xml.sax.xmlreader import Locator
from zope.interface import implementer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import urlpath
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import fullyQualifiedName
from twisted.web import resource
from twisted.web._element import Element, renderer
from twisted.web._flatten import Flattenable, flatten, flattenString
from twisted.web._stan import CDATA, Comment, Tag, slot
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader

def _PRE(text):
    if False:
        i = 10
        return i + 15
    '\n    Wraps <pre> tags around some text and HTML-escape it.\n\n    This is here since once twisted.web.html was deprecated it was hard to\n    migrate the html.PRE from current code to twisted.web.template.\n\n    For new code consider using twisted.web.template.\n\n    @return: Escaped text wrapped in <pre> tags.\n    @rtype: C{str}\n    '
    return f'<pre>{escape(text)}</pre>'

def redirectTo(URL: bytes, request: IRequest) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate a redirect to the given location.\n\n    @param URL: A L{bytes} giving the location to which to redirect.\n\n    @param request: The request object to use to generate the redirect.\n    @type request: L{IRequest<twisted.web.iweb.IRequest>} provider\n\n    @raise TypeError: If the type of C{URL} a L{str} instead of L{bytes}.\n\n    @return: A L{bytes} containing HTML which tries to convince the client\n        agent\n        to visit the new location even if it doesn\'t respect the I{FOUND}\n        response code.  This is intended to be returned from a render method,\n        eg::\n\n            def render_GET(self, request):\n                return redirectTo(b"http://example.com/", request)\n    '
    if not isinstance(URL, bytes):
        raise TypeError('URL must be bytes')
    request.setHeader(b'Content-Type', b'text/html; charset=utf-8')
    request.redirect(URL)
    content = b'\n<html>\n    <head>\n        <meta http-equiv="refresh" content="0;URL=%(url)s">\n    </head>\n    <body bgcolor="#FFFFFF" text="#000000">\n    <a href="%(url)s">click here</a>\n    </body>\n</html>\n' % {b'url': URL}
    return content

class Redirect(resource.Resource):
    """
    Resource that redirects to a specific URL.

    @ivar url: Redirect target URL to put in the I{Location} response header.
    @type url: L{bytes}
    """
    isLeaf = True

    def __init__(self, url: bytes):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.url = url

    def render(self, request):
        if False:
            for i in range(10):
                print('nop')
        return redirectTo(self.url, request)

    def getChild(self, name, request):
        if False:
            print('Hello World!')
        return self

class ChildRedirector(Redirect):
    isLeaf = False

    def __init__(self, url):
        if False:
            while True:
                i = 10
        if url.find('://') == -1 and (not url.startswith('..')) and (not url.startswith('/')):
            raise ValueError("It seems you've given me a redirect (%s) that is a child of myself! That's not good, it'll cause an infinite redirect." % url)
        Redirect.__init__(self, url)

    def getChild(self, name, request):
        if False:
            print('Hello World!')
        newUrl = self.url
        if not newUrl.endswith('/'):
            newUrl += '/'
        newUrl += name
        return ChildRedirector(newUrl)

class ParentRedirect(resource.Resource):
    """
    Redirect to the nearest directory and strip any query string.

    This generates redirects like::

        /              →  /
        /foo           →  /
        /foo?bar       →  /
        /foo/          →  /foo/
        /foo/bar       →  /foo/
        /foo/bar?baz   →  /foo/

    However, the generated I{Location} header contains an absolute URL rather
    than a path.

    The response is the same regardless of HTTP method.
    """
    isLeaf = 1

    def render(self, request: IRequest) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Respond to all requests by redirecting to nearest directory.\n        '
        here = str(urlpath.URLPath.fromRequest(request).here()).encode('ascii')
        return redirectTo(here, request)

class DeferredResource(resource.Resource):
    """
    I wrap up a Deferred that will eventually result in a Resource
    object.
    """
    isLeaf = 1

    def __init__(self, d):
        if False:
            for i in range(10):
                print('nop')
        resource.Resource.__init__(self)
        self.d = d

    def getChild(self, name, request):
        if False:
            print('Hello World!')
        return self

    def render(self, request):
        if False:
            return 10
        self.d.addCallback(self._cbChild, request).addErrback(self._ebChild, request)
        from twisted.web.server import NOT_DONE_YET
        return NOT_DONE_YET

    def _cbChild(self, child, request):
        if False:
            return 10
        request.render(resource.getChildForRequest(child, request))

    def _ebChild(self, reason, request):
        if False:
            print('Hello World!')
        request.processingFailed(reason)

class _SourceLineElement(Element):
    """
    L{_SourceLineElement} is an L{IRenderable} which can render a single line of
    source code.

    @ivar number: A C{int} giving the line number of the source code to be
        rendered.
    @ivar source: A C{str} giving the source code to be rendered.
    """

    def __init__(self, loader, number, source):
        if False:
            while True:
                i = 10
        Element.__init__(self, loader)
        self.number = number
        self.source = source

    @renderer
    def sourceLine(self, request, tag):
        if False:
            print('Hello World!')
        '\n        Render the line of source as a child of C{tag}.\n        '
        return tag(self.source.replace('  ', ' \xa0'))

    @renderer
    def lineNumber(self, request, tag):
        if False:
            while True:
                i = 10
        '\n        Render the line number as a child of C{tag}.\n        '
        return tag(str(self.number))

class _SourceFragmentElement(Element):
    """
    L{_SourceFragmentElement} is an L{IRenderable} which can render several lines
    of source code near the line number of a particular frame object.

    @ivar frame: A L{Failure<twisted.python.failure.Failure>}-style frame object
        for which to load a source line to render.  This is really a tuple
        holding some information from a frame object.  See
        L{Failure.frames<twisted.python.failure.Failure>} for specifics.
    """

    def __init__(self, loader, frame):
        if False:
            i = 10
            return i + 15
        Element.__init__(self, loader)
        self.frame = frame

    def _getSourceLines(self):
        if False:
            while True:
                i = 10
        '\n        Find the source line references by C{self.frame} and yield, in source\n        line order, it and the previous and following lines.\n\n        @return: A generator which yields two-tuples.  Each tuple gives a source\n            line number and the contents of that source line.\n        '
        filename = self.frame[1]
        lineNumber = self.frame[2]
        for snipLineNumber in range(lineNumber - 1, lineNumber + 2):
            yield (snipLineNumber, linecache.getline(filename, snipLineNumber).rstrip())

    @renderer
    def sourceLines(self, request, tag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Render the source line indicated by C{self.frame} and several\n        surrounding lines.  The active line will be given a I{class} of\n        C{"snippetHighlightLine"}.  Other lines will be given a I{class} of\n        C{"snippetLine"}.\n        '
        for (lineNumber, sourceLine) in self._getSourceLines():
            newTag = tag.clone()
            if lineNumber == self.frame[2]:
                cssClass = 'snippetHighlightLine'
            else:
                cssClass = 'snippetLine'
            loader = TagLoader(newTag(**{'class': cssClass}))
            yield _SourceLineElement(loader, lineNumber, sourceLine)

class _FrameElement(Element):
    """
    L{_FrameElement} is an L{IRenderable} which can render details about one
    frame from a L{Failure<twisted.python.failure.Failure>}.

    @ivar frame: A L{Failure<twisted.python.failure.Failure>}-style frame object
        for which to load a source line to render.  This is really a tuple
        holding some information from a frame object.  See
        L{Failure.frames<twisted.python.failure.Failure>} for specifics.
    """

    def __init__(self, loader, frame):
        if False:
            while True:
                i = 10
        Element.__init__(self, loader)
        self.frame = frame

    @renderer
    def filename(self, request, tag):
        if False:
            while True:
                i = 10
        '\n        Render the name of the file this frame references as a child of C{tag}.\n        '
        return tag(self.frame[1])

    @renderer
    def lineNumber(self, request, tag):
        if False:
            i = 10
            return i + 15
        '\n        Render the source line number this frame references as a child of\n        C{tag}.\n        '
        return tag(str(self.frame[2]))

    @renderer
    def function(self, request, tag):
        if False:
            while True:
                i = 10
        '\n        Render the function name this frame references as a child of C{tag}.\n        '
        return tag(self.frame[0])

    @renderer
    def source(self, request, tag):
        if False:
            return 10
        '\n        Render the source code surrounding the line this frame references,\n        replacing C{tag}.\n        '
        return _SourceFragmentElement(TagLoader(tag), self.frame)

class _StackElement(Element):
    """
    L{_StackElement} renders an L{IRenderable} which can render a list of frames.
    """

    def __init__(self, loader, stackFrames):
        if False:
            print('Hello World!')
        Element.__init__(self, loader)
        self.stackFrames = stackFrames

    @renderer
    def frames(self, request, tag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Render the list of frames in this L{_StackElement}, replacing C{tag}.\n        '
        return [_FrameElement(TagLoader(tag.clone()), frame) for frame in self.stackFrames]

class _NSContext:
    """
    A mapping from XML namespaces onto their prefixes in the document.
    """

    def __init__(self, parent: Optional['_NSContext']=None):
        if False:
            return 10
        "\n        Pull out the parent's namespaces, if there's no parent then default to\n        XML.\n        "
        self.parent = parent
        if parent is not None:
            self.nss: Dict[Optional[str], Optional[str]] = OrderedDict(parent.nss)
        else:
            self.nss = {'http://www.w3.org/XML/1998/namespace': 'xml'}

    def get(self, k: Optional[str], d: Optional[str]=None) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Get a prefix for a namespace.\n\n        @param d: The default prefix value.\n        '
        return self.nss.get(k, d)

    def __setitem__(self, k: Optional[str], v: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Proxy through to setting the prefix for the namespace.\n        '
        self.nss.__setitem__(k, v)

    def __getitem__(self, k: Optional[str]) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        Proxy through to getting the prefix for the namespace.\n        '
        return self.nss.__getitem__(k)
TEMPLATE_NAMESPACE = 'http://twistedmatrix.com/ns/twisted.web.template/0.1'

class _ToStan(handler.ContentHandler, handler.EntityResolver):
    """
    A SAX parser which converts an XML document to the Twisted STAN
    Document Object Model.
    """

    def __init__(self, sourceFilename: Optional[str]):
        if False:
            return 10
        '\n        @param sourceFilename: the filename the XML was loaded out of.\n        '
        self.sourceFilename = sourceFilename
        self.prefixMap = _NSContext()
        self.inCDATA = False

    def setDocumentLocator(self, locator: Locator) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the document locator, which knows about line and character numbers.\n        '
        self.locator = locator

    def startDocument(self) -> None:
        if False:
            print('Hello World!')
        '\n        Initialise the document.\n        '
        self.document: List[Any] = []
        self.current = self.document
        self.stack: List[Any] = []
        self.xmlnsAttrs: List[Tuple[str, str]] = []

    def endDocument(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Document ended.\n        '

    def processingInstruction(self, target: str, data: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Processing instructions are ignored.\n        '

    def startPrefixMapping(self, prefix: Optional[str], uri: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set up the prefix mapping, which maps fully qualified namespace URIs\n        onto namespace prefixes.\n\n        This gets called before startElementNS whenever an C{xmlns} attribute\n        is seen.\n        '
        self.prefixMap = _NSContext(self.prefixMap)
        self.prefixMap[uri] = prefix
        if uri == TEMPLATE_NAMESPACE:
            return
        if prefix is None:
            self.xmlnsAttrs.append(('xmlns', uri))
        else:
            self.xmlnsAttrs.append(('xmlns:%s' % prefix, uri))

    def endPrefixMapping(self, prefix: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        "Pops the stack" on the prefix mapping.\n\n        Gets called after endElementNS.\n        '
        parent = self.prefixMap.parent
        assert parent is not None, 'More prefix mapping ends than starts'
        self.prefixMap = parent

    def startElementNS(self, namespaceAndName: Tuple[str, str], qname: Optional[str], attrs: Mapping[Tuple[Optional[str], str], str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets called when we encounter a new xmlns attribute.\n\n        @param namespaceAndName: a (namespace, name) tuple, where name\n            determines which type of action to take, if the namespace matches\n            L{TEMPLATE_NAMESPACE}.\n        @param qname: ignored.\n        @param attrs: attributes on the element being started.\n        '
        filename = self.sourceFilename
        lineNumber = self.locator.getLineNumber()
        columnNumber = self.locator.getColumnNumber()
        (ns, name) = namespaceAndName
        if ns == TEMPLATE_NAMESPACE:
            if name == 'transparent':
                name = ''
            elif name == 'slot':
                default: Optional[str]
                try:
                    default = attrs[None, 'default']
                except KeyError:
                    default = None
                sl = slot(attrs[None, 'name'], default=default, filename=filename, lineNumber=lineNumber, columnNumber=columnNumber)
                self.stack.append(sl)
                self.current.append(sl)
                self.current = sl.children
                return
        render = None
        attrs = OrderedDict(attrs)
        for (k, v) in list(attrs.items()):
            (attrNS, justTheName) = k
            if attrNS != TEMPLATE_NAMESPACE:
                continue
            if justTheName == 'render':
                render = v
                del attrs[k]
        nonTemplateAttrs = OrderedDict()
        for ((attrNs, attrName), v) in attrs.items():
            nsPrefix = self.prefixMap.get(attrNs)
            if nsPrefix is None:
                attrKey = attrName
            else:
                attrKey = f'{nsPrefix}:{attrName}'
            nonTemplateAttrs[attrKey] = v
        if ns == TEMPLATE_NAMESPACE and name == 'attr':
            if not self.stack:
                raise AssertionError(f'<{{{TEMPLATE_NAMESPACE}}}attr> as top-level element')
            if 'name' not in nonTemplateAttrs:
                raise AssertionError(f'<{{{TEMPLATE_NAMESPACE}}}attr> requires a name attribute')
            el = Tag('', render=render, filename=filename, lineNumber=lineNumber, columnNumber=columnNumber)
            self.stack[-1].attributes[nonTemplateAttrs['name']] = el
            self.stack.append(el)
            self.current = el.children
            return
        if self.xmlnsAttrs:
            nonTemplateAttrs.update(OrderedDict(self.xmlnsAttrs))
            self.xmlnsAttrs = []
        if ns != TEMPLATE_NAMESPACE and ns is not None:
            prefix = self.prefixMap[ns]
            if prefix is not None:
                name = f'{self.prefixMap[ns]}:{name}'
        el = Tag(name, attributes=OrderedDict(cast(Mapping[Union[bytes, str], str], nonTemplateAttrs)), render=render, filename=filename, lineNumber=lineNumber, columnNumber=columnNumber)
        self.stack.append(el)
        self.current.append(el)
        self.current = el.children

    def characters(self, ch: str) -> None:
        if False:
            print('Hello World!')
        '\n        Called when we receive some characters.  CDATA characters get passed\n        through as is.\n        '
        if self.inCDATA:
            self.stack[-1].append(ch)
            return
        self.current.append(ch)

    def endElementNS(self, name: Tuple[str, str], qname: Optional[str]) -> None:
        if False:
            while True:
                i = 10
        "\n        A namespace tag is closed.  Pop the stack, if there's anything left in\n        it, otherwise return to the document's namespace.\n        "
        self.stack.pop()
        if self.stack:
            self.current = self.stack[-1].children
        else:
            self.current = self.document

    def startDTD(self, name: str, publicId: str, systemId: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        DTDs are ignored.\n        '

    def endDTD(self, *args: object) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        DTDs are ignored.\n        '

    def startCDATA(self) -> None:
        if False:
            print('Hello World!')
        "\n        We're starting to be in a CDATA element, make a note of this.\n        "
        self.inCDATA = True
        self.stack.append([])

    def endCDATA(self) -> None:
        if False:
            print('Hello World!')
        "\n        We're no longer in a CDATA element.  Collect up the characters we've\n        parsed and put them in a new CDATA object.\n        "
        self.inCDATA = False
        comment = ''.join(self.stack.pop())
        self.current.append(CDATA(comment))

    def comment(self, content: str) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Add an XML comment which we've encountered.\n        "
        self.current.append(Comment(content))

def _flatsaxParse(fl: Union[IO[AnyStr], str]) -> List['Flattenable']:
    if False:
        print('Hello World!')
    '\n    Perform a SAX parse of an XML document with the _ToStan class.\n\n    @param fl: The XML document to be parsed.\n\n    @return: a C{list} of Stan objects.\n    '
    parser = make_parser()
    parser.setFeature(handler.feature_validation, 0)
    parser.setFeature(handler.feature_namespaces, 1)
    parser.setFeature(handler.feature_external_ges, 0)
    parser.setFeature(handler.feature_external_pes, 0)
    s = _ToStan(getattr(fl, 'name', None))
    parser.setContentHandler(s)
    parser.setEntityResolver(s)
    parser.setProperty(handler.property_lexical_handler, s)
    parser.parse(fl)
    return s.document

@implementer(ITemplateLoader)
class XMLString:
    """
    An L{ITemplateLoader} that loads and parses XML from a string.
    """

    def __init__(self, s: Union[str, bytes]):
        if False:
            print('Hello World!')
        '\n        Run the parser on a L{io.StringIO} copy of the string.\n\n        @param s: The string from which to load the XML.\n        @type s: L{str}, or a UTF-8 encoded L{bytes}.\n        '
        if not isinstance(s, str):
            s = s.decode('utf8')
        self._loadedTemplate: List['Flattenable'] = _flatsaxParse(io.StringIO(s))
        'The loaded document.'

    def load(self) -> List['Flattenable']:
        if False:
            print('Hello World!')
        '\n        Return the document.\n\n        @return: the loaded document.\n        '
        return self._loadedTemplate

class FailureElement(Element):
    """
    L{FailureElement} is an L{IRenderable} which can render detailed information
    about a L{Failure<twisted.python.failure.Failure>}.

    @ivar failure: The L{Failure<twisted.python.failure.Failure>} instance which
        will be rendered.

    @since: 12.1
    """
    loader = XMLString('\n<div xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1">\n  <style type="text/css">\n    div.error {\n      color: red;\n      font-family: Verdana, Arial, helvetica, sans-serif;\n      font-weight: bold;\n    }\n\n    div {\n      font-family: Verdana, Arial, helvetica, sans-serif;\n    }\n\n    div.stackTrace {\n    }\n\n    div.frame {\n      padding: 1em;\n      background: white;\n      border-bottom: thin black dashed;\n    }\n\n    div.frame:first-child {\n      padding: 1em;\n      background: white;\n      border-top: thin black dashed;\n      border-bottom: thin black dashed;\n    }\n\n    div.location {\n    }\n\n    span.function {\n      font-weight: bold;\n      font-family: "Courier New", courier, monospace;\n    }\n\n    div.snippet {\n      margin-bottom: 0.5em;\n      margin-left: 1em;\n      background: #FFFFDD;\n    }\n\n    div.snippetHighlightLine {\n      color: red;\n    }\n\n    span.code {\n      font-family: "Courier New", courier, monospace;\n    }\n  </style>\n\n  <div class="error">\n    <span t:render="type" />: <span t:render="value" />\n  </div>\n  <div class="stackTrace" t:render="traceback">\n    <div class="frame" t:render="frames">\n      <div class="location">\n        <span t:render="filename" />:<span t:render="lineNumber" /> in\n        <span class="function" t:render="function" />\n      </div>\n      <div class="snippet" t:render="source">\n        <div t:render="sourceLines">\n          <span class="lineno" t:render="lineNumber" />\n          <code class="code" t:render="sourceLine" />\n        </div>\n      </div>\n    </div>\n  </div>\n  <div class="error">\n    <span t:render="type" />: <span t:render="value" />\n  </div>\n</div>\n')

    def __init__(self, failure, loader=None):
        if False:
            while True:
                i = 10
        Element.__init__(self, loader)
        self.failure = failure

    @renderer
    def type(self, request, tag):
        if False:
            return 10
        '\n        Render the exception type as a child of C{tag}.\n        '
        return tag(fullyQualifiedName(self.failure.type))

    @renderer
    def value(self, request, tag):
        if False:
            i = 10
            return i + 15
        '\n        Render the exception value as a child of C{tag}.\n        '
        return tag(str(self.failure.value).encode('utf8'))

    @renderer
    def traceback(self, request, tag):
        if False:
            i = 10
            return i + 15
        "\n        Render all the frames in the wrapped\n        L{Failure<twisted.python.failure.Failure>}'s traceback stack, replacing\n        C{tag}.\n        "
        return _StackElement(TagLoader(tag), self.failure.frames)

def formatFailure(myFailure):
    if False:
        while True:
            i = 10
    '\n    Construct an HTML representation of the given failure.\n\n    Consider using L{FailureElement} instead.\n\n    @type myFailure: L{Failure<twisted.python.failure.Failure>}\n\n    @rtype: L{bytes}\n    @return: A string containing the HTML representation of the given failure.\n    '
    result = []
    flattenString(None, FailureElement(myFailure)).addBoth(result.append)
    if isinstance(result[0], bytes):
        return result[0].decode('utf-8').encode('ascii', 'xmlcharrefreplace')
    result[0].raiseException()
NOT_DONE_YET = 1
_moduleLog = Logger()

@implementer(ITemplateLoader)
class TagLoader:
    """
    An L{ITemplateLoader} that loads an existing flattenable object.
    """

    def __init__(self, tag: 'Flattenable'):
        if False:
            print('Hello World!')
        '\n        @param tag: The object which will be loaded.\n        '
        self.tag: 'Flattenable' = tag
        'The object which will be loaded.'

    def load(self) -> List['Flattenable']:
        if False:
            print('Hello World!')
        return [self.tag]

@implementer(ITemplateLoader)
class XMLFile:
    """
    An L{ITemplateLoader} that loads and parses XML from a file.
    """

    def __init__(self, path: FilePath[Any]):
        if False:
            return 10
        '\n        Run the parser on a file.\n\n        @param path: The file from which to load the XML.\n        '
        if not isinstance(path, FilePath):
            warnings.warn('Passing filenames or file objects to XMLFile is deprecated since Twisted 12.1.  Pass a FilePath instead.', category=DeprecationWarning, stacklevel=2)
        self._loadedTemplate: Optional[List['Flattenable']] = None
        'The loaded document, or L{None}, if not loaded.'
        self._path: FilePath[Any] = path
        'The file that is being loaded from.'

    def _loadDoc(self) -> List['Flattenable']:
        if False:
            print('Hello World!')
        '\n        Read and parse the XML.\n\n        @return: the loaded document.\n        '
        if not isinstance(self._path, FilePath):
            return _flatsaxParse(self._path)
        else:
            with self._path.open('r') as f:
                return _flatsaxParse(f)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<XMLFile of {self._path!r}>'

    def load(self) -> List['Flattenable']:
        if False:
            while True:
                i = 10
        '\n        Return the document, first loading it if necessary.\n\n        @return: the loaded document.\n        '
        if self._loadedTemplate is None:
            self._loadedTemplate = self._loadDoc()
        return self._loadedTemplate
VALID_HTML_TAG_NAMES = {'a', 'abbr', 'acronym', 'address', 'applet', 'area', 'article', 'aside', 'audio', 'b', 'base', 'basefont', 'bdi', 'bdo', 'big', 'blockquote', 'body', 'br', 'button', 'canvas', 'caption', 'center', 'cite', 'code', 'col', 'colgroup', 'command', 'datalist', 'dd', 'del', 'details', 'dfn', 'dir', 'div', 'dl', 'dt', 'em', 'embed', 'fieldset', 'figcaption', 'figure', 'font', 'footer', 'form', 'frame', 'frameset', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'header', 'hgroup', 'hr', 'html', 'i', 'iframe', 'img', 'input', 'ins', 'isindex', 'keygen', 'kbd', 'label', 'legend', 'li', 'link', 'map', 'mark', 'menu', 'meta', 'meter', 'nav', 'noframes', 'noscript', 'object', 'ol', 'optgroup', 'option', 'output', 'p', 'param', 'pre', 'progress', 'q', 'rp', 'rt', 'ruby', 's', 'samp', 'script', 'section', 'select', 'small', 'source', 'span', 'strike', 'strong', 'style', 'sub', 'summary', 'sup', 'table', 'tbody', 'td', 'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr', 'tt', 'u', 'ul', 'var', 'video', 'wbr'}

class _TagFactory:
    """
    A factory for L{Tag} objects; the implementation of the L{tags} object.

    This allows for the syntactic convenience of C{from twisted.web.template
    import tags; tags.a(href="linked-page.html")}, where 'a' can be basically
    any HTML tag.

    The class is not exposed publicly because you only ever need one of these,
    and we already made it for you.

    @see: L{tags}
    """

    def __getattr__(self, tagName: str) -> Tag:
        if False:
            print('Hello World!')
        if tagName == 'transparent':
            return Tag('')
        tagName = tagName.rstrip('_')
        if tagName not in VALID_HTML_TAG_NAMES:
            raise AttributeError(f'unknown tag {tagName!r}')
        return Tag(tagName)
tags = _TagFactory()

def renderElement(request: IRequest, element: IRenderable, doctype: Optional[bytes]=b'<!DOCTYPE html>', _failElement: Optional[Callable[[Failure], 'Element']]=None) -> object:
    if False:
        i = 10
        return i + 15
    "\n    Render an element or other L{IRenderable}.\n\n    @param request: The L{IRequest} being rendered to.\n    @param element: An L{IRenderable} which will be rendered.\n    @param doctype: A L{bytes} which will be written as the first line of\n        the request, or L{None} to disable writing of a doctype.  The argument\n        should not include a trailing newline and will default to the HTML5\n        doctype C{'<!DOCTYPE html>'}.\n\n    @returns: NOT_DONE_YET\n\n    @since: 12.1\n    "
    if doctype is not None:
        request.write(doctype)
        request.write(b'\n')
    if _failElement is None:
        _failElement = FailureElement
    d = flatten(request, element, request.write)

    def eb(failure: Failure) -> Optional[Deferred[None]]:
        if False:
            while True:
                i = 10
        _moduleLog.failure('An error occurred while rendering the response.', failure=failure)
        site = getattr(request, 'site', None)
        if site is not None and site.displayTracebacks:
            assert _failElement is not None
            return flatten(request, _failElement(failure), request.write)
        else:
            request.write(b'<div style="font-size:800%;background-color:#FFF;color:#F00">An error occurred while rendering the response.</div>')
            return None

    def finish(result: object, *, request: IRequest=request) -> object:
        if False:
            print('Hello World!')
        request.finish()
        return result
    d.addErrback(eb)
    d.addBoth(finish)
    return NOT_DONE_YET