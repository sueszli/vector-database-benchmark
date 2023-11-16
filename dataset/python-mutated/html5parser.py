from __future__ import absolute_import, division, unicode_literals
from pip._vendor.six import with_metaclass, viewkeys
import types
from collections import OrderedDict
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import spaceCharacters, asciiUpper2Lower, specialElements, headingElements, cdataElements, rcdataElements, tokenTypes, tagTokenTypes, namespaces, htmlIntegrationPointElements, mathmlTextIntegrationPointElements, adjustForeignAttributes as adjustForeignAttributesMap, adjustMathMLAttributes, adjustSVGAttributes, E, _ReparseException

def parse(doc, treebuilder='etree', namespaceHTMLElements=True, **kwargs):
    if False:
        while True:
            i = 10
    "Parse an HTML document as a string or file-like object into a tree\n\n    :arg doc: the document to parse as a string or file-like object\n\n    :arg treebuilder: the treebuilder to use when parsing\n\n    :arg namespaceHTMLElements: whether or not to namespace HTML elements\n\n    :returns: parsed tree\n\n    Example:\n\n    >>> from html5lib.html5parser import parse\n    >>> parse('<html><body><p>This is a doc</p></body></html>')\n    <Element u'{http://www.w3.org/1999/xhtml}html' at 0x7feac4909db0>\n\n    "
    tb = treebuilders.getTreeBuilder(treebuilder)
    p = HTMLParser(tb, namespaceHTMLElements=namespaceHTMLElements)
    return p.parse(doc, **kwargs)

def parseFragment(doc, container='div', treebuilder='etree', namespaceHTMLElements=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "Parse an HTML fragment as a string or file-like object into a tree\n\n    :arg doc: the fragment to parse as a string or file-like object\n\n    :arg container: the container context to parse the fragment in\n\n    :arg treebuilder: the treebuilder to use when parsing\n\n    :arg namespaceHTMLElements: whether or not to namespace HTML elements\n\n    :returns: parsed tree\n\n    Example:\n\n    >>> from html5lib.html5libparser import parseFragment\n    >>> parseFragment('<b>this is a fragment</b>')\n    <Element u'DOCUMENT_FRAGMENT' at 0x7feac484b090>\n\n    "
    tb = treebuilders.getTreeBuilder(treebuilder)
    p = HTMLParser(tb, namespaceHTMLElements=namespaceHTMLElements)
    return p.parseFragment(doc, container=container, **kwargs)

def method_decorator_metaclass(function):
    if False:
        print('Hello World!')

    class Decorated(type):

        def __new__(meta, classname, bases, classDict):
            if False:
                print('Hello World!')
            for (attributeName, attribute) in classDict.items():
                if isinstance(attribute, types.FunctionType):
                    attribute = function(attribute)
                classDict[attributeName] = attribute
            return type.__new__(meta, classname, bases, classDict)
    return Decorated

class HTMLParser(object):
    """HTML parser

    Generates a tree structure from a stream of (possibly malformed) HTML.

    """

    def __init__(self, tree=None, strict=False, namespaceHTMLElements=True, debug=False):
        if False:
            i = 10
            return i + 15
        "\n        :arg tree: a treebuilder class controlling the type of tree that will be\n            returned. Built in treebuilders can be accessed through\n            html5lib.treebuilders.getTreeBuilder(treeType)\n\n        :arg strict: raise an exception when a parse error is encountered\n\n        :arg namespaceHTMLElements: whether or not to namespace HTML elements\n\n        :arg debug: whether or not to enable debug mode which logs things\n\n        Example:\n\n        >>> from html5lib.html5parser import HTMLParser\n        >>> parser = HTMLParser()                     # generates parser with etree builder\n        >>> parser = HTMLParser('lxml', strict=True)  # generates parser with lxml builder which is strict\n\n        "
        self.strict = strict
        if tree is None:
            tree = treebuilders.getTreeBuilder('etree')
        self.tree = tree(namespaceHTMLElements)
        self.errors = []
        self.phases = dict([(name, cls(self, self.tree)) for (name, cls) in getPhases(debug).items()])

    def _parse(self, stream, innerHTML=False, container='div', scripting=False, **kwargs):
        if False:
            print('Hello World!')
        self.innerHTMLMode = innerHTML
        self.container = container
        self.scripting = scripting
        self.tokenizer = _tokenizer.HTMLTokenizer(stream, parser=self, **kwargs)
        self.reset()
        try:
            self.mainLoop()
        except _ReparseException:
            self.reset()
            self.mainLoop()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.tree.reset()
        self.firstStartTag = False
        self.errors = []
        self.log = []
        self.compatMode = 'no quirks'
        if self.innerHTMLMode:
            self.innerHTML = self.container.lower()
            if self.innerHTML in cdataElements:
                self.tokenizer.state = self.tokenizer.rcdataState
            elif self.innerHTML in rcdataElements:
                self.tokenizer.state = self.tokenizer.rawtextState
            elif self.innerHTML == 'plaintext':
                self.tokenizer.state = self.tokenizer.plaintextState
            else:
                pass
            self.phase = self.phases['beforeHtml']
            self.phase.insertHtmlElement()
            self.resetInsertionMode()
        else:
            self.innerHTML = False
            self.phase = self.phases['initial']
        self.lastPhase = None
        self.beforeRCDataPhase = None
        self.framesetOK = True

    @property
    def documentEncoding(self):
        if False:
            print('Hello World!')
        'Name of the character encoding that was used to decode the input stream, or\n        :obj:`None` if that is not determined yet\n\n        '
        if not hasattr(self, 'tokenizer'):
            return None
        return self.tokenizer.stream.charEncoding[0].name

    def isHTMLIntegrationPoint(self, element):
        if False:
            i = 10
            return i + 15
        if element.name == 'annotation-xml' and element.namespace == namespaces['mathml']:
            return 'encoding' in element.attributes and element.attributes['encoding'].translate(asciiUpper2Lower) in ('text/html', 'application/xhtml+xml')
        else:
            return (element.namespace, element.name) in htmlIntegrationPointElements

    def isMathMLTextIntegrationPoint(self, element):
        if False:
            print('Hello World!')
        return (element.namespace, element.name) in mathmlTextIntegrationPointElements

    def mainLoop(self):
        if False:
            for i in range(10):
                print('nop')
        CharactersToken = tokenTypes['Characters']
        SpaceCharactersToken = tokenTypes['SpaceCharacters']
        StartTagToken = tokenTypes['StartTag']
        EndTagToken = tokenTypes['EndTag']
        CommentToken = tokenTypes['Comment']
        DoctypeToken = tokenTypes['Doctype']
        ParseErrorToken = tokenTypes['ParseError']
        for token in self.normalizedTokens():
            prev_token = None
            new_token = token
            while new_token is not None:
                prev_token = new_token
                currentNode = self.tree.openElements[-1] if self.tree.openElements else None
                currentNodeNamespace = currentNode.namespace if currentNode else None
                currentNodeName = currentNode.name if currentNode else None
                type = new_token['type']
                if type == ParseErrorToken:
                    self.parseError(new_token['data'], new_token.get('datavars', {}))
                    new_token = None
                else:
                    if len(self.tree.openElements) == 0 or currentNodeNamespace == self.tree.defaultNamespace or (self.isMathMLTextIntegrationPoint(currentNode) and (type == StartTagToken and token['name'] not in frozenset(['mglyph', 'malignmark']) or type in (CharactersToken, SpaceCharactersToken))) or (currentNodeNamespace == namespaces['mathml'] and currentNodeName == 'annotation-xml' and (type == StartTagToken) and (token['name'] == 'svg')) or (self.isHTMLIntegrationPoint(currentNode) and type in (StartTagToken, CharactersToken, SpaceCharactersToken)):
                        phase = self.phase
                    else:
                        phase = self.phases['inForeignContent']
                    if type == CharactersToken:
                        new_token = phase.processCharacters(new_token)
                    elif type == SpaceCharactersToken:
                        new_token = phase.processSpaceCharacters(new_token)
                    elif type == StartTagToken:
                        new_token = phase.processStartTag(new_token)
                    elif type == EndTagToken:
                        new_token = phase.processEndTag(new_token)
                    elif type == CommentToken:
                        new_token = phase.processComment(new_token)
                    elif type == DoctypeToken:
                        new_token = phase.processDoctype(new_token)
            if type == StartTagToken and prev_token['selfClosing'] and (not prev_token['selfClosingAcknowledged']):
                self.parseError('non-void-element-with-trailing-solidus', {'name': prev_token['name']})
        reprocess = True
        phases = []
        while reprocess:
            phases.append(self.phase)
            reprocess = self.phase.processEOF()
            if reprocess:
                assert self.phase not in phases

    def normalizedTokens(self):
        if False:
            i = 10
            return i + 15
        for token in self.tokenizer:
            yield self.normalizeToken(token)

    def parse(self, stream, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Parse a HTML document into a well-formed tree\n\n        :arg stream: a file-like object or string containing the HTML to be parsed\n\n            The optional encoding parameter must be a string that indicates\n            the encoding.  If specified, that encoding will be used,\n            regardless of any BOM or later declaration (such as in a meta\n            element).\n\n        :arg scripting: treat noscript elements as if JavaScript was turned on\n\n        :returns: parsed tree\n\n        Example:\n\n        >>> from html5lib.html5parser import HTMLParser\n        >>> parser = HTMLParser()\n        >>> parser.parse('<html><body><p>This is a doc</p></body></html>')\n        <Element u'{http://www.w3.org/1999/xhtml}html' at 0x7feac4909db0>\n\n        "
        self._parse(stream, False, None, *args, **kwargs)
        return self.tree.getDocument()

    def parseFragment(self, stream, *args, **kwargs):
        if False:
            while True:
                i = 10
        "Parse a HTML fragment into a well-formed tree fragment\n\n        :arg container: name of the element we're setting the innerHTML\n            property if set to None, default to 'div'\n\n        :arg stream: a file-like object or string containing the HTML to be parsed\n\n            The optional encoding parameter must be a string that indicates\n            the encoding.  If specified, that encoding will be used,\n            regardless of any BOM or later declaration (such as in a meta\n            element)\n\n        :arg scripting: treat noscript elements as if JavaScript was turned on\n\n        :returns: parsed tree\n\n        Example:\n\n        >>> from html5lib.html5libparser import HTMLParser\n        >>> parser = HTMLParser()\n        >>> parser.parseFragment('<b>this is a fragment</b>')\n        <Element u'DOCUMENT_FRAGMENT' at 0x7feac484b090>\n\n        "
        self._parse(stream, True, *args, **kwargs)
        return self.tree.getFragment()

    def parseError(self, errorcode='XXX-undefined-error', datavars=None):
        if False:
            return 10
        if datavars is None:
            datavars = {}
        self.errors.append((self.tokenizer.stream.position(), errorcode, datavars))
        if self.strict:
            raise ParseError(E[errorcode] % datavars)

    def normalizeToken(self, token):
        if False:
            i = 10
            return i + 15
        if token['type'] == tokenTypes['StartTag']:
            raw = token['data']
            token['data'] = OrderedDict(raw)
            if len(raw) > len(token['data']):
                token['data'].update(raw[::-1])
        return token

    def adjustMathMLAttributes(self, token):
        if False:
            print('Hello World!')
        adjust_attributes(token, adjustMathMLAttributes)

    def adjustSVGAttributes(self, token):
        if False:
            i = 10
            return i + 15
        adjust_attributes(token, adjustSVGAttributes)

    def adjustForeignAttributes(self, token):
        if False:
            return 10
        adjust_attributes(token, adjustForeignAttributesMap)

    def reparseTokenNormal(self, token):
        if False:
            while True:
                i = 10
        self.parser.phase()

    def resetInsertionMode(self):
        if False:
            return 10
        last = False
        newModes = {'select': 'inSelect', 'td': 'inCell', 'th': 'inCell', 'tr': 'inRow', 'tbody': 'inTableBody', 'thead': 'inTableBody', 'tfoot': 'inTableBody', 'caption': 'inCaption', 'colgroup': 'inColumnGroup', 'table': 'inTable', 'head': 'inBody', 'body': 'inBody', 'frameset': 'inFrameset', 'html': 'beforeHead'}
        for node in self.tree.openElements[::-1]:
            nodeName = node.name
            new_phase = None
            if node == self.tree.openElements[0]:
                assert self.innerHTML
                last = True
                nodeName = self.innerHTML
            if nodeName in ('select', 'colgroup', 'head', 'html'):
                assert self.innerHTML
            if not last and node.namespace != self.tree.defaultNamespace:
                continue
            if nodeName in newModes:
                new_phase = self.phases[newModes[nodeName]]
                break
            elif last:
                new_phase = self.phases['inBody']
                break
        self.phase = new_phase

    def parseRCDataRawtext(self, token, contentType):
        if False:
            print('Hello World!')
        assert contentType in ('RAWTEXT', 'RCDATA')
        self.tree.insertElement(token)
        if contentType == 'RAWTEXT':
            self.tokenizer.state = self.tokenizer.rawtextState
        else:
            self.tokenizer.state = self.tokenizer.rcdataState
        self.originalPhase = self.phase
        self.phase = self.phases['text']

@_utils.memoize
def getPhases(debug):
    if False:
        for i in range(10):
            print('nop')

    def log(function):
        if False:
            print('Hello World!')
        'Logger that records which phase processes each token'
        type_names = dict(((value, key) for (key, value) in tokenTypes.items()))

        def wrapped(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            if function.__name__.startswith('process') and len(args) > 0:
                token = args[0]
                try:
                    info = {'type': type_names[token['type']]}
                except:
                    raise
                if token['type'] in tagTokenTypes:
                    info['name'] = token['name']
                self.parser.log.append((self.parser.tokenizer.state.__name__, self.parser.phase.__class__.__name__, self.__class__.__name__, function.__name__, info))
                return function(self, *args, **kwargs)
            else:
                return function(self, *args, **kwargs)
        return wrapped

    def getMetaclass(use_metaclass, metaclass_func):
        if False:
            print('Hello World!')
        if use_metaclass:
            return method_decorator_metaclass(metaclass_func)
        else:
            return type

    class Phase(with_metaclass(getMetaclass(debug, log))):
        """Base class for helper object that implements each phase of processing
        """

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            self.parser = parser
            self.tree = tree

        def processEOF(self):
            if False:
                while True:
                    i = 10
            raise NotImplementedError

        def processComment(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.tree.insertComment(token, self.tree.openElements[-1])

        def processDoctype(self, token):
            if False:
                return 10
            self.parser.parseError('unexpected-doctype')

        def processCharacters(self, token):
            if False:
                print('Hello World!')
            self.tree.insertText(token['data'])

        def processSpaceCharacters(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.tree.insertText(token['data'])

        def processStartTag(self, token):
            if False:
                for i in range(10):
                    print('nop')
            return self.startTagHandler[token['name']](token)

        def startTagHtml(self, token):
            if False:
                return 10
            if not self.parser.firstStartTag and token['name'] == 'html':
                self.parser.parseError('non-html-root')
            for (attr, value) in token['data'].items():
                if attr not in self.tree.openElements[0].attributes:
                    self.tree.openElements[0].attributes[attr] = value
            self.parser.firstStartTag = False

        def processEndTag(self, token):
            if False:
                return 10
            return self.endTagHandler[token['name']](token)

    class InitialPhase(Phase):

        def processSpaceCharacters(self, token):
            if False:
                return 10
            pass

        def processComment(self, token):
            if False:
                while True:
                    i = 10
            self.tree.insertComment(token, self.tree.document)

        def processDoctype(self, token):
            if False:
                print('Hello World!')
            name = token['name']
            publicId = token['publicId']
            systemId = token['systemId']
            correct = token['correct']
            if name != 'html' or publicId is not None or (systemId is not None and systemId != 'about:legacy-compat'):
                self.parser.parseError('unknown-doctype')
            if publicId is None:
                publicId = ''
            self.tree.insertDoctype(token)
            if publicId != '':
                publicId = publicId.translate(asciiUpper2Lower)
            if not correct or token['name'] != 'html' or publicId.startswith(('+//silmaril//dtd html pro v0r11 19970101//', '-//advasoft ltd//dtd html 3.0 aswedit + extensions//', '-//as//dtd html 3.0 aswedit + extensions//', '-//ietf//dtd html 2.0 level 1//', '-//ietf//dtd html 2.0 level 2//', '-//ietf//dtd html 2.0 strict level 1//', '-//ietf//dtd html 2.0 strict level 2//', '-//ietf//dtd html 2.0 strict//', '-//ietf//dtd html 2.0//', '-//ietf//dtd html 2.1e//', '-//ietf//dtd html 3.0//', '-//ietf//dtd html 3.2 final//', '-//ietf//dtd html 3.2//', '-//ietf//dtd html 3//', '-//ietf//dtd html level 0//', '-//ietf//dtd html level 1//', '-//ietf//dtd html level 2//', '-//ietf//dtd html level 3//', '-//ietf//dtd html strict level 0//', '-//ietf//dtd html strict level 1//', '-//ietf//dtd html strict level 2//', '-//ietf//dtd html strict level 3//', '-//ietf//dtd html strict//', '-//ietf//dtd html//', '-//metrius//dtd metrius presentational//', '-//microsoft//dtd internet explorer 2.0 html strict//', '-//microsoft//dtd internet explorer 2.0 html//', '-//microsoft//dtd internet explorer 2.0 tables//', '-//microsoft//dtd internet explorer 3.0 html strict//', '-//microsoft//dtd internet explorer 3.0 html//', '-//microsoft//dtd internet explorer 3.0 tables//', '-//netscape comm. corp.//dtd html//', '-//netscape comm. corp.//dtd strict html//', "-//o'reilly and associates//dtd html 2.0//", "-//o'reilly and associates//dtd html extended 1.0//", "-//o'reilly and associates//dtd html extended relaxed 1.0//", '-//softquad software//dtd hotmetal pro 6.0::19990601::extensions to html 4.0//', '-//softquad//dtd hotmetal pro 4.0::19971010::extensions to html 4.0//', '-//spyglass//dtd html 2.0 extended//', '-//sq//dtd html 2.0 hotmetal + extensions//', '-//sun microsystems corp.//dtd hotjava html//', '-//sun microsystems corp.//dtd hotjava strict html//', '-//w3c//dtd html 3 1995-03-24//', '-//w3c//dtd html 3.2 draft//', '-//w3c//dtd html 3.2 final//', '-//w3c//dtd html 3.2//', '-//w3c//dtd html 3.2s draft//', '-//w3c//dtd html 4.0 frameset//', '-//w3c//dtd html 4.0 transitional//', '-//w3c//dtd html experimental 19960712//', '-//w3c//dtd html experimental 970421//', '-//w3c//dtd w3 html//', '-//w3o//dtd w3 html 3.0//', '-//webtechs//dtd mozilla html 2.0//', '-//webtechs//dtd mozilla html//')) or (publicId in ('-//w3o//dtd w3 html strict 3.0//en//', '-/w3c/dtd html 4.0 transitional/en', 'html')) or (publicId.startswith(('-//w3c//dtd html 4.01 frameset//', '-//w3c//dtd html 4.01 transitional//')) and systemId is None) or (systemId and systemId.lower() == 'http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd'):
                self.parser.compatMode = 'quirks'
            elif publicId.startswith(('-//w3c//dtd xhtml 1.0 frameset//', '-//w3c//dtd xhtml 1.0 transitional//')) or (publicId.startswith(('-//w3c//dtd html 4.01 frameset//', '-//w3c//dtd html 4.01 transitional//')) and systemId is not None):
                self.parser.compatMode = 'limited quirks'
            self.parser.phase = self.parser.phases['beforeHtml']

        def anythingElse(self):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.compatMode = 'quirks'
            self.parser.phase = self.parser.phases['beforeHtml']

        def processCharacters(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('expected-doctype-but-got-chars')
            self.anythingElse()
            return token

        def processStartTag(self, token):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('expected-doctype-but-got-start-tag', {'name': token['name']})
            self.anythingElse()
            return token

        def processEndTag(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('expected-doctype-but-got-end-tag', {'name': token['name']})
            self.anythingElse()
            return token

        def processEOF(self):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('expected-doctype-but-got-eof')
            self.anythingElse()
            return True

    class BeforeHtmlPhase(Phase):

        def insertHtmlElement(self):
            if False:
                while True:
                    i = 10
            self.tree.insertRoot(impliedTagToken('html', 'StartTag'))
            self.parser.phase = self.parser.phases['beforeHead']

        def processEOF(self):
            if False:
                for i in range(10):
                    print('nop')
            self.insertHtmlElement()
            return True

        def processComment(self, token):
            if False:
                i = 10
                return i + 15
            self.tree.insertComment(token, self.tree.document)

        def processSpaceCharacters(self, token):
            if False:
                return 10
            pass

        def processCharacters(self, token):
            if False:
                print('Hello World!')
            self.insertHtmlElement()
            return token

        def processStartTag(self, token):
            if False:
                return 10
            if token['name'] == 'html':
                self.parser.firstStartTag = True
            self.insertHtmlElement()
            return token

        def processEndTag(self, token):
            if False:
                print('Hello World!')
            if token['name'] not in ('head', 'body', 'html', 'br'):
                self.parser.parseError('unexpected-end-tag-before-html', {'name': token['name']})
            else:
                self.insertHtmlElement()
                return token

    class BeforeHeadPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('head', self.startTagHead)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([(('head', 'body', 'html', 'br'), self.endTagImplyHead)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                print('Hello World!')
            self.startTagHead(impliedTagToken('head', 'StartTag'))
            return True

        def processSpaceCharacters(self, token):
            if False:
                i = 10
                return i + 15
            pass

        def processCharacters(self, token):
            if False:
                i = 10
                return i + 15
            self.startTagHead(impliedTagToken('head', 'StartTag'))
            return token

        def startTagHtml(self, token):
            if False:
                return 10
            return self.parser.phases['inBody'].processStartTag(token)

        def startTagHead(self, token):
            if False:
                i = 10
                return i + 15
            self.tree.insertElement(token)
            self.tree.headPointer = self.tree.openElements[-1]
            self.parser.phase = self.parser.phases['inHead']

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            self.startTagHead(impliedTagToken('head', 'StartTag'))
            return token

        def endTagImplyHead(self, token):
            if False:
                return 10
            self.startTagHead(impliedTagToken('head', 'StartTag'))
            return token

        def endTagOther(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('end-tag-after-implied-root', {'name': token['name']})

    class InHeadPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('title', self.startTagTitle), (('noframes', 'style'), self.startTagNoFramesStyle), ('noscript', self.startTagNoscript), ('script', self.startTagScript), (('base', 'basefont', 'bgsound', 'command', 'link'), self.startTagBaseLinkCommand), ('meta', self.startTagMeta), ('head', self.startTagHead)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('head', self.endTagHead), (('br', 'html', 'body'), self.endTagHtmlBodyBr)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                print('Hello World!')
            self.anythingElse()
            return True

        def processCharacters(self, token):
            if False:
                while True:
                    i = 10
            self.anythingElse()
            return token

        def startTagHtml(self, token):
            if False:
                while True:
                    i = 10
            return self.parser.phases['inBody'].processStartTag(token)

        def startTagHead(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('two-heads-are-not-better-than-one')

        def startTagBaseLinkCommand(self, token):
            if False:
                print('Hello World!')
            self.tree.insertElement(token)
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True

        def startTagMeta(self, token):
            if False:
                print('Hello World!')
            self.tree.insertElement(token)
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True
            attributes = token['data']
            if self.parser.tokenizer.stream.charEncoding[1] == 'tentative':
                if 'charset' in attributes:
                    self.parser.tokenizer.stream.changeEncoding(attributes['charset'])
                elif 'content' in attributes and 'http-equiv' in attributes and (attributes['http-equiv'].lower() == 'content-type'):
                    data = _inputstream.EncodingBytes(attributes['content'].encode('utf-8'))
                    parser = _inputstream.ContentAttrParser(data)
                    codec = parser.parse()
                    self.parser.tokenizer.stream.changeEncoding(codec)

        def startTagTitle(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseRCDataRawtext(token, 'RCDATA')

        def startTagNoFramesStyle(self, token):
            if False:
                return 10
            self.parser.parseRCDataRawtext(token, 'RAWTEXT')

        def startTagNoscript(self, token):
            if False:
                print('Hello World!')
            if self.parser.scripting:
                self.parser.parseRCDataRawtext(token, 'RAWTEXT')
            else:
                self.tree.insertElement(token)
                self.parser.phase = self.parser.phases['inHeadNoscript']

        def startTagScript(self, token):
            if False:
                i = 10
                return i + 15
            self.tree.insertElement(token)
            self.parser.tokenizer.state = self.parser.tokenizer.scriptDataState
            self.parser.originalPhase = self.parser.phase
            self.parser.phase = self.parser.phases['text']

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            self.anythingElse()
            return token

        def endTagHead(self, token):
            if False:
                while True:
                    i = 10
            node = self.parser.tree.openElements.pop()
            assert node.name == 'head', 'Expected head got %s' % node.name
            self.parser.phase = self.parser.phases['afterHead']

        def endTagHtmlBodyBr(self, token):
            if False:
                while True:
                    i = 10
            self.anythingElse()
            return token

        def endTagOther(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})

        def anythingElse(self):
            if False:
                return 10
            self.endTagHead(impliedTagToken('head'))

    class InHeadNoscriptPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), (('basefont', 'bgsound', 'link', 'meta', 'noframes', 'style'), self.startTagBaseLinkCommand), (('head', 'noscript'), self.startTagHeadNoscript)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('noscript', self.endTagNoscript), ('br', self.endTagBr)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                while True:
                    i = 10
            self.parser.parseError('eof-in-head-noscript')
            self.anythingElse()
            return True

        def processComment(self, token):
            if False:
                print('Hello World!')
            return self.parser.phases['inHead'].processComment(token)

        def processCharacters(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('char-in-head-noscript')
            self.anythingElse()
            return token

        def processSpaceCharacters(self, token):
            if False:
                return 10
            return self.parser.phases['inHead'].processSpaceCharacters(token)

        def startTagHtml(self, token):
            if False:
                i = 10
                return i + 15
            return self.parser.phases['inBody'].processStartTag(token)

        def startTagBaseLinkCommand(self, token):
            if False:
                while True:
                    i = 10
            return self.parser.phases['inHead'].processStartTag(token)

        def startTagHeadNoscript(self, token):
            if False:
                return 10
            self.parser.parseError('unexpected-start-tag', {'name': token['name']})

        def startTagOther(self, token):
            if False:
                return 10
            self.parser.parseError('unexpected-inhead-noscript-tag', {'name': token['name']})
            self.anythingElse()
            return token

        def endTagNoscript(self, token):
            if False:
                for i in range(10):
                    print('nop')
            node = self.parser.tree.openElements.pop()
            assert node.name == 'noscript', 'Expected noscript got %s' % node.name
            self.parser.phase = self.parser.phases['inHead']

        def endTagBr(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('unexpected-inhead-noscript-tag', {'name': token['name']})
            self.anythingElse()
            return token

        def endTagOther(self, token):
            if False:
                return 10
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})

        def anythingElse(self):
            if False:
                i = 10
                return i + 15
            self.endTagNoscript(impliedTagToken('noscript'))

    class AfterHeadPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                for i in range(10):
                    print('nop')
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('body', self.startTagBody), ('frameset', self.startTagFrameset), (('base', 'basefont', 'bgsound', 'link', 'meta', 'noframes', 'script', 'style', 'title'), self.startTagFromHead), ('head', self.startTagHead)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([(('body', 'html', 'br'), self.endTagHtmlBodyBr)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                i = 10
                return i + 15
            self.anythingElse()
            return True

        def processCharacters(self, token):
            if False:
                return 10
            self.anythingElse()
            return token

        def startTagHtml(self, token):
            if False:
                i = 10
                return i + 15
            return self.parser.phases['inBody'].processStartTag(token)

        def startTagBody(self, token):
            if False:
                return 10
            self.parser.framesetOK = False
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inBody']

        def startTagFrameset(self, token):
            if False:
                while True:
                    i = 10
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inFrameset']

        def startTagFromHead(self, token):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('unexpected-start-tag-out-of-my-head', {'name': token['name']})
            self.tree.openElements.append(self.tree.headPointer)
            self.parser.phases['inHead'].processStartTag(token)
            for node in self.tree.openElements[::-1]:
                if node.name == 'head':
                    self.tree.openElements.remove(node)
                    break

        def startTagHead(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-start-tag', {'name': token['name']})

        def startTagOther(self, token):
            if False:
                return 10
            self.anythingElse()
            return token

        def endTagHtmlBodyBr(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.anythingElse()
            return token

        def endTagOther(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})

        def anythingElse(self):
            if False:
                i = 10
                return i + 15
            self.tree.insertElement(impliedTagToken('body', 'StartTag'))
            self.parser.phase = self.parser.phases['inBody']
            self.parser.framesetOK = True

    class InBodyPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                for i in range(10):
                    print('nop')
            Phase.__init__(self, parser, tree)
            self.processSpaceCharacters = self.processSpaceCharactersNonPre
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), (('base', 'basefont', 'bgsound', 'command', 'link', 'meta', 'script', 'style', 'title'), self.startTagProcessInHead), ('body', self.startTagBody), ('frameset', self.startTagFrameset), (('address', 'article', 'aside', 'blockquote', 'center', 'details', 'dir', 'div', 'dl', 'fieldset', 'figcaption', 'figure', 'footer', 'header', 'hgroup', 'main', 'menu', 'nav', 'ol', 'p', 'section', 'summary', 'ul'), self.startTagCloseP), (headingElements, self.startTagHeading), (('pre', 'listing'), self.startTagPreListing), ('form', self.startTagForm), (('li', 'dd', 'dt'), self.startTagListItem), ('plaintext', self.startTagPlaintext), ('a', self.startTagA), (('b', 'big', 'code', 'em', 'font', 'i', 's', 'small', 'strike', 'strong', 'tt', 'u'), self.startTagFormatting), ('nobr', self.startTagNobr), ('button', self.startTagButton), (('applet', 'marquee', 'object'), self.startTagAppletMarqueeObject), ('xmp', self.startTagXmp), ('table', self.startTagTable), (('area', 'br', 'embed', 'img', 'keygen', 'wbr'), self.startTagVoidFormatting), (('param', 'source', 'track'), self.startTagParamSource), ('input', self.startTagInput), ('hr', self.startTagHr), ('image', self.startTagImage), ('isindex', self.startTagIsIndex), ('textarea', self.startTagTextarea), ('iframe', self.startTagIFrame), ('noscript', self.startTagNoscript), (('noembed', 'noframes'), self.startTagRawtext), ('select', self.startTagSelect), (('rp', 'rt'), self.startTagRpRt), (('option', 'optgroup'), self.startTagOpt), ('math', self.startTagMath), ('svg', self.startTagSvg), (('caption', 'col', 'colgroup', 'frame', 'head', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), self.startTagMisplaced)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('body', self.endTagBody), ('html', self.endTagHtml), (('address', 'article', 'aside', 'blockquote', 'button', 'center', 'details', 'dialog', 'dir', 'div', 'dl', 'fieldset', 'figcaption', 'figure', 'footer', 'header', 'hgroup', 'listing', 'main', 'menu', 'nav', 'ol', 'pre', 'section', 'summary', 'ul'), self.endTagBlock), ('form', self.endTagForm), ('p', self.endTagP), (('dd', 'dt', 'li'), self.endTagListItem), (headingElements, self.endTagHeading), (('a', 'b', 'big', 'code', 'em', 'font', 'i', 'nobr', 's', 'small', 'strike', 'strong', 'tt', 'u'), self.endTagFormatting), (('applet', 'marquee', 'object'), self.endTagAppletMarqueeObject), ('br', self.endTagBr)])
            self.endTagHandler.default = self.endTagOther

        def isMatchingFormattingElement(self, node1, node2):
            if False:
                print('Hello World!')
            return node1.name == node2.name and node1.namespace == node2.namespace and (node1.attributes == node2.attributes)

        def addFormattingElement(self, token):
            if False:
                print('Hello World!')
            self.tree.insertElement(token)
            element = self.tree.openElements[-1]
            matchingElements = []
            for node in self.tree.activeFormattingElements[::-1]:
                if node is Marker:
                    break
                elif self.isMatchingFormattingElement(node, element):
                    matchingElements.append(node)
            assert len(matchingElements) <= 3
            if len(matchingElements) == 3:
                self.tree.activeFormattingElements.remove(matchingElements[-1])
            self.tree.activeFormattingElements.append(element)

        def processEOF(self):
            if False:
                i = 10
                return i + 15
            allowed_elements = frozenset(('dd', 'dt', 'li', 'p', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'body', 'html'))
            for node in self.tree.openElements[::-1]:
                if node.name not in allowed_elements:
                    self.parser.parseError('expected-closing-tag-but-got-eof')
                    break

        def processSpaceCharactersDropNewline(self, token):
            if False:
                while True:
                    i = 10
            data = token['data']
            self.processSpaceCharacters = self.processSpaceCharactersNonPre
            if data.startswith('\n') and self.tree.openElements[-1].name in ('pre', 'listing', 'textarea') and (not self.tree.openElements[-1].hasContent()):
                data = data[1:]
            if data:
                self.tree.reconstructActiveFormattingElements()
                self.tree.insertText(data)

        def processCharacters(self, token):
            if False:
                i = 10
                return i + 15
            if token['data'] == '\x00':
                return
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertText(token['data'])
            if self.parser.framesetOK and any([char not in spaceCharacters for char in token['data']]):
                self.parser.framesetOK = False

        def processSpaceCharactersNonPre(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertText(token['data'])

        def startTagProcessInHead(self, token):
            if False:
                i = 10
                return i + 15
            return self.parser.phases['inHead'].processStartTag(token)

        def startTagBody(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-start-tag', {'name': 'body'})
            if len(self.tree.openElements) == 1 or self.tree.openElements[1].name != 'body':
                assert self.parser.innerHTML
            else:
                self.parser.framesetOK = False
                for (attr, value) in token['data'].items():
                    if attr not in self.tree.openElements[1].attributes:
                        self.tree.openElements[1].attributes[attr] = value

        def startTagFrameset(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-start-tag', {'name': 'frameset'})
            if len(self.tree.openElements) == 1 or self.tree.openElements[1].name != 'body':
                assert self.parser.innerHTML
            elif not self.parser.framesetOK:
                pass
            else:
                if self.tree.openElements[1].parent:
                    self.tree.openElements[1].parent.removeChild(self.tree.openElements[1])
                while self.tree.openElements[-1].name != 'html':
                    self.tree.openElements.pop()
                self.tree.insertElement(token)
                self.parser.phase = self.parser.phases['inFrameset']

        def startTagCloseP(self, token):
            if False:
                while True:
                    i = 10
            if self.tree.elementInScope('p', variant='button'):
                self.endTagP(impliedTagToken('p'))
            self.tree.insertElement(token)

        def startTagPreListing(self, token):
            if False:
                return 10
            if self.tree.elementInScope('p', variant='button'):
                self.endTagP(impliedTagToken('p'))
            self.tree.insertElement(token)
            self.parser.framesetOK = False
            self.processSpaceCharacters = self.processSpaceCharactersDropNewline

        def startTagForm(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if self.tree.formPointer:
                self.parser.parseError('unexpected-start-tag', {'name': 'form'})
            else:
                if self.tree.elementInScope('p', variant='button'):
                    self.endTagP(impliedTagToken('p'))
                self.tree.insertElement(token)
                self.tree.formPointer = self.tree.openElements[-1]

        def startTagListItem(self, token):
            if False:
                while True:
                    i = 10
            self.parser.framesetOK = False
            stopNamesMap = {'li': ['li'], 'dt': ['dt', 'dd'], 'dd': ['dt', 'dd']}
            stopNames = stopNamesMap[token['name']]
            for node in reversed(self.tree.openElements):
                if node.name in stopNames:
                    self.parser.phase.processEndTag(impliedTagToken(node.name, 'EndTag'))
                    break
                if node.nameTuple in specialElements and node.name not in ('address', 'div', 'p'):
                    break
            if self.tree.elementInScope('p', variant='button'):
                self.parser.phase.processEndTag(impliedTagToken('p', 'EndTag'))
            self.tree.insertElement(token)

        def startTagPlaintext(self, token):
            if False:
                i = 10
                return i + 15
            if self.tree.elementInScope('p', variant='button'):
                self.endTagP(impliedTagToken('p'))
            self.tree.insertElement(token)
            self.parser.tokenizer.state = self.parser.tokenizer.plaintextState

        def startTagHeading(self, token):
            if False:
                i = 10
                return i + 15
            if self.tree.elementInScope('p', variant='button'):
                self.endTagP(impliedTagToken('p'))
            if self.tree.openElements[-1].name in headingElements:
                self.parser.parseError('unexpected-start-tag', {'name': token['name']})
                self.tree.openElements.pop()
            self.tree.insertElement(token)

        def startTagA(self, token):
            if False:
                print('Hello World!')
            afeAElement = self.tree.elementInActiveFormattingElements('a')
            if afeAElement:
                self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'a', 'endName': 'a'})
                self.endTagFormatting(impliedTagToken('a'))
                if afeAElement in self.tree.openElements:
                    self.tree.openElements.remove(afeAElement)
                if afeAElement in self.tree.activeFormattingElements:
                    self.tree.activeFormattingElements.remove(afeAElement)
            self.tree.reconstructActiveFormattingElements()
            self.addFormattingElement(token)

        def startTagFormatting(self, token):
            if False:
                print('Hello World!')
            self.tree.reconstructActiveFormattingElements()
            self.addFormattingElement(token)

        def startTagNobr(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.tree.reconstructActiveFormattingElements()
            if self.tree.elementInScope('nobr'):
                self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'nobr', 'endName': 'nobr'})
                self.processEndTag(impliedTagToken('nobr'))
                self.tree.reconstructActiveFormattingElements()
            self.addFormattingElement(token)

        def startTagButton(self, token):
            if False:
                i = 10
                return i + 15
            if self.tree.elementInScope('button'):
                self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'button', 'endName': 'button'})
                self.processEndTag(impliedTagToken('button'))
                return token
            else:
                self.tree.reconstructActiveFormattingElements()
                self.tree.insertElement(token)
                self.parser.framesetOK = False

        def startTagAppletMarqueeObject(self, token):
            if False:
                return 10
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertElement(token)
            self.tree.activeFormattingElements.append(Marker)
            self.parser.framesetOK = False

        def startTagXmp(self, token):
            if False:
                i = 10
                return i + 15
            if self.tree.elementInScope('p', variant='button'):
                self.endTagP(impliedTagToken('p'))
            self.tree.reconstructActiveFormattingElements()
            self.parser.framesetOK = False
            self.parser.parseRCDataRawtext(token, 'RAWTEXT')

        def startTagTable(self, token):
            if False:
                print('Hello World!')
            if self.parser.compatMode != 'quirks':
                if self.tree.elementInScope('p', variant='button'):
                    self.processEndTag(impliedTagToken('p'))
            self.tree.insertElement(token)
            self.parser.framesetOK = False
            self.parser.phase = self.parser.phases['inTable']

        def startTagVoidFormatting(self, token):
            if False:
                return 10
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertElement(token)
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True
            self.parser.framesetOK = False

        def startTagInput(self, token):
            if False:
                i = 10
                return i + 15
            framesetOK = self.parser.framesetOK
            self.startTagVoidFormatting(token)
            if 'type' in token['data'] and token['data']['type'].translate(asciiUpper2Lower) == 'hidden':
                self.parser.framesetOK = framesetOK

        def startTagParamSource(self, token):
            if False:
                while True:
                    i = 10
            self.tree.insertElement(token)
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True

        def startTagHr(self, token):
            if False:
                print('Hello World!')
            if self.tree.elementInScope('p', variant='button'):
                self.endTagP(impliedTagToken('p'))
            self.tree.insertElement(token)
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True
            self.parser.framesetOK = False

        def startTagImage(self, token):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('unexpected-start-tag-treated-as', {'originalName': 'image', 'newName': 'img'})
            self.processStartTag(impliedTagToken('img', 'StartTag', attributes=token['data'], selfClosing=token['selfClosing']))

        def startTagIsIndex(self, token):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('deprecated-tag', {'name': 'isindex'})
            if self.tree.formPointer:
                return
            form_attrs = {}
            if 'action' in token['data']:
                form_attrs['action'] = token['data']['action']
            self.processStartTag(impliedTagToken('form', 'StartTag', attributes=form_attrs))
            self.processStartTag(impliedTagToken('hr', 'StartTag'))
            self.processStartTag(impliedTagToken('label', 'StartTag'))
            if 'prompt' in token['data']:
                prompt = token['data']['prompt']
            else:
                prompt = 'This is a searchable index. Enter search keywords: '
            self.processCharacters({'type': tokenTypes['Characters'], 'data': prompt})
            attributes = token['data'].copy()
            if 'action' in attributes:
                del attributes['action']
            if 'prompt' in attributes:
                del attributes['prompt']
            attributes['name'] = 'isindex'
            self.processStartTag(impliedTagToken('input', 'StartTag', attributes=attributes, selfClosing=token['selfClosing']))
            self.processEndTag(impliedTagToken('label'))
            self.processStartTag(impliedTagToken('hr', 'StartTag'))
            self.processEndTag(impliedTagToken('form'))

        def startTagTextarea(self, token):
            if False:
                while True:
                    i = 10
            self.tree.insertElement(token)
            self.parser.tokenizer.state = self.parser.tokenizer.rcdataState
            self.processSpaceCharacters = self.processSpaceCharactersDropNewline
            self.parser.framesetOK = False

        def startTagIFrame(self, token):
            if False:
                print('Hello World!')
            self.parser.framesetOK = False
            self.startTagRawtext(token)

        def startTagNoscript(self, token):
            if False:
                i = 10
                return i + 15
            if self.parser.scripting:
                self.startTagRawtext(token)
            else:
                self.startTagOther(token)

        def startTagRawtext(self, token):
            if False:
                print('Hello World!')
            'iframe, noembed noframes, noscript(if scripting enabled)'
            self.parser.parseRCDataRawtext(token, 'RAWTEXT')

        def startTagOpt(self, token):
            if False:
                return 10
            if self.tree.openElements[-1].name == 'option':
                self.parser.phase.processEndTag(impliedTagToken('option'))
            self.tree.reconstructActiveFormattingElements()
            self.parser.tree.insertElement(token)

        def startTagSelect(self, token):
            if False:
                while True:
                    i = 10
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertElement(token)
            self.parser.framesetOK = False
            if self.parser.phase in (self.parser.phases['inTable'], self.parser.phases['inCaption'], self.parser.phases['inColumnGroup'], self.parser.phases['inTableBody'], self.parser.phases['inRow'], self.parser.phases['inCell']):
                self.parser.phase = self.parser.phases['inSelectInTable']
            else:
                self.parser.phase = self.parser.phases['inSelect']

        def startTagRpRt(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if self.tree.elementInScope('ruby'):
                self.tree.generateImpliedEndTags()
                if self.tree.openElements[-1].name != 'ruby':
                    self.parser.parseError()
            self.tree.insertElement(token)

        def startTagMath(self, token):
            if False:
                while True:
                    i = 10
            self.tree.reconstructActiveFormattingElements()
            self.parser.adjustMathMLAttributes(token)
            self.parser.adjustForeignAttributes(token)
            token['namespace'] = namespaces['mathml']
            self.tree.insertElement(token)
            if token['selfClosing']:
                self.tree.openElements.pop()
                token['selfClosingAcknowledged'] = True

        def startTagSvg(self, token):
            if False:
                while True:
                    i = 10
            self.tree.reconstructActiveFormattingElements()
            self.parser.adjustSVGAttributes(token)
            self.parser.adjustForeignAttributes(token)
            token['namespace'] = namespaces['svg']
            self.tree.insertElement(token)
            if token['selfClosing']:
                self.tree.openElements.pop()
                token['selfClosingAcknowledged'] = True

        def startTagMisplaced(self, token):
            if False:
                while True:
                    i = 10
            ' Elements that should be children of other elements that have a\n            different insertion mode; here they are ignored\n            "caption", "col", "colgroup", "frame", "frameset", "head",\n            "option", "optgroup", "tbody", "td", "tfoot", "th", "thead",\n            "tr", "noscript"\n            '
            self.parser.parseError('unexpected-start-tag-ignored', {'name': token['name']})

        def startTagOther(self, token):
            if False:
                return 10
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertElement(token)

        def endTagP(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if not self.tree.elementInScope('p', variant='button'):
                self.startTagCloseP(impliedTagToken('p', 'StartTag'))
                self.parser.parseError('unexpected-end-tag', {'name': 'p'})
                self.endTagP(impliedTagToken('p', 'EndTag'))
            else:
                self.tree.generateImpliedEndTags('p')
                if self.tree.openElements[-1].name != 'p':
                    self.parser.parseError('unexpected-end-tag', {'name': 'p'})
                node = self.tree.openElements.pop()
                while node.name != 'p':
                    node = self.tree.openElements.pop()

        def endTagBody(self, token):
            if False:
                print('Hello World!')
            if not self.tree.elementInScope('body'):
                self.parser.parseError()
                return
            elif self.tree.openElements[-1].name != 'body':
                for node in self.tree.openElements[2:]:
                    if node.name not in frozenset(('dd', 'dt', 'li', 'optgroup', 'option', 'p', 'rp', 'rt', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'body', 'html')):
                        self.parser.parseError('expected-one-end-tag-but-got-another', {'gotName': 'body', 'expectedName': node.name})
                        break
            self.parser.phase = self.parser.phases['afterBody']

        def endTagHtml(self, token):
            if False:
                while True:
                    i = 10
            if self.tree.elementInScope('body'):
                self.endTagBody(impliedTagToken('body'))
                return token

        def endTagBlock(self, token):
            if False:
                print('Hello World!')
            if token['name'] == 'pre':
                self.processSpaceCharacters = self.processSpaceCharactersNonPre
            inScope = self.tree.elementInScope(token['name'])
            if inScope:
                self.tree.generateImpliedEndTags()
            if self.tree.openElements[-1].name != token['name']:
                self.parser.parseError('end-tag-too-early', {'name': token['name']})
            if inScope:
                node = self.tree.openElements.pop()
                while node.name != token['name']:
                    node = self.tree.openElements.pop()

        def endTagForm(self, token):
            if False:
                while True:
                    i = 10
            node = self.tree.formPointer
            self.tree.formPointer = None
            if node is None or not self.tree.elementInScope(node):
                self.parser.parseError('unexpected-end-tag', {'name': 'form'})
            else:
                self.tree.generateImpliedEndTags()
                if self.tree.openElements[-1] != node:
                    self.parser.parseError('end-tag-too-early-ignored', {'name': 'form'})
                self.tree.openElements.remove(node)

        def endTagListItem(self, token):
            if False:
                return 10
            if token['name'] == 'li':
                variant = 'list'
            else:
                variant = None
            if not self.tree.elementInScope(token['name'], variant=variant):
                self.parser.parseError('unexpected-end-tag', {'name': token['name']})
            else:
                self.tree.generateImpliedEndTags(exclude=token['name'])
                if self.tree.openElements[-1].name != token['name']:
                    self.parser.parseError('end-tag-too-early', {'name': token['name']})
                node = self.tree.openElements.pop()
                while node.name != token['name']:
                    node = self.tree.openElements.pop()

        def endTagHeading(self, token):
            if False:
                for i in range(10):
                    print('nop')
            for item in headingElements:
                if self.tree.elementInScope(item):
                    self.tree.generateImpliedEndTags()
                    break
            if self.tree.openElements[-1].name != token['name']:
                self.parser.parseError('end-tag-too-early', {'name': token['name']})
            for item in headingElements:
                if self.tree.elementInScope(item):
                    item = self.tree.openElements.pop()
                    while item.name not in headingElements:
                        item = self.tree.openElements.pop()
                    break

        def endTagFormatting(self, token):
            if False:
                for i in range(10):
                    print('nop')
            'The much-feared adoption agency algorithm'
            outerLoopCounter = 0
            while outerLoopCounter < 8:
                outerLoopCounter += 1
                formattingElement = self.tree.elementInActiveFormattingElements(token['name'])
                if not formattingElement or (formattingElement in self.tree.openElements and (not self.tree.elementInScope(formattingElement.name))):
                    self.endTagOther(token)
                    return
                elif formattingElement not in self.tree.openElements:
                    self.parser.parseError('adoption-agency-1.2', {'name': token['name']})
                    self.tree.activeFormattingElements.remove(formattingElement)
                    return
                elif not self.tree.elementInScope(formattingElement.name):
                    self.parser.parseError('adoption-agency-4.4', {'name': token['name']})
                    return
                elif formattingElement != self.tree.openElements[-1]:
                    self.parser.parseError('adoption-agency-1.3', {'name': token['name']})
                afeIndex = self.tree.openElements.index(formattingElement)
                furthestBlock = None
                for element in self.tree.openElements[afeIndex:]:
                    if element.nameTuple in specialElements:
                        furthestBlock = element
                        break
                if furthestBlock is None:
                    element = self.tree.openElements.pop()
                    while element != formattingElement:
                        element = self.tree.openElements.pop()
                    self.tree.activeFormattingElements.remove(element)
                    return
                commonAncestor = self.tree.openElements[afeIndex - 1]
                bookmark = self.tree.activeFormattingElements.index(formattingElement)
                lastNode = node = furthestBlock
                innerLoopCounter = 0
                index = self.tree.openElements.index(node)
                while innerLoopCounter < 3:
                    innerLoopCounter += 1
                    index -= 1
                    node = self.tree.openElements[index]
                    if node not in self.tree.activeFormattingElements:
                        self.tree.openElements.remove(node)
                        continue
                    if node == formattingElement:
                        break
                    if lastNode == furthestBlock:
                        bookmark = self.tree.activeFormattingElements.index(node) + 1
                    clone = node.cloneNode()
                    self.tree.activeFormattingElements[self.tree.activeFormattingElements.index(node)] = clone
                    self.tree.openElements[self.tree.openElements.index(node)] = clone
                    node = clone
                    if lastNode.parent:
                        lastNode.parent.removeChild(lastNode)
                    node.appendChild(lastNode)
                    lastNode = node
                if lastNode.parent:
                    lastNode.parent.removeChild(lastNode)
                if commonAncestor.name in frozenset(('table', 'tbody', 'tfoot', 'thead', 'tr')):
                    (parent, insertBefore) = self.tree.getTableMisnestedNodePosition()
                    parent.insertBefore(lastNode, insertBefore)
                else:
                    commonAncestor.appendChild(lastNode)
                clone = formattingElement.cloneNode()
                furthestBlock.reparentChildren(clone)
                furthestBlock.appendChild(clone)
                self.tree.activeFormattingElements.remove(formattingElement)
                self.tree.activeFormattingElements.insert(bookmark, clone)
                self.tree.openElements.remove(formattingElement)
                self.tree.openElements.insert(self.tree.openElements.index(furthestBlock) + 1, clone)

        def endTagAppletMarqueeObject(self, token):
            if False:
                print('Hello World!')
            if self.tree.elementInScope(token['name']):
                self.tree.generateImpliedEndTags()
            if self.tree.openElements[-1].name != token['name']:
                self.parser.parseError('end-tag-too-early', {'name': token['name']})
            if self.tree.elementInScope(token['name']):
                element = self.tree.openElements.pop()
                while element.name != token['name']:
                    element = self.tree.openElements.pop()
                self.tree.clearActiveFormattingElements()

        def endTagBr(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-end-tag-treated-as', {'originalName': 'br', 'newName': 'br element'})
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertElement(impliedTagToken('br', 'StartTag'))
            self.tree.openElements.pop()

        def endTagOther(self, token):
            if False:
                for i in range(10):
                    print('nop')
            for node in self.tree.openElements[::-1]:
                if node.name == token['name']:
                    self.tree.generateImpliedEndTags(exclude=token['name'])
                    if self.tree.openElements[-1].name != token['name']:
                        self.parser.parseError('unexpected-end-tag', {'name': token['name']})
                    while self.tree.openElements.pop() != node:
                        pass
                    break
                elif node.nameTuple in specialElements:
                    self.parser.parseError('unexpected-end-tag', {'name': token['name']})
                    break

    class TextPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('script', self.endTagScript)])
            self.endTagHandler.default = self.endTagOther

        def processCharacters(self, token):
            if False:
                print('Hello World!')
            self.tree.insertText(token['data'])

        def processEOF(self):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('expected-named-closing-tag-but-got-eof', {'name': self.tree.openElements[-1].name})
            self.tree.openElements.pop()
            self.parser.phase = self.parser.originalPhase
            return True

        def startTagOther(self, token):
            if False:
                return 10
            assert False, 'Tried to process start tag %s in RCDATA/RAWTEXT mode' % token['name']

        def endTagScript(self, token):
            if False:
                print('Hello World!')
            node = self.tree.openElements.pop()
            assert node.name == 'script'
            self.parser.phase = self.parser.originalPhase

        def endTagOther(self, token):
            if False:
                print('Hello World!')
            self.tree.openElements.pop()
            self.parser.phase = self.parser.originalPhase

    class InTablePhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('caption', self.startTagCaption), ('colgroup', self.startTagColgroup), ('col', self.startTagCol), (('tbody', 'tfoot', 'thead'), self.startTagRowGroup), (('td', 'th', 'tr'), self.startTagImplyTbody), ('table', self.startTagTable), (('style', 'script'), self.startTagStyleScript), ('input', self.startTagInput), ('form', self.startTagForm)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('table', self.endTagTable), (('body', 'caption', 'col', 'colgroup', 'html', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), self.endTagIgnore)])
            self.endTagHandler.default = self.endTagOther

        def clearStackToTableContext(self):
            if False:
                i = 10
                return i + 15
            while self.tree.openElements[-1].name not in ('table', 'html'):
                self.tree.openElements.pop()

        def processEOF(self):
            if False:
                return 10
            if self.tree.openElements[-1].name != 'html':
                self.parser.parseError('eof-in-table')
            else:
                assert self.parser.innerHTML

        def processSpaceCharacters(self, token):
            if False:
                i = 10
                return i + 15
            originalPhase = self.parser.phase
            self.parser.phase = self.parser.phases['inTableText']
            self.parser.phase.originalPhase = originalPhase
            self.parser.phase.processSpaceCharacters(token)

        def processCharacters(self, token):
            if False:
                print('Hello World!')
            originalPhase = self.parser.phase
            self.parser.phase = self.parser.phases['inTableText']
            self.parser.phase.originalPhase = originalPhase
            self.parser.phase.processCharacters(token)

        def insertText(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.tree.insertFromTable = True
            self.parser.phases['inBody'].processCharacters(token)
            self.tree.insertFromTable = False

        def startTagCaption(self, token):
            if False:
                while True:
                    i = 10
            self.clearStackToTableContext()
            self.tree.activeFormattingElements.append(Marker)
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inCaption']

        def startTagColgroup(self, token):
            if False:
                while True:
                    i = 10
            self.clearStackToTableContext()
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inColumnGroup']

        def startTagCol(self, token):
            if False:
                while True:
                    i = 10
            self.startTagColgroup(impliedTagToken('colgroup', 'StartTag'))
            return token

        def startTagRowGroup(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.clearStackToTableContext()
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inTableBody']

        def startTagImplyTbody(self, token):
            if False:
                return 10
            self.startTagRowGroup(impliedTagToken('tbody', 'StartTag'))
            return token

        def startTagTable(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'table', 'endName': 'table'})
            self.parser.phase.processEndTag(impliedTagToken('table'))
            if not self.parser.innerHTML:
                return token

        def startTagStyleScript(self, token):
            if False:
                for i in range(10):
                    print('nop')
            return self.parser.phases['inHead'].processStartTag(token)

        def startTagInput(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if 'type' in token['data'] and token['data']['type'].translate(asciiUpper2Lower) == 'hidden':
                self.parser.parseError('unexpected-hidden-input-in-table')
                self.tree.insertElement(token)
                self.tree.openElements.pop()
            else:
                self.startTagOther(token)

        def startTagForm(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-form-in-table')
            if self.tree.formPointer is None:
                self.tree.insertElement(token)
                self.tree.formPointer = self.tree.openElements[-1]
                self.tree.openElements.pop()

        def startTagOther(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-start-tag-implies-table-voodoo', {'name': token['name']})
            self.tree.insertFromTable = True
            self.parser.phases['inBody'].processStartTag(token)
            self.tree.insertFromTable = False

        def endTagTable(self, token):
            if False:
                while True:
                    i = 10
            if self.tree.elementInScope('table', variant='table'):
                self.tree.generateImpliedEndTags()
                if self.tree.openElements[-1].name != 'table':
                    self.parser.parseError('end-tag-too-early-named', {'gotName': 'table', 'expectedName': self.tree.openElements[-1].name})
                while self.tree.openElements[-1].name != 'table':
                    self.tree.openElements.pop()
                self.tree.openElements.pop()
                self.parser.resetInsertionMode()
            else:
                assert self.parser.innerHTML
                self.parser.parseError()

        def endTagIgnore(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})

        def endTagOther(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('unexpected-end-tag-implies-table-voodoo', {'name': token['name']})
            self.tree.insertFromTable = True
            self.parser.phases['inBody'].processEndTag(token)
            self.tree.insertFromTable = False

    class InTableTextPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.originalPhase = None
            self.characterTokens = []

        def flushCharacters(self):
            if False:
                i = 10
                return i + 15
            data = ''.join([item['data'] for item in self.characterTokens])
            if any([item not in spaceCharacters for item in data]):
                token = {'type': tokenTypes['Characters'], 'data': data}
                self.parser.phases['inTable'].insertText(token)
            elif data:
                self.tree.insertText(data)
            self.characterTokens = []

        def processComment(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.flushCharacters()
            self.parser.phase = self.originalPhase
            return token

        def processEOF(self):
            if False:
                print('Hello World!')
            self.flushCharacters()
            self.parser.phase = self.originalPhase
            return True

        def processCharacters(self, token):
            if False:
                print('Hello World!')
            if token['data'] == '\x00':
                return
            self.characterTokens.append(token)

        def processSpaceCharacters(self, token):
            if False:
                i = 10
                return i + 15
            self.characterTokens.append(token)

        def processStartTag(self, token):
            if False:
                print('Hello World!')
            self.flushCharacters()
            self.parser.phase = self.originalPhase
            return token

        def processEndTag(self, token):
            if False:
                print('Hello World!')
            self.flushCharacters()
            self.parser.phase = self.originalPhase
            return token

    class InCaptionPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                for i in range(10):
                    print('nop')
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), (('caption', 'col', 'colgroup', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), self.startTagTableElement)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('caption', self.endTagCaption), ('table', self.endTagTable), (('body', 'col', 'colgroup', 'html', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), self.endTagIgnore)])
            self.endTagHandler.default = self.endTagOther

        def ignoreEndTagCaption(self):
            if False:
                print('Hello World!')
            return not self.tree.elementInScope('caption', variant='table')

        def processEOF(self):
            if False:
                while True:
                    i = 10
            self.parser.phases['inBody'].processEOF()

        def processCharacters(self, token):
            if False:
                print('Hello World!')
            return self.parser.phases['inBody'].processCharacters(token)

        def startTagTableElement(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError()
            ignoreEndTag = self.ignoreEndTagCaption()
            self.parser.phase.processEndTag(impliedTagToken('caption'))
            if not ignoreEndTag:
                return token

        def startTagOther(self, token):
            if False:
                return 10
            return self.parser.phases['inBody'].processStartTag(token)

        def endTagCaption(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if not self.ignoreEndTagCaption():
                self.tree.generateImpliedEndTags()
                if self.tree.openElements[-1].name != 'caption':
                    self.parser.parseError('expected-one-end-tag-but-got-another', {'gotName': 'caption', 'expectedName': self.tree.openElements[-1].name})
                while self.tree.openElements[-1].name != 'caption':
                    self.tree.openElements.pop()
                self.tree.openElements.pop()
                self.tree.clearActiveFormattingElements()
                self.parser.phase = self.parser.phases['inTable']
            else:
                assert self.parser.innerHTML
                self.parser.parseError()

        def endTagTable(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError()
            ignoreEndTag = self.ignoreEndTagCaption()
            self.parser.phase.processEndTag(impliedTagToken('caption'))
            if not ignoreEndTag:
                return token

        def endTagIgnore(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})

        def endTagOther(self, token):
            if False:
                i = 10
                return i + 15
            return self.parser.phases['inBody'].processEndTag(token)

    class InColumnGroupPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                for i in range(10):
                    print('nop')
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('col', self.startTagCol)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('colgroup', self.endTagColgroup), ('col', self.endTagCol)])
            self.endTagHandler.default = self.endTagOther

        def ignoreEndTagColgroup(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.tree.openElements[-1].name == 'html'

        def processEOF(self):
            if False:
                return 10
            if self.tree.openElements[-1].name == 'html':
                assert self.parser.innerHTML
                return
            else:
                ignoreEndTag = self.ignoreEndTagColgroup()
                self.endTagColgroup(impliedTagToken('colgroup'))
                if not ignoreEndTag:
                    return True

        def processCharacters(self, token):
            if False:
                return 10
            ignoreEndTag = self.ignoreEndTagColgroup()
            self.endTagColgroup(impliedTagToken('colgroup'))
            if not ignoreEndTag:
                return token

        def startTagCol(self, token):
            if False:
                while True:
                    i = 10
            self.tree.insertElement(token)
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            ignoreEndTag = self.ignoreEndTagColgroup()
            self.endTagColgroup(impliedTagToken('colgroup'))
            if not ignoreEndTag:
                return token

        def endTagColgroup(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if self.ignoreEndTagColgroup():
                assert self.parser.innerHTML
                self.parser.parseError()
            else:
                self.tree.openElements.pop()
                self.parser.phase = self.parser.phases['inTable']

        def endTagCol(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('no-end-tag', {'name': 'col'})

        def endTagOther(self, token):
            if False:
                print('Hello World!')
            ignoreEndTag = self.ignoreEndTagColgroup()
            self.endTagColgroup(impliedTagToken('colgroup'))
            if not ignoreEndTag:
                return token

    class InTableBodyPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('tr', self.startTagTr), (('td', 'th'), self.startTagTableCell), (('caption', 'col', 'colgroup', 'tbody', 'tfoot', 'thead'), self.startTagTableOther)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([(('tbody', 'tfoot', 'thead'), self.endTagTableRowGroup), ('table', self.endTagTable), (('body', 'caption', 'col', 'colgroup', 'html', 'td', 'th', 'tr'), self.endTagIgnore)])
            self.endTagHandler.default = self.endTagOther

        def clearStackToTableBodyContext(self):
            if False:
                i = 10
                return i + 15
            while self.tree.openElements[-1].name not in ('tbody', 'tfoot', 'thead', 'html'):
                self.tree.openElements.pop()
            if self.tree.openElements[-1].name == 'html':
                assert self.parser.innerHTML

        def processEOF(self):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.phases['inTable'].processEOF()

        def processSpaceCharacters(self, token):
            if False:
                return 10
            return self.parser.phases['inTable'].processSpaceCharacters(token)

        def processCharacters(self, token):
            if False:
                while True:
                    i = 10
            return self.parser.phases['inTable'].processCharacters(token)

        def startTagTr(self, token):
            if False:
                i = 10
                return i + 15
            self.clearStackToTableBodyContext()
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inRow']

        def startTagTableCell(self, token):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('unexpected-cell-in-table-body', {'name': token['name']})
            self.startTagTr(impliedTagToken('tr', 'StartTag'))
            return token

        def startTagTableOther(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if self.tree.elementInScope('tbody', variant='table') or self.tree.elementInScope('thead', variant='table') or self.tree.elementInScope('tfoot', variant='table'):
                self.clearStackToTableBodyContext()
                self.endTagTableRowGroup(impliedTagToken(self.tree.openElements[-1].name))
                return token
            else:
                assert self.parser.innerHTML
                self.parser.parseError()

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            return self.parser.phases['inTable'].processStartTag(token)

        def endTagTableRowGroup(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if self.tree.elementInScope(token['name'], variant='table'):
                self.clearStackToTableBodyContext()
                self.tree.openElements.pop()
                self.parser.phase = self.parser.phases['inTable']
            else:
                self.parser.parseError('unexpected-end-tag-in-table-body', {'name': token['name']})

        def endTagTable(self, token):
            if False:
                while True:
                    i = 10
            if self.tree.elementInScope('tbody', variant='table') or self.tree.elementInScope('thead', variant='table') or self.tree.elementInScope('tfoot', variant='table'):
                self.clearStackToTableBodyContext()
                self.endTagTableRowGroup(impliedTagToken(self.tree.openElements[-1].name))
                return token
            else:
                assert self.parser.innerHTML
                self.parser.parseError()

        def endTagIgnore(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('unexpected-end-tag-in-table-body', {'name': token['name']})

        def endTagOther(self, token):
            if False:
                for i in range(10):
                    print('nop')
            return self.parser.phases['inTable'].processEndTag(token)

    class InRowPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                return 10
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), (('td', 'th'), self.startTagTableCell), (('caption', 'col', 'colgroup', 'tbody', 'tfoot', 'thead', 'tr'), self.startTagTableOther)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('tr', self.endTagTr), ('table', self.endTagTable), (('tbody', 'tfoot', 'thead'), self.endTagTableRowGroup), (('body', 'caption', 'col', 'colgroup', 'html', 'td', 'th'), self.endTagIgnore)])
            self.endTagHandler.default = self.endTagOther

        def clearStackToTableRowContext(self):
            if False:
                print('Hello World!')
            while self.tree.openElements[-1].name not in ('tr', 'html'):
                self.parser.parseError('unexpected-implied-end-tag-in-table-row', {'name': self.tree.openElements[-1].name})
                self.tree.openElements.pop()

        def ignoreEndTagTr(self):
            if False:
                i = 10
                return i + 15
            return not self.tree.elementInScope('tr', variant='table')

        def processEOF(self):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.phases['inTable'].processEOF()

        def processSpaceCharacters(self, token):
            if False:
                return 10
            return self.parser.phases['inTable'].processSpaceCharacters(token)

        def processCharacters(self, token):
            if False:
                i = 10
                return i + 15
            return self.parser.phases['inTable'].processCharacters(token)

        def startTagTableCell(self, token):
            if False:
                print('Hello World!')
            self.clearStackToTableRowContext()
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inCell']
            self.tree.activeFormattingElements.append(Marker)

        def startTagTableOther(self, token):
            if False:
                i = 10
                return i + 15
            ignoreEndTag = self.ignoreEndTagTr()
            self.endTagTr(impliedTagToken('tr'))
            if not ignoreEndTag:
                return token

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            return self.parser.phases['inTable'].processStartTag(token)

        def endTagTr(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if not self.ignoreEndTagTr():
                self.clearStackToTableRowContext()
                self.tree.openElements.pop()
                self.parser.phase = self.parser.phases['inTableBody']
            else:
                assert self.parser.innerHTML
                self.parser.parseError()

        def endTagTable(self, token):
            if False:
                i = 10
                return i + 15
            ignoreEndTag = self.ignoreEndTagTr()
            self.endTagTr(impliedTagToken('tr'))
            if not ignoreEndTag:
                return token

        def endTagTableRowGroup(self, token):
            if False:
                while True:
                    i = 10
            if self.tree.elementInScope(token['name'], variant='table'):
                self.endTagTr(impliedTagToken('tr'))
                return token
            else:
                self.parser.parseError()

        def endTagIgnore(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-end-tag-in-table-row', {'name': token['name']})

        def endTagOther(self, token):
            if False:
                print('Hello World!')
            return self.parser.phases['inTable'].processEndTag(token)

    class InCellPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                while True:
                    i = 10
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), (('caption', 'col', 'colgroup', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), self.startTagTableOther)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([(('td', 'th'), self.endTagTableCell), (('body', 'caption', 'col', 'colgroup', 'html'), self.endTagIgnore), (('table', 'tbody', 'tfoot', 'thead', 'tr'), self.endTagImply)])
            self.endTagHandler.default = self.endTagOther

        def closeCell(self):
            if False:
                i = 10
                return i + 15
            if self.tree.elementInScope('td', variant='table'):
                self.endTagTableCell(impliedTagToken('td'))
            elif self.tree.elementInScope('th', variant='table'):
                self.endTagTableCell(impliedTagToken('th'))

        def processEOF(self):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.phases['inBody'].processEOF()

        def processCharacters(self, token):
            if False:
                while True:
                    i = 10
            return self.parser.phases['inBody'].processCharacters(token)

        def startTagTableOther(self, token):
            if False:
                return 10
            if self.tree.elementInScope('td', variant='table') or self.tree.elementInScope('th', variant='table'):
                self.closeCell()
                return token
            else:
                assert self.parser.innerHTML
                self.parser.parseError()

        def startTagOther(self, token):
            if False:
                while True:
                    i = 10
            return self.parser.phases['inBody'].processStartTag(token)

        def endTagTableCell(self, token):
            if False:
                print('Hello World!')
            if self.tree.elementInScope(token['name'], variant='table'):
                self.tree.generateImpliedEndTags(token['name'])
                if self.tree.openElements[-1].name != token['name']:
                    self.parser.parseError('unexpected-cell-end-tag', {'name': token['name']})
                    while True:
                        node = self.tree.openElements.pop()
                        if node.name == token['name']:
                            break
                else:
                    self.tree.openElements.pop()
                self.tree.clearActiveFormattingElements()
                self.parser.phase = self.parser.phases['inRow']
            else:
                self.parser.parseError('unexpected-end-tag', {'name': token['name']})

        def endTagIgnore(self, token):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})

        def endTagImply(self, token):
            if False:
                while True:
                    i = 10
            if self.tree.elementInScope(token['name'], variant='table'):
                self.closeCell()
                return token
            else:
                self.parser.parseError()

        def endTagOther(self, token):
            if False:
                for i in range(10):
                    print('nop')
            return self.parser.phases['inBody'].processEndTag(token)

    class InSelectPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                while True:
                    i = 10
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('option', self.startTagOption), ('optgroup', self.startTagOptgroup), ('select', self.startTagSelect), (('input', 'keygen', 'textarea'), self.startTagInput), ('script', self.startTagScript)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('option', self.endTagOption), ('optgroup', self.endTagOptgroup), ('select', self.endTagSelect)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                while True:
                    i = 10
            if self.tree.openElements[-1].name != 'html':
                self.parser.parseError('eof-in-select')
            else:
                assert self.parser.innerHTML

        def processCharacters(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if token['data'] == '\x00':
                return
            self.tree.insertText(token['data'])

        def startTagOption(self, token):
            if False:
                return 10
            if self.tree.openElements[-1].name == 'option':
                self.tree.openElements.pop()
            self.tree.insertElement(token)

        def startTagOptgroup(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if self.tree.openElements[-1].name == 'option':
                self.tree.openElements.pop()
            if self.tree.openElements[-1].name == 'optgroup':
                self.tree.openElements.pop()
            self.tree.insertElement(token)

        def startTagSelect(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-select-in-select')
            self.endTagSelect(impliedTagToken('select'))

        def startTagInput(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-input-in-select')
            if self.tree.elementInScope('select', variant='select'):
                self.endTagSelect(impliedTagToken('select'))
                return token
            else:
                assert self.parser.innerHTML

        def startTagScript(self, token):
            if False:
                while True:
                    i = 10
            return self.parser.phases['inHead'].processStartTag(token)

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-start-tag-in-select', {'name': token['name']})

        def endTagOption(self, token):
            if False:
                for i in range(10):
                    print('nop')
            if self.tree.openElements[-1].name == 'option':
                self.tree.openElements.pop()
            else:
                self.parser.parseError('unexpected-end-tag-in-select', {'name': 'option'})

        def endTagOptgroup(self, token):
            if False:
                i = 10
                return i + 15
            if self.tree.openElements[-1].name == 'option' and self.tree.openElements[-2].name == 'optgroup':
                self.tree.openElements.pop()
            if self.tree.openElements[-1].name == 'optgroup':
                self.tree.openElements.pop()
            else:
                self.parser.parseError('unexpected-end-tag-in-select', {'name': 'optgroup'})

        def endTagSelect(self, token):
            if False:
                return 10
            if self.tree.elementInScope('select', variant='select'):
                node = self.tree.openElements.pop()
                while node.name != 'select':
                    node = self.tree.openElements.pop()
                self.parser.resetInsertionMode()
            else:
                assert self.parser.innerHTML
                self.parser.parseError()

        def endTagOther(self, token):
            if False:
                i = 10
                return i + 15
            self.parser.parseError('unexpected-end-tag-in-select', {'name': token['name']})

    class InSelectInTablePhase(Phase):

        def __init__(self, parser, tree):
            if False:
                while True:
                    i = 10
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([(('caption', 'table', 'tbody', 'tfoot', 'thead', 'tr', 'td', 'th'), self.startTagTable)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([(('caption', 'table', 'tbody', 'tfoot', 'thead', 'tr', 'td', 'th'), self.endTagTable)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                i = 10
                return i + 15
            self.parser.phases['inSelect'].processEOF()

        def processCharacters(self, token):
            if False:
                while True:
                    i = 10
            return self.parser.phases['inSelect'].processCharacters(token)

        def startTagTable(self, token):
            if False:
                return 10
            self.parser.parseError('unexpected-table-element-start-tag-in-select-in-table', {'name': token['name']})
            self.endTagOther(impliedTagToken('select'))
            return token

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            return self.parser.phases['inSelect'].processStartTag(token)

        def endTagTable(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-table-element-end-tag-in-select-in-table', {'name': token['name']})
            if self.tree.elementInScope(token['name'], variant='table'):
                self.endTagOther(impliedTagToken('select'))
                return token

        def endTagOther(self, token):
            if False:
                return 10
            return self.parser.phases['inSelect'].processEndTag(token)

    class InForeignContentPhase(Phase):
        breakoutElements = frozenset(['b', 'big', 'blockquote', 'body', 'br', 'center', 'code', 'dd', 'div', 'dl', 'dt', 'em', 'embed', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'hr', 'i', 'img', 'li', 'listing', 'menu', 'meta', 'nobr', 'ol', 'p', 'pre', 'ruby', 's', 'small', 'span', 'strong', 'strike', 'sub', 'sup', 'table', 'tt', 'u', 'ul', 'var'])

        def __init__(self, parser, tree):
            if False:
                print('Hello World!')
            Phase.__init__(self, parser, tree)

        def adjustSVGTagNames(self, token):
            if False:
                return 10
            replacements = {'altglyph': 'altGlyph', 'altglyphdef': 'altGlyphDef', 'altglyphitem': 'altGlyphItem', 'animatecolor': 'animateColor', 'animatemotion': 'animateMotion', 'animatetransform': 'animateTransform', 'clippath': 'clipPath', 'feblend': 'feBlend', 'fecolormatrix': 'feColorMatrix', 'fecomponenttransfer': 'feComponentTransfer', 'fecomposite': 'feComposite', 'feconvolvematrix': 'feConvolveMatrix', 'fediffuselighting': 'feDiffuseLighting', 'fedisplacementmap': 'feDisplacementMap', 'fedistantlight': 'feDistantLight', 'feflood': 'feFlood', 'fefunca': 'feFuncA', 'fefuncb': 'feFuncB', 'fefuncg': 'feFuncG', 'fefuncr': 'feFuncR', 'fegaussianblur': 'feGaussianBlur', 'feimage': 'feImage', 'femerge': 'feMerge', 'femergenode': 'feMergeNode', 'femorphology': 'feMorphology', 'feoffset': 'feOffset', 'fepointlight': 'fePointLight', 'fespecularlighting': 'feSpecularLighting', 'fespotlight': 'feSpotLight', 'fetile': 'feTile', 'feturbulence': 'feTurbulence', 'foreignobject': 'foreignObject', 'glyphref': 'glyphRef', 'lineargradient': 'linearGradient', 'radialgradient': 'radialGradient', 'textpath': 'textPath'}
            if token['name'] in replacements:
                token['name'] = replacements[token['name']]

        def processCharacters(self, token):
            if False:
                print('Hello World!')
            if token['data'] == '\x00':
                token['data'] = '�'
            elif self.parser.framesetOK and any((char not in spaceCharacters for char in token['data'])):
                self.parser.framesetOK = False
            Phase.processCharacters(self, token)

        def processStartTag(self, token):
            if False:
                i = 10
                return i + 15
            currentNode = self.tree.openElements[-1]
            if token['name'] in self.breakoutElements or (token['name'] == 'font' and set(token['data'].keys()) & set(['color', 'face', 'size'])):
                self.parser.parseError('unexpected-html-element-in-foreign-content', {'name': token['name']})
                while self.tree.openElements[-1].namespace != self.tree.defaultNamespace and (not self.parser.isHTMLIntegrationPoint(self.tree.openElements[-1])) and (not self.parser.isMathMLTextIntegrationPoint(self.tree.openElements[-1])):
                    self.tree.openElements.pop()
                return token
            else:
                if currentNode.namespace == namespaces['mathml']:
                    self.parser.adjustMathMLAttributes(token)
                elif currentNode.namespace == namespaces['svg']:
                    self.adjustSVGTagNames(token)
                    self.parser.adjustSVGAttributes(token)
                self.parser.adjustForeignAttributes(token)
                token['namespace'] = currentNode.namespace
                self.tree.insertElement(token)
                if token['selfClosing']:
                    self.tree.openElements.pop()
                    token['selfClosingAcknowledged'] = True

        def processEndTag(self, token):
            if False:
                return 10
            nodeIndex = len(self.tree.openElements) - 1
            node = self.tree.openElements[-1]
            if node.name.translate(asciiUpper2Lower) != token['name']:
                self.parser.parseError('unexpected-end-tag', {'name': token['name']})
            while True:
                if node.name.translate(asciiUpper2Lower) == token['name']:
                    if self.parser.phase == self.parser.phases['inTableText']:
                        self.parser.phase.flushCharacters()
                        self.parser.phase = self.parser.phase.originalPhase
                    while self.tree.openElements.pop() != node:
                        assert self.tree.openElements
                    new_token = None
                    break
                nodeIndex -= 1
                node = self.tree.openElements[nodeIndex]
                if node.namespace != self.tree.defaultNamespace:
                    continue
                else:
                    new_token = self.parser.phase.processEndTag(token)
                    break
            return new_token

    class AfterBodyPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('html', self.endTagHtml)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                while True:
                    i = 10
            pass

        def processComment(self, token):
            if False:
                return 10
            self.tree.insertComment(token, self.tree.openElements[0])

        def processCharacters(self, token):
            if False:
                return 10
            self.parser.parseError('unexpected-char-after-body')
            self.parser.phase = self.parser.phases['inBody']
            return token

        def startTagHtml(self, token):
            if False:
                print('Hello World!')
            return self.parser.phases['inBody'].processStartTag(token)

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('unexpected-start-tag-after-body', {'name': token['name']})
            self.parser.phase = self.parser.phases['inBody']
            return token

        def endTagHtml(self, name):
            if False:
                i = 10
                return i + 15
            if self.parser.innerHTML:
                self.parser.parseError('unexpected-end-tag-after-body-innerhtml')
            else:
                self.parser.phase = self.parser.phases['afterAfterBody']

        def endTagOther(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-end-tag-after-body', {'name': token['name']})
            self.parser.phase = self.parser.phases['inBody']
            return token

    class InFramesetPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                print('Hello World!')
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('frameset', self.startTagFrameset), ('frame', self.startTagFrame), ('noframes', self.startTagNoframes)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('frameset', self.endTagFrameset)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                while True:
                    i = 10
            if self.tree.openElements[-1].name != 'html':
                self.parser.parseError('eof-in-frameset')
            else:
                assert self.parser.innerHTML

        def processCharacters(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-char-in-frameset')

        def startTagFrameset(self, token):
            if False:
                while True:
                    i = 10
            self.tree.insertElement(token)

        def startTagFrame(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.tree.insertElement(token)
            self.tree.openElements.pop()

        def startTagNoframes(self, token):
            if False:
                i = 10
                return i + 15
            return self.parser.phases['inBody'].processStartTag(token)

        def startTagOther(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('unexpected-start-tag-in-frameset', {'name': token['name']})

        def endTagFrameset(self, token):
            if False:
                i = 10
                return i + 15
            if self.tree.openElements[-1].name == 'html':
                self.parser.parseError('unexpected-frameset-in-frameset-innerhtml')
            else:
                self.tree.openElements.pop()
            if not self.parser.innerHTML and self.tree.openElements[-1].name != 'frameset':
                self.parser.phase = self.parser.phases['afterFrameset']

        def endTagOther(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('unexpected-end-tag-in-frameset', {'name': token['name']})

    class AfterFramesetPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                return 10
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('noframes', self.startTagNoframes)])
            self.startTagHandler.default = self.startTagOther
            self.endTagHandler = _utils.MethodDispatcher([('html', self.endTagHtml)])
            self.endTagHandler.default = self.endTagOther

        def processEOF(self):
            if False:
                print('Hello World!')
            pass

        def processCharacters(self, token):
            if False:
                return 10
            self.parser.parseError('unexpected-char-after-frameset')

        def startTagNoframes(self, token):
            if False:
                print('Hello World!')
            return self.parser.phases['inHead'].processStartTag(token)

        def startTagOther(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('unexpected-start-tag-after-frameset', {'name': token['name']})

        def endTagHtml(self, token):
            if False:
                return 10
            self.parser.phase = self.parser.phases['afterAfterFrameset']

        def endTagOther(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('unexpected-end-tag-after-frameset', {'name': token['name']})

    class AfterAfterBodyPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                i = 10
                return i + 15
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml)])
            self.startTagHandler.default = self.startTagOther

        def processEOF(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def processComment(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.tree.insertComment(token, self.tree.document)

        def processSpaceCharacters(self, token):
            if False:
                i = 10
                return i + 15
            return self.parser.phases['inBody'].processSpaceCharacters(token)

        def processCharacters(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('expected-eof-but-got-char')
            self.parser.phase = self.parser.phases['inBody']
            return token

        def startTagHtml(self, token):
            if False:
                return 10
            return self.parser.phases['inBody'].processStartTag(token)

        def startTagOther(self, token):
            if False:
                while True:
                    i = 10
            self.parser.parseError('expected-eof-but-got-start-tag', {'name': token['name']})
            self.parser.phase = self.parser.phases['inBody']
            return token

        def processEndTag(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.parser.parseError('expected-eof-but-got-end-tag', {'name': token['name']})
            self.parser.phase = self.parser.phases['inBody']
            return token

    class AfterAfterFramesetPhase(Phase):

        def __init__(self, parser, tree):
            if False:
                while True:
                    i = 10
            Phase.__init__(self, parser, tree)
            self.startTagHandler = _utils.MethodDispatcher([('html', self.startTagHtml), ('noframes', self.startTagNoFrames)])
            self.startTagHandler.default = self.startTagOther

        def processEOF(self):
            if False:
                print('Hello World!')
            pass

        def processComment(self, token):
            if False:
                print('Hello World!')
            self.tree.insertComment(token, self.tree.document)

        def processSpaceCharacters(self, token):
            if False:
                i = 10
                return i + 15
            return self.parser.phases['inBody'].processSpaceCharacters(token)

        def processCharacters(self, token):
            if False:
                return 10
            self.parser.parseError('expected-eof-but-got-char')

        def startTagHtml(self, token):
            if False:
                return 10
            return self.parser.phases['inBody'].processStartTag(token)

        def startTagNoFrames(self, token):
            if False:
                return 10
            return self.parser.phases['inHead'].processStartTag(token)

        def startTagOther(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('expected-eof-but-got-start-tag', {'name': token['name']})

        def processEndTag(self, token):
            if False:
                print('Hello World!')
            self.parser.parseError('expected-eof-but-got-end-tag', {'name': token['name']})
    return {'initial': InitialPhase, 'beforeHtml': BeforeHtmlPhase, 'beforeHead': BeforeHeadPhase, 'inHead': InHeadPhase, 'inHeadNoscript': InHeadNoscriptPhase, 'afterHead': AfterHeadPhase, 'inBody': InBodyPhase, 'text': TextPhase, 'inTable': InTablePhase, 'inTableText': InTableTextPhase, 'inCaption': InCaptionPhase, 'inColumnGroup': InColumnGroupPhase, 'inTableBody': InTableBodyPhase, 'inRow': InRowPhase, 'inCell': InCellPhase, 'inSelect': InSelectPhase, 'inSelectInTable': InSelectInTablePhase, 'inForeignContent': InForeignContentPhase, 'afterBody': AfterBodyPhase, 'inFrameset': InFramesetPhase, 'afterFrameset': AfterFramesetPhase, 'afterAfterBody': AfterAfterBodyPhase, 'afterAfterFrameset': AfterAfterFramesetPhase}

def adjust_attributes(token, replacements):
    if False:
        print('Hello World!')
    needs_adjustment = viewkeys(token['data']) & viewkeys(replacements)
    if needs_adjustment:
        token['data'] = OrderedDict(((replacements.get(k, k), v) for (k, v) in token['data'].items()))

def impliedTagToken(name, type='EndTag', attributes=None, selfClosing=False):
    if False:
        i = 10
        return i + 15
    if attributes is None:
        attributes = {}
    return {'type': tokenTypes[type], 'name': name, 'data': attributes, 'selfClosing': selfClosing}

class ParseError(Exception):
    """Error in parsed document"""
    pass