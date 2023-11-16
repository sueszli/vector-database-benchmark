import libxml2mod
import types
import sys

class libxmlError(Exception):
    pass

def checkWrapper(obj):
    if False:
        for i in range(10):
            print('nop')
    try:
        n = type(_obj).__name__
        if n != 'PyCObject' and n != 'PyCapsule':
            return 1
    except:
        return 0
    return 0

def pos_id(o):
    if False:
        while True:
            i = 10
    i = id(o)
    if i < 0:
        return sys.maxsize - i
    return i

class treeError(libxmlError):

    def __init__(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.msg = msg

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.msg

class parserError(libxmlError):

    def __init__(self, msg):
        if False:
            i = 10
            return i + 15
        self.msg = msg

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.msg

class uriError(libxmlError):

    def __init__(self, msg):
        if False:
            i = 10
            return i + 15
        self.msg = msg

    def __str__(self):
        if False:
            return 10
        return self.msg

class xpathError(libxmlError):

    def __init__(self, msg):
        if False:
            return 10
        self.msg = msg

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.msg

class ioWrapper:

    def __init__(self, _obj):
        if False:
            print('Hello World!')
        self.__io = _obj
        self._o = None

    def io_close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.__io == None:
            return -1
        self.__io.close()
        self.__io = None
        return 0

    def io_flush(self):
        if False:
            for i in range(10):
                print('nop')
        if self.__io == None:
            return -1
        self.__io.flush()
        return 0

    def io_read(self, len=-1):
        if False:
            print('Hello World!')
        if self.__io == None:
            return -1
        try:
            if len < 0:
                ret = self.__io.read()
            else:
                ret = self.__io.read(len)
        except Exception:
            import sys
            e = sys.exc_info()[1]
            print('failed to read from Python:', type(e))
            print('on IO:', self.__io)
            self.__io == None
            return -1
        return ret

    def io_write(self, str, len=-1):
        if False:
            i = 10
            return i + 15
        if self.__io == None:
            return -1
        if len < 0:
            return self.__io.write(str)
        return self.__io.write(str, len)

class ioReadWrapper(ioWrapper):

    def __init__(self, _obj, enc=''):
        if False:
            i = 10
            return i + 15
        ioWrapper.__init__(self, _obj)
        self._o = libxml2mod.xmlCreateInputBuffer(self, enc)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        print('__del__')
        self.io_close()
        if self._o != None:
            libxml2mod.xmlFreeParserInputBuffer(self._o)
        self._o = None

    def close(self):
        if False:
            i = 10
            return i + 15
        self.io_close()
        if self._o != None:
            libxml2mod.xmlFreeParserInputBuffer(self._o)
        self._o = None

class ioWriteWrapper(ioWrapper):

    def __init__(self, _obj, enc=''):
        if False:
            i = 10
            return i + 15
        if type(_obj) == type(''):
            print('write io from a string')
            self.o = None
        elif type(_obj).__name__ == 'PyCapsule':
            file = libxml2mod.outputBufferGetPythonFile(_obj)
            if file != None:
                ioWrapper.__init__(self, file)
            else:
                ioWrapper.__init__(self, _obj)
            self._o = _obj
        else:
            file = libxml2mod.outputBufferGetPythonFile(_obj)
            if file != None:
                ioWrapper.__init__(self, file)
            else:
                ioWrapper.__init__(self, _obj)
            self._o = _obj

    def __del__(self):
        if False:
            print('Hello World!')
        self.io_close()
        if self._o != None:
            libxml2mod.xmlOutputBufferClose(self._o)
        self._o = None

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        self.io_flush()
        if self._o != None:
            libxml2mod.xmlOutputBufferClose(self._o)
        self._o = None

    def close(self):
        if False:
            while True:
                i = 10
        self.io_flush()
        if self._o != None:
            libxml2mod.xmlOutputBufferClose(self._o)
        self._o = None

class SAXCallback:
    """Base class for SAX handlers"""

    def startDocument(self):
        if False:
            print('Hello World!')
        'called at the start of the document'
        pass

    def endDocument(self):
        if False:
            return 10
        'called at the end of the document'
        pass

    def startElement(self, tag, attrs):
        if False:
            print('Hello World!')
        "called at the start of every element, tag is the name of\n           the element, attrs is a dictionary of the element's attributes"
        pass

    def endElement(self, tag):
        if False:
            for i in range(10):
                print('nop')
        'called at the start of every element, tag is the name of\n           the element'
        pass

    def characters(self, data):
        if False:
            while True:
                i = 10
        'called when character data have been read, data is the string\n           containing the data, multiple consecutive characters() callback\n           are possible.'
        pass

    def cdataBlock(self, data):
        if False:
            for i in range(10):
                print('nop')
        'called when CDATA section have been read, data is the string\n           containing the data, multiple consecutive cdataBlock() callback\n           are possible.'
        pass

    def reference(self, name):
        if False:
            i = 10
            return i + 15
        'called when an entity reference has been found'
        pass

    def ignorableWhitespace(self, data):
        if False:
            return 10
        'called when potentially ignorable white spaces have been found'
        pass

    def processingInstruction(self, target, data):
        if False:
            for i in range(10):
                print('nop')
        'called when a PI has been found, target contains the PI name and\n           data is the associated data in the PI'
        pass

    def comment(self, content):
        if False:
            i = 10
            return i + 15
        'called when a comment has been found, content contains the comment'
        pass

    def externalSubset(self, name, externalID, systemID):
        if False:
            i = 10
            return i + 15
        'called when a DOCTYPE declaration has been found, name is the\n           DTD name and externalID, systemID are the DTD public and system\n           identifier for that DTd if available'
        pass

    def internalSubset(self, name, externalID, systemID):
        if False:
            for i in range(10):
                print('nop')
        'called when a DOCTYPE declaration has been found, name is the\n           DTD name and externalID, systemID are the DTD public and system\n           identifier for that DTD if available'
        pass

    def entityDecl(self, name, type, externalID, systemID, content):
        if False:
            print('Hello World!')
        "called when an ENTITY declaration has been found, name is the\n           entity name and externalID, systemID are the entity public and\n           system identifier for that entity if available, type indicates\n           the entity type, and content reports it's string content"
        pass

    def notationDecl(self, name, externalID, systemID):
        if False:
            while True:
                i = 10
        'called when an NOTATION declaration has been found, name is the\n           notation name and externalID, systemID are the notation public and\n           system identifier for that notation if available'
        pass

    def attributeDecl(self, elem, name, type, defi, defaultValue, nameList):
        if False:
            return 10
        'called when an ATTRIBUTE definition has been found'
        pass

    def elementDecl(self, name, type, content):
        if False:
            while True:
                i = 10
        'called when an ELEMENT definition has been found'
        pass

    def entityDecl(self, name, publicId, systemID, notationName):
        if False:
            i = 10
            return i + 15
        'called when an unparsed ENTITY declaration has been found,\n           name is the entity name and publicId,, systemID are the entity\n           public and system identifier for that entity if available,\n           and notationName indicate the associated NOTATION'
        pass

    def warning(self, msg):
        if False:
            while True:
                i = 10
        pass

    def error(self, msg):
        if False:
            i = 10
            return i + 15
        raise parserError(msg)

    def fatalError(self, msg):
        if False:
            while True:
                i = 10
        raise parserError(msg)

class xmlCore:

    def __init__(self, _obj=None):
        if False:
            print('Hello World!')
        if _obj != None:
            self._o = _obj
            return
        self._o = None

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other == None:
            return False
        ret = libxml2mod.compareNodesEqual(self._o, other._o)
        if ret == None:
            return False
        return ret == True

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        if other == None:
            return True
        ret = libxml2mod.compareNodesEqual(self._o, other._o)
        return not ret

    def __hash__(self):
        if False:
            return 10
        ret = libxml2mod.nodeHash(self._o)
        return ret

    def __str__(self):
        if False:
            return 10
        return self.serialize()

    def get_parent(self):
        if False:
            i = 10
            return i + 15
        ret = libxml2mod.parent(self._o)
        if ret == None:
            return None
        return nodeWrap(ret)

    def get_children(self):
        if False:
            i = 10
            return i + 15
        ret = libxml2mod.children(self._o)
        if ret == None:
            return None
        return nodeWrap(ret)

    def get_last(self):
        if False:
            print('Hello World!')
        ret = libxml2mod.last(self._o)
        if ret == None:
            return None
        return nodeWrap(ret)

    def get_next(self):
        if False:
            while True:
                i = 10
        ret = libxml2mod.next(self._o)
        if ret == None:
            return None
        return nodeWrap(ret)

    def get_properties(self):
        if False:
            return 10
        ret = libxml2mod.properties(self._o)
        if ret == None:
            return None
        return xmlAttr(_obj=ret)

    def get_prev(self):
        if False:
            print('Hello World!')
        ret = libxml2mod.prev(self._o)
        if ret == None:
            return None
        return nodeWrap(ret)

    def get_content(self):
        if False:
            for i in range(10):
                print('nop')
        return libxml2mod.xmlNodeGetContent(self._o)
    getContent = get_content

    def get_name(self):
        if False:
            for i in range(10):
                print('nop')
        return libxml2mod.name(self._o)

    def get_type(self):
        if False:
            print('Hello World!')
        return libxml2mod.type(self._o)

    def get_doc(self):
        if False:
            print('Hello World!')
        ret = libxml2mod.doc(self._o)
        if ret == None:
            if self.type in ['document_xml', 'document_html']:
                return xmlDoc(_obj=self._o)
            else:
                return None
        return xmlDoc(_obj=ret)
    import sys
    if float(sys.version[0:3]) < 2.2:

        def __getattr__(self, attr):
            if False:
                while True:
                    i = 10
            if attr == 'parent':
                ret = libxml2mod.parent(self._o)
                if ret == None:
                    return None
                return nodeWrap(ret)
            elif attr == 'properties':
                ret = libxml2mod.properties(self._o)
                if ret == None:
                    return None
                return xmlAttr(_obj=ret)
            elif attr == 'children':
                ret = libxml2mod.children(self._o)
                if ret == None:
                    return None
                return nodeWrap(ret)
            elif attr == 'last':
                ret = libxml2mod.last(self._o)
                if ret == None:
                    return None
                return nodeWrap(ret)
            elif attr == 'next':
                ret = libxml2mod.next(self._o)
                if ret == None:
                    return None
                return nodeWrap(ret)
            elif attr == 'prev':
                ret = libxml2mod.prev(self._o)
                if ret == None:
                    return None
                return nodeWrap(ret)
            elif attr == 'content':
                return libxml2mod.xmlNodeGetContent(self._o)
            elif attr == 'name':
                return libxml2mod.name(self._o)
            elif attr == 'type':
                return libxml2mod.type(self._o)
            elif attr == 'doc':
                ret = libxml2mod.doc(self._o)
                if ret == None:
                    if self.type == 'document_xml' or self.type == 'document_html':
                        return xmlDoc(_obj=self._o)
                    else:
                        return None
                return xmlDoc(_obj=ret)
            raise AttributeError(attr)
    else:
        parent = property(get_parent, None, None, 'Parent node')
        children = property(get_children, None, None, 'First child node')
        last = property(get_last, None, None, 'Last sibling node')
        next = property(get_next, None, None, 'Next sibling node')
        prev = property(get_prev, None, None, 'Previous sibling node')
        properties = property(get_properties, None, None, 'List of properies')
        content = property(get_content, None, None, 'Content of this node')
        name = property(get_name, None, None, 'Node name')
        type = property(get_type, None, None, 'Node type')
        doc = property(get_doc, None, None, 'The document this node belongs to')

    def serialize(self, encoding=None, format=0):
        if False:
            for i in range(10):
                print('nop')
        return libxml2mod.serializeNode(self._o, encoding, format)

    def saveTo(self, file, encoding=None, format=0):
        if False:
            i = 10
            return i + 15
        return libxml2mod.saveNodeTo(self._o, file, encoding, format)

    def c14nMemory(self, nodes=None, exclusive=0, prefixes=None, with_comments=0):
        if False:
            i = 10
            return i + 15
        if nodes:
            nodes = [n._o for n in nodes]
        return libxml2mod.xmlC14NDocDumpMemory(self.get_doc()._o, nodes, exclusive != 0, prefixes, with_comments != 0)

    def c14nSaveTo(self, file, nodes=None, exclusive=0, prefixes=None, with_comments=0):
        if False:
            print('Hello World!')
        if nodes:
            nodes = [n._o for n in nodes]
        return libxml2mod.xmlC14NDocSaveTo(self.get_doc()._o, nodes, exclusive != 0, prefixes, with_comments != 0, file)

    def xpathEval(self, expr):
        if False:
            return 10
        doc = self.doc
        if doc == None:
            return None
        ctxt = doc.xpathNewContext()
        ctxt.setContextNode(self)
        res = ctxt.xpathEval(expr)
        ctxt.xpathFreeContext()
        return res

    def xpathEval2(self, expr):
        if False:
            return 10
        return self.xpathEval(expr)

    def removeNsDef(self, href):
        if False:
            while True:
                i = 10
        '\n        Remove a namespace definition from a node.  If href is None,\n        remove all of the ns definitions on that node.  The removed\n        namespaces are returned as a linked list.\n\n        Note: If any child nodes referred to the removed namespaces,\n        they will be left with dangling links.  You should call\n        renconciliateNs() to fix those pointers.\n\n        Note: This method does not free memory taken by the ns\n        definitions.  You will need to free it manually with the\n        freeNsList() method on the returns xmlNs object.\n        '
        ret = libxml2mod.xmlNodeRemoveNsDef(self._o, href)
        if ret is None:
            return None
        __tmp = xmlNs(_obj=ret)
        return __tmp

    def walk_depth_first(self):
        if False:
            i = 10
            return i + 15
        return xmlCoreDepthFirstItertor(self)

    def walk_breadth_first(self):
        if False:
            i = 10
            return i + 15
        return xmlCoreBreadthFirstItertor(self)
    __iter__ = walk_depth_first

    def free(self):
        if False:
            print('Hello World!')
        try:
            self.doc._ctxt.xpathFreeContext()
        except:
            pass
        libxml2mod.xmlFreeDoc(self._o)

class xmlCoreDepthFirstItertor:

    def __init__(self, node):
        if False:
            i = 10
            return i + 15
        self.node = node
        self.parents = []

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def next(self):
        if False:
            while True:
                i = 10
        while 1:
            if self.node:
                ret = self.node
                self.parents.append(self.node)
                self.node = self.node.children
                return ret
            try:
                parent = self.parents.pop()
            except IndexError:
                raise StopIteration
            self.node = parent.next

class xmlCoreBreadthFirstItertor:

    def __init__(self, node):
        if False:
            print('Hello World!')
        self.node = node
        self.parents = []

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        while 1:
            if self.node:
                ret = self.node
                self.parents.append(self.node)
                self.node = self.node.next
                return ret
            try:
                parent = self.parents.pop()
            except IndexError:
                raise StopIteration
            self.node = parent.children

def nodeWrap(o):
    if False:
        print('Hello World!')
    name = libxml2mod.type(o)
    if name == 'element' or name == 'text':
        return xmlNode(_obj=o)
    if name == 'attribute':
        return xmlAttr(_obj=o)
    if name[0:8] == 'document':
        return xmlDoc(_obj=o)
    if name == 'namespace':
        return xmlNs(_obj=o)
    if name == 'elem_decl':
        return xmlElement(_obj=o)
    if name == 'attribute_decl':
        return xmlAttribute(_obj=o)
    if name == 'entity_decl':
        return xmlEntity(_obj=o)
    if name == 'dtd':
        return xmlDtd(_obj=o)
    return xmlNode(_obj=o)

def xpathObjectRet(o):
    if False:
        i = 10
        return i + 15
    otype = type(o)
    if otype == type([]):
        ret = list(map(xpathObjectRet, o))
        return ret
    elif otype == type(()):
        ret = list(map(xpathObjectRet, o))
        return tuple(ret)
    elif otype == type('') or otype == type(0) or otype == type(0.0):
        return o
    else:
        return nodeWrap(o)

def registerXPathFunction(ctxt, name, ns_uri, f):
    if False:
        for i in range(10):
            print('nop')
    ret = libxml2mod.xmlRegisterXPathFunction(ctxt, name, ns_uri, f)
PARSER_LOADDTD = 1
PARSER_DEFAULTATTRS = 2
PARSER_VALIDATE = 3
PARSER_SUBST_ENTITIES = 4
PARSER_SEVERITY_VALIDITY_WARNING = 1
PARSER_SEVERITY_VALIDITY_ERROR = 2
PARSER_SEVERITY_WARNING = 3
PARSER_SEVERITY_ERROR = 4

def registerErrorHandler(f, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Register a Python written function to for error reporting.\n       The function is called back as f(ctx, error). '
    import sys
    if 'libxslt' not in sys.modules:
        ret = libxml2mod.xmlRegisterErrorHandler(f, ctx)
    else:
        import libxslt
        ret = libxslt.registerErrorHandler(f, ctx)
    return ret

class parserCtxtCore:

    def __init__(self, _obj=None):
        if False:
            for i in range(10):
                print('nop')
        if _obj != None:
            self._o = _obj
            return
        self._o = None

    def __del__(self):
        if False:
            return 10
        if self._o != None:
            libxml2mod.xmlFreeParserCtxt(self._o)
        self._o = None

    def setErrorHandler(self, f, arg):
        if False:
            print('Hello World!')
        'Register an error handler that will be called back as\n           f(arg,msg,severity,reserved).\n\n           @reserved is currently always None.'
        libxml2mod.xmlParserCtxtSetErrorHandler(self._o, f, arg)

    def getErrorHandler(self):
        if False:
            for i in range(10):
                print('nop')
        'Return (f,arg) as previously registered with setErrorHandler\n           or (None,None).'
        return libxml2mod.xmlParserCtxtGetErrorHandler(self._o)

    def addLocalCatalog(self, uri):
        if False:
            return 10
        'Register a local catalog with the parser'
        return libxml2mod.addLocalCatalog(self._o, uri)

class ValidCtxtCore:

    def __init__(self, *args, **kw):
        if False:
            while True:
                i = 10
        pass

    def setValidityErrorHandler(self, err_func, warn_func, arg=None):
        if False:
            i = 10
            return i + 15
        '\n        Register error and warning handlers for DTD validation.\n        These will be called back as f(msg,arg)\n        '
        libxml2mod.xmlSetValidErrors(self._o, err_func, warn_func, arg)

class SchemaValidCtxtCore:

    def __init__(self, *args, **kw):
        if False:
            print('Hello World!')
        pass

    def setValidityErrorHandler(self, err_func, warn_func, arg=None):
        if False:
            while True:
                i = 10
        '\n        Register error and warning handlers for Schema validation.\n        These will be called back as f(msg,arg)\n        '
        libxml2mod.xmlSchemaSetValidErrors(self._o, err_func, warn_func, arg)

class relaxNgValidCtxtCore:

    def __init__(self, *args, **kw):
        if False:
            while True:
                i = 10
        pass

    def setValidityErrorHandler(self, err_func, warn_func, arg=None):
        if False:
            i = 10
            return i + 15
        '\n        Register error and warning handlers for RelaxNG validation.\n        These will be called back as f(msg,arg)\n        '
        libxml2mod.xmlRelaxNGSetValidErrors(self._o, err_func, warn_func, arg)

def _xmlTextReaderErrorFunc(xxx_todo_changeme, msg, severity, locator):
    if False:
        i = 10
        return i + 15
    'Intermediate callback to wrap the locator'
    (f, arg) = xxx_todo_changeme
    return f(arg, msg, severity, xmlTextReaderLocator(locator))

class xmlTextReaderCore:

    def __init__(self, _obj=None):
        if False:
            i = 10
            return i + 15
        self.input = None
        if _obj != None:
            self._o = _obj
            return
        self._o = None

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._o != None:
            libxml2mod.xmlFreeTextReader(self._o)
        self._o = None

    def SetErrorHandler(self, f, arg):
        if False:
            while True:
                i = 10
        'Register an error handler that will be called back as\n           f(arg,msg,severity,locator).'
        if f is None:
            libxml2mod.xmlTextReaderSetErrorHandler(self._o, None, None)
        else:
            libxml2mod.xmlTextReaderSetErrorHandler(self._o, _xmlTextReaderErrorFunc, (f, arg))

    def GetErrorHandler(self):
        if False:
            while True:
                i = 10
        'Return (f,arg) as previously registered with setErrorHandler\n           or (None,None).'
        (f, arg) = libxml2mod.xmlTextReaderGetErrorHandler(self._o)
        if f is None:
            return (None, None)
        else:
            return arg

def cleanupParser():
    if False:
        return 10
    libxml2mod.xmlPythonCleanupParser()
__input_callbacks = []

def registerInputCallback(func):
    if False:
        while True:
            i = 10

    def findOpenCallback(URI):
        if False:
            for i in range(10):
                print('nop')
        for cb in reversed(__input_callbacks):
            o = cb(URI)
            if o is not None:
                return o
    libxml2mod.xmlRegisterInputCallback(findOpenCallback)
    __input_callbacks.append(func)

def popInputCallbacks():
    if False:
        print('Hello World!')
    if len(__input_callbacks) > 0:
        __input_callbacks.pop()
    if len(__input_callbacks) == 0:
        libxml2mod.xmlUnregisterInputCallback()