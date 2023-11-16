"""
A library for performing interesting tasks with DOM objects.

This module is now deprecated.
"""
import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
warningString = 'twisted.web.domhelpers was deprecated at {}'.format(getVersionString(Version('Twisted', 23, 10, 0)))
warnings.warn(warningString, DeprecationWarning, stacklevel=3)
escape
getElementsByTagName

class NodeLookupError(Exception):
    pass

def substitute(request, node, subs):
    if False:
        print('Hello World!')
    "\n    Look through the given node's children for strings, and\n    attempt to do string substitution with the given parameter.\n    "
    for child in node.childNodes:
        if hasattr(child, 'nodeValue') and child.nodeValue:
            child.replaceData(0, len(child.nodeValue), child.nodeValue % subs)
        substitute(request, child, subs)

def _get(node, nodeId, nodeAttrs=('id', 'class', 'model', 'pattern')):
    if False:
        return 10
    '\n    (internal) Get a node with the specified C{nodeId} as any of the C{class},\n    C{id} or C{pattern} attributes.\n    '
    if hasattr(node, 'hasAttributes') and node.hasAttributes():
        for nodeAttr in nodeAttrs:
            if str(node.getAttribute(nodeAttr)) == nodeId:
                return node
    if node.hasChildNodes():
        if hasattr(node.childNodes, 'length'):
            length = node.childNodes.length
        else:
            length = len(node.childNodes)
        for childNum in range(length):
            result = _get(node.childNodes[childNum], nodeId)
            if result:
                return result

def get(node, nodeId):
    if False:
        print('Hello World!')
    '\n    Get a node with the specified C{nodeId} as any of the C{class},\n    C{id} or C{pattern} attributes. If there is no such node, raise\n    L{NodeLookupError}.\n    '
    result = _get(node, nodeId)
    if result:
        return result
    raise NodeLookupError(nodeId)

def getIfExists(node, nodeId):
    if False:
        while True:
            i = 10
    '\n    Get a node with the specified C{nodeId} as any of the C{class},\n    C{id} or C{pattern} attributes.  If there is no such node, return\n    L{None}.\n    '
    return _get(node, nodeId)

def getAndClear(node, nodeId):
    if False:
        print('Hello World!')
    'Get a node with the specified C{nodeId} as any of the C{class},\n    C{id} or C{pattern} attributes. If there is no such node, raise\n    L{NodeLookupError}. Remove all child nodes before returning.\n    '
    result = get(node, nodeId)
    if result:
        clearNode(result)
    return result

def clearNode(node):
    if False:
        return 10
    '\n    Remove all children from the given node.\n    '
    node.childNodes[:] = []

def locateNodes(nodeList, key, value, noNesting=1):
    if False:
        i = 10
        return i + 15
    '\n    Find subnodes in the given node where the given attribute\n    has the given value.\n    '
    returnList = []
    if not isinstance(nodeList, type([])):
        return locateNodes(nodeList.childNodes, key, value, noNesting)
    for childNode in nodeList:
        if not hasattr(childNode, 'getAttribute'):
            continue
        if str(childNode.getAttribute(key)) == value:
            returnList.append(childNode)
            if noNesting:
                continue
        returnList.extend(locateNodes(childNode, key, value, noNesting))
    return returnList

def superSetAttribute(node, key, value):
    if False:
        i = 10
        return i + 15
    if not hasattr(node, 'setAttribute'):
        return
    node.setAttribute(key, value)
    if node.hasChildNodes():
        for child in node.childNodes:
            superSetAttribute(child, key, value)

def superPrependAttribute(node, key, value):
    if False:
        print('Hello World!')
    if not hasattr(node, 'setAttribute'):
        return
    old = node.getAttribute(key)
    if old:
        node.setAttribute(key, value + '/' + old)
    else:
        node.setAttribute(key, value)
    if node.hasChildNodes():
        for child in node.childNodes:
            superPrependAttribute(child, key, value)

def superAppendAttribute(node, key, value):
    if False:
        i = 10
        return i + 15
    if not hasattr(node, 'setAttribute'):
        return
    old = node.getAttribute(key)
    if old:
        node.setAttribute(key, old + '/' + value)
    else:
        node.setAttribute(key, value)
    if node.hasChildNodes():
        for child in node.childNodes:
            superAppendAttribute(child, key, value)

def gatherTextNodes(iNode, dounescape=0, joinWith=''):
    if False:
        for i in range(10):
            print('nop')
    "Visit each child node and collect its text data, if any, into a string.\n    For example::\n        >>> doc=microdom.parseString('<a>1<b>2<c>3</c>4</b></a>')\n        >>> gatherTextNodes(doc.documentElement)\n        '1234'\n    With dounescape=1, also convert entities back into normal characters.\n    @return: the gathered nodes as a single string\n    @rtype: str"
    gathered = []
    gathered_append = gathered.append
    slice = [iNode]
    while len(slice) > 0:
        c = slice.pop(0)
        if hasattr(c, 'nodeValue') and c.nodeValue is not None:
            if dounescape:
                val = unescape(c.nodeValue)
            else:
                val = c.nodeValue
            gathered_append(val)
        slice[:0] = c.childNodes
    return joinWith.join(gathered)

class RawText(microdom.Text):
    """This is an evil and horrible speed hack. Basically, if you have a big
    chunk of XML that you want to insert into the DOM, but you don't want to
    incur the cost of parsing it, you can construct one of these and insert it
    into the DOM. This will most certainly only work with microdom as the API
    for converting nodes to xml is different in every DOM implementation.

    This could be improved by making this class a Lazy parser, so if you
    inserted this into the DOM and then later actually tried to mutate this
    node, it would be parsed then.
    """

    def writexml(self, writer, indent='', addindent='', newl='', strip=0, nsprefixes=None, namespace=None):
        if False:
            for i in range(10):
                print('nop')
        writer.write(f'{indent}{self.data}{newl}')

def findNodes(parent, matcher, accum=None):
    if False:
        i = 10
        return i + 15
    if accum is None:
        accum = []
    if not parent.hasChildNodes():
        return accum
    for child in parent.childNodes:
        if matcher(child):
            accum.append(child)
        findNodes(child, matcher, accum)
    return accum

def findNodesShallowOnMatch(parent, matcher, recurseMatcher, accum=None):
    if False:
        return 10
    if accum is None:
        accum = []
    if not parent.hasChildNodes():
        return accum
    for child in parent.childNodes:
        if matcher(child):
            accum.append(child)
        if recurseMatcher(child):
            findNodesShallowOnMatch(child, matcher, recurseMatcher, accum)
    return accum

def findNodesShallow(parent, matcher, accum=None):
    if False:
        print('Hello World!')
    if accum is None:
        accum = []
    if not parent.hasChildNodes():
        return accum
    for child in parent.childNodes:
        if matcher(child):
            accum.append(child)
        else:
            findNodes(child, matcher, accum)
    return accum

def findElementsWithAttributeShallow(parent, attribute):
    if False:
        print('Hello World!')
    '\n    Return an iterable of the elements which are direct children of C{parent}\n    and which have the C{attribute} attribute.\n    '
    return findNodesShallow(parent, lambda n: getattr(n, 'tagName', None) is not None and n.hasAttribute(attribute))

def findElements(parent, matcher):
    if False:
        return 10
    '\n    Return an iterable of the elements which are children of C{parent} for\n    which the predicate C{matcher} returns true.\n    '
    return findNodes(parent, lambda n, matcher=matcher: getattr(n, 'tagName', None) is not None and matcher(n))

def findElementsWithAttribute(parent, attribute, value=None):
    if False:
        i = 10
        return i + 15
    if value:
        return findElements(parent, lambda n, attribute=attribute, value=value: n.hasAttribute(attribute) and n.getAttribute(attribute) == value)
    else:
        return findElements(parent, lambda n, attribute=attribute: n.hasAttribute(attribute))

def findNodesNamed(parent, name):
    if False:
        print('Hello World!')
    return findNodes(parent, lambda n, name=name: n.nodeName == name)

def writeNodeData(node, oldio):
    if False:
        return 10
    for subnode in node.childNodes:
        if hasattr(subnode, 'data'):
            oldio.write('' + subnode.data)
        else:
            writeNodeData(subnode, oldio)

def getNodeText(node):
    if False:
        while True:
            i = 10
    oldio = StringIO()
    writeNodeData(node, oldio)
    return oldio.getvalue()

def getParents(node):
    if False:
        for i in range(10):
            print('nop')
    l = []
    while node:
        l.append(node)
        node = node.parentNode
    return l

def namedChildren(parent, nodeName):
    if False:
        return 10
    'namedChildren(parent, nodeName) -> children (not descendants) of parent\n    that have tagName == nodeName\n    '
    return [n for n in parent.childNodes if getattr(n, 'tagName', '') == nodeName]