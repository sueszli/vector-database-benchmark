"""
Context-free flattener/serializer for rendering Python objects, possibly
complex or arbitrarily nested, as strings.
"""
from __future__ import annotations
from inspect import iscoroutine
from io import BytesIO
from sys import exc_info
from traceback import extract_tb
from types import GeneratorType
from typing import Any, Callable, Coroutine, Generator, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.web._stan import CDATA, CharRef, Comment, Tag, slot, voidElements
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest
T = TypeVar('T')
FlattenableRecursive = Any
"\nFor documentation purposes, read C{FlattenableRecursive} as L{Flattenable}.\nHowever, since mypy doesn't support recursive type definitions (yet?),\nwe'll put Any in the actual definition.\n"
Flattenable = Union[bytes, str, slot, CDATA, Comment, Tag, Tuple[FlattenableRecursive, ...], List[FlattenableRecursive], Generator[FlattenableRecursive, None, None], CharRef, Deferred[FlattenableRecursive], Coroutine[Deferred[FlattenableRecursive], object, FlattenableRecursive], IRenderable]
'\nType alias containing all types that can be flattened by L{flatten()}.\n'
BUFFER_SIZE = 2 ** 16

def escapeForContent(data: Union[bytes, str]) -> bytes:
    if False:
        i = 10
        return i + 15
    "\n    Escape some character or UTF-8 byte data for inclusion in an HTML or XML\n    document, by replacing metacharacters (C{&<>}) with their entity\n    equivalents (C{&amp;&lt;&gt;}).\n\n    This is used as an input to L{_flattenElement}'s C{dataEscaper} parameter.\n\n    @param data: The string to escape.\n\n    @return: The quoted form of C{data}.  If C{data} is L{str}, return a utf-8\n        encoded string.\n    "
    if isinstance(data, str):
        data = data.encode('utf-8')
    data = data.replace(b'&', b'&amp;').replace(b'<', b'&lt;').replace(b'>', b'&gt;')
    return data

def attributeEscapingDoneOutside(data: Union[bytes, str]) -> bytes:
    if False:
        print('Hello World!')
    '\n    Escape some character or UTF-8 byte data for inclusion in the top level of\n    an attribute.  L{attributeEscapingDoneOutside} actually passes the data\n    through unchanged, because L{writeWithAttributeEscaping} handles the\n    quoting of the text within attributes outside the generator returned by\n    L{_flattenElement}; this is used as the C{dataEscaper} argument to that\n    L{_flattenElement} call so that that generator does not redundantly escape\n    its text output.\n\n    @param data: The string to escape.\n\n    @return: The string, unchanged, except for encoding.\n    '
    if isinstance(data, str):
        return data.encode('utf-8')
    return data

def writeWithAttributeEscaping(write: Callable[[bytes], object]) -> Callable[[bytes], None]:
    if False:
        while True:
            i = 10
    '\n    Decorate a C{write} callable so that all output written is properly quoted\n    for inclusion within an XML attribute value.\n\n    If a L{Tag <twisted.web.template.Tag>} C{x} is flattened within the context\n    of the contents of another L{Tag <twisted.web.template.Tag>} C{y}, the\n    metacharacters (C{<>&"}) delimiting C{x} should be passed through\n    unchanged, but the textual content of C{x} should still be quoted, as\n    usual.  For example: C{<y><x>&amp;</x></y>}.  That is the default behavior\n    of L{_flattenElement} when L{escapeForContent} is passed as the\n    C{dataEscaper}.\n\n    However, when a L{Tag <twisted.web.template.Tag>} C{x} is flattened within\n    the context of an I{attribute} of another L{Tag <twisted.web.template.Tag>}\n    C{y}, then the metacharacters delimiting C{x} should be quoted so that it\n    can be parsed from the attribute\'s value.  In the DOM itself, this is not a\n    valid thing to do, but given that renderers and slots may be freely moved\n    around in a L{twisted.web.template} template, it is a condition which may\n    arise in a document and must be handled in a way which produces valid\n    output.  So, for example, you should be able to get C{<y attr="&lt;x /&gt;"\n    />}.  This should also be true for other XML/HTML meta-constructs such as\n    comments and CDATA, so if you were to serialize a L{comment\n    <twisted.web.template.Comment>} in an attribute you should get C{<y\n    attr="&lt;-- comment --&gt;" />}.  Therefore in order to capture these\n    meta-characters, flattening is done with C{write} callable that is wrapped\n    with L{writeWithAttributeEscaping}.\n\n    The final case, and hopefully the much more common one as compared to\n    serializing L{Tag <twisted.web.template.Tag>} and arbitrary L{IRenderable}\n    objects within an attribute, is to serialize a simple string, and those\n    should be passed through for L{writeWithAttributeEscaping} to quote\n    without applying a second, redundant level of quoting.\n\n    @param write: A callable which will be invoked with the escaped L{bytes}.\n\n    @return: A callable that writes data with escaping.\n    '

    def _write(data: bytes) -> None:
        if False:
            while True:
                i = 10
        write(escapeForContent(data).replace(b'"', b'&quot;'))
    return _write

def escapedCDATA(data: Union[bytes, str]) -> bytes:
    if False:
        i = 10
        return i + 15
    '\n    Escape CDATA for inclusion in a document.\n\n    @param data: The string to escape.\n\n    @return: The quoted form of C{data}. If C{data} is unicode, return a utf-8\n        encoded string.\n    '
    if isinstance(data, str):
        data = data.encode('utf-8')
    return data.replace(b']]>', b']]]]><![CDATA[>')

def escapedComment(data: Union[bytes, str]) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    '\n    Within comments the sequence C{-->} can be mistaken as the end of the comment.\n    To ensure consistent parsing and valid output the sequence is replaced with C{--&gt;}.\n    Furthermore, whitespace is added when a comment ends in a dash. This is done to break\n    the connection of the ending C{-} with the closing C{-->}.\n\n    @param data: The string to escape.\n\n    @return: The quoted form of C{data}. If C{data} is unicode, return a utf-8\n        encoded string.\n    '
    if isinstance(data, str):
        data = data.encode('utf-8')
    data = data.replace(b'-->', b'--&gt;')
    if data and data[-1:] == b'-':
        data += b' '
    return data

def _getSlotValue(name: str, slotData: Sequence[Optional[Mapping[str, Flattenable]]], default: Optional[Flattenable]=None) -> Flattenable:
    if False:
        return 10
    '\n    Find the value of the named slot in the given stack of slot data.\n    '
    for slotFrame in reversed(slotData):
        if slotFrame is not None and name in slotFrame:
            return slotFrame[name]
    else:
        if default is not None:
            return default
        raise UnfilledSlot(name)

def _fork(d: Deferred[T]) -> Deferred[T]:
    if False:
        print('Hello World!')
    "\n    Create a new L{Deferred} based on C{d} that will fire and fail with C{d}'s\n    result or error, but will not modify C{d}'s callback type.\n    "
    d2: Deferred[T] = Deferred(lambda _: d.cancel())

    def callback(result: T) -> T:
        if False:
            print('Hello World!')
        d2.callback(result)
        return result

    def errback(failure: Failure) -> Failure:
        if False:
            return 10
        d2.errback(failure)
        return failure
    d.addCallbacks(callback, errback)
    return d2

def _flattenElement(request: Optional[IRequest], root: Flattenable, write: Callable[[bytes], object], slotData: List[Optional[Mapping[str, Flattenable]]], renderFactory: Optional[IRenderable], dataEscaper: Callable[[Union[bytes, str]], bytes]) -> Generator[Union[Generator[Any, Any, Any], Deferred[Flattenable]], None, None]:
    if False:
        return 10
    '\n    Make C{root} slightly more flat by yielding all its immediate contents as\n    strings, deferreds or generators that are recursive calls to itself.\n\n    @param request: A request object which will be passed to\n        L{IRenderable.render}.\n\n    @param root: An object to be made flatter.  This may be of type C{unicode},\n        L{str}, L{slot}, L{Tag <twisted.web.template.Tag>}, L{tuple}, L{list},\n        L{types.GeneratorType}, L{Deferred}, or an object that implements\n        L{IRenderable}.\n\n    @param write: A callable which will be invoked with each L{bytes} produced\n        by flattening C{root}.\n\n    @param slotData: A L{list} of L{dict} mapping L{str} slot names to data\n        with which those slots will be replaced.\n\n    @param renderFactory: If not L{None}, an object that provides\n        L{IRenderable}.\n\n    @param dataEscaper: A 1-argument callable which takes L{bytes} or\n        L{unicode} and returns L{bytes}, quoted as appropriate for the\n        rendering context.  This is really only one of two values:\n        L{attributeEscapingDoneOutside} or L{escapeForContent}, depending on\n        whether the rendering context is within an attribute or not.  See the\n        explanation in L{writeWithAttributeEscaping}.\n\n    @return: An iterator that eventually writes L{bytes} to C{write}.\n        It can yield other iterators or L{Deferred}s; if it yields another\n        iterator, the caller will iterate it; if it yields a L{Deferred},\n        the result of that L{Deferred} will be another generator, in which\n        case it is iterated.  See L{_flattenTree} for the trampoline that\n        consumes said values.\n    '

    def keepGoing(newRoot: Flattenable, dataEscaper: Callable[[Union[bytes, str]], bytes]=dataEscaper, renderFactory: Optional[IRenderable]=renderFactory, write: Callable[[bytes], object]=write) -> Generator[Union[Flattenable, Deferred[Flattenable]], None, None]:
        if False:
            for i in range(10):
                print('nop')
        return _flattenElement(request, newRoot, write, slotData, renderFactory, dataEscaper)

    def keepGoingAsync(result: Deferred[Flattenable]) -> Deferred[Flattenable]:
        if False:
            return 10
        return result.addCallback(keepGoing)
    if isinstance(root, (bytes, str)):
        write(dataEscaper(root))
    elif isinstance(root, slot):
        slotValue = _getSlotValue(root.name, slotData, root.default)
        yield keepGoing(slotValue)
    elif isinstance(root, CDATA):
        write(b'<![CDATA[')
        write(escapedCDATA(root.data))
        write(b']]>')
    elif isinstance(root, Comment):
        write(b'<!--')
        write(escapedComment(root.data))
        write(b'-->')
    elif isinstance(root, Tag):
        slotData.append(root.slotData)
        rendererName = root.render
        if rendererName is not None:
            if renderFactory is None:
                raise ValueError(f'Tag wants to be rendered by method "{rendererName}" but is not contained in any IRenderable')
            rootClone = root.clone(False)
            rootClone.render = None
            renderMethod = renderFactory.lookupRenderMethod(rendererName)
            result = renderMethod(request, rootClone)
            yield keepGoing(result)
            slotData.pop()
            return
        if not root.tagName:
            yield keepGoing(root.children)
            return
        write(b'<')
        if isinstance(root.tagName, str):
            tagName = root.tagName.encode('ascii')
        else:
            tagName = root.tagName
        write(tagName)
        for (k, v) in root.attributes.items():
            if isinstance(k, str):
                k = k.encode('ascii')
            write(b' ' + k + b'="')
            yield keepGoing(v, attributeEscapingDoneOutside, write=writeWithAttributeEscaping(write))
            write(b'"')
        if root.children or nativeString(tagName) not in voidElements:
            write(b'>')
            yield keepGoing(root.children, escapeForContent)
            write(b'</' + tagName + b'>')
        else:
            write(b' />')
    elif isinstance(root, (tuple, list, GeneratorType)):
        for element in root:
            yield keepGoing(element)
    elif isinstance(root, CharRef):
        escaped = '&#%d;' % (root.ordinal,)
        write(escaped.encode('ascii'))
    elif isinstance(root, Deferred):
        yield keepGoingAsync(_fork(root))
    elif iscoroutine(root):
        yield keepGoingAsync(Deferred.fromCoroutine(cast(Coroutine[Deferred[Flattenable], object, Flattenable], root)))
    elif IRenderable.providedBy(root):
        result = root.render(request)
        yield keepGoing(result, renderFactory=root)
    else:
        raise UnsupportedType(root)

async def _flattenTree(request: Optional[IRequest], root: Flattenable, write: Callable[[bytes], object]) -> None:
    """
    Make C{root} into an iterable of L{bytes} and L{Deferred} by doing a depth
    first traversal of the tree.

    @param request: A request object which will be passed to
        L{IRenderable.render}.

    @param root: An object to be made flatter.  This may be of type C{unicode},
        L{bytes}, L{slot}, L{Tag <twisted.web.template.Tag>}, L{tuple},
        L{list}, L{types.GeneratorType}, L{Deferred}, or something providing
        L{IRenderable}.

    @param write: A callable which will be invoked with each L{bytes} produced
        by flattening C{root}.

    @return: A C{Deferred}-returning coroutine that resolves to C{None}.
    """
    buf = []
    bufSize = 0

    def bufferedWrite(bs: bytes) -> None:
        if False:
            while True:
                i = 10
        nonlocal bufSize
        buf.append(bs)
        bufSize += len(bs)
        if bufSize >= BUFFER_SIZE:
            flushBuffer()

    def flushBuffer() -> None:
        if False:
            return 10
        nonlocal bufSize
        if bufSize > 0:
            write(b''.join(buf))
            del buf[:]
            bufSize = 0
    stack: List[Generator[Any, Any, Any]] = [_flattenElement(request, root, bufferedWrite, [], None, escapeForContent)]
    while stack:
        try:
            frame = stack[-1].gi_frame
            element = next(stack[-1])
            if isinstance(element, Deferred):
                flushBuffer()
                element = await element
        except StopIteration:
            stack.pop()
        except Exception as e:
            stack.pop()
            roots = []
            for generator in stack:
                roots.append(generator.gi_frame.f_locals['root'])
            roots.append(frame.f_locals['root'])
            raise FlattenerError(e, roots, extract_tb(exc_info()[2]))
        else:
            stack.append(element)
    flushBuffer()

def flatten(request: Optional[IRequest], root: Flattenable, write: Callable[[bytes], object]) -> Deferred[None]:
    if False:
        print('Hello World!')
    '\n    Incrementally write out a string representation of C{root} using C{write}.\n\n    In order to create a string representation, C{root} will be decomposed into\n    simpler objects which will themselves be decomposed and so on until strings\n    or objects which can easily be converted to strings are encountered.\n\n    @param request: A request object which will be passed to the C{render}\n        method of any L{IRenderable} provider which is encountered.\n\n    @param root: An object to be made flatter.  This may be of type L{str},\n        L{bytes}, L{slot}, L{Tag <twisted.web.template.Tag>}, L{tuple},\n        L{list}, L{types.GeneratorType}, L{Deferred}, or something that\n        provides L{IRenderable}.\n\n    @param write: A callable which will be invoked with each L{bytes} produced\n        by flattening C{root}.\n\n    @return: A L{Deferred} which will be called back with C{None} when C{root}\n        has been completely flattened into C{write} or which will be errbacked\n        if an unexpected exception occurs.\n    '
    return ensureDeferred(_flattenTree(request, root, write))

def flattenString(request: Optional[IRequest], root: Flattenable) -> Deferred[bytes]:
    if False:
        print('Hello World!')
    '\n    Collate a string representation of C{root} into a single string.\n\n    This is basically gluing L{flatten} to an L{io.BytesIO} and returning\n    the results. See L{flatten} for the exact meanings of C{request} and\n    C{root}.\n\n    @return: A L{Deferred} which will be called back with a single UTF-8 encoded\n        string as its result when C{root} has been completely flattened or which\n        will be errbacked if an unexpected exception occurs.\n    '
    io = BytesIO()
    d = flatten(request, root, io.write)
    d.addCallback(lambda _: io.getvalue())
    return cast(Deferred[bytes], d)