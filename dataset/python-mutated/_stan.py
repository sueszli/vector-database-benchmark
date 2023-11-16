"""
An s-expression-like syntax for expressing xml in pure python.

Stan tags allow you to build XML documents using Python.

Stan is a DOM, or Document Object Model, implemented using basic Python types
and functions called "flatteners". A flattener is a function that knows how to
turn an object of a specific type into something that is closer to an HTML
string. Stan differs from the W3C DOM by not being as cumbersome and heavy
weight. Since the object model is built using simple python types such as lists,
strings, and dictionaries, the API is simpler and constructing a DOM less
cumbersome.

@var voidElements: the names of HTML 'U{void
    elements<http://www.whatwg.org/specs/web-apps/current-work/multipage/syntax.html#void-elements>}';
    those which can't have contents and can therefore be self-closing in the
    output.
"""
from inspect import iscoroutine, isgenerator
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from warnings import warn
import attr
if TYPE_CHECKING:
    from twisted.web.template import Flattenable

@attr.s(hash=False, eq=False, auto_attribs=True)
class slot:
    """
    Marker for markup insertion in a template.
    """
    name: str
    '\n    The name of this slot.\n\n    The key which must be used in L{Tag.fillSlots} to fill it.\n    '
    children: List['Tag'] = attr.ib(init=False, factory=list)
    "\n    The L{Tag} objects included in this L{slot}'s template.\n    "
    default: Optional['Flattenable'] = None
    '\n    The default contents of this slot, if it is left unfilled.\n\n    If this is L{None}, an L{UnfilledSlot} will be raised, rather than\n    L{None} actually being used.\n    '
    filename: Optional[str] = None
    '\n    The name of the XML file from which this tag was parsed.\n\n    If it was not parsed from an XML file, L{None}.\n    '
    lineNumber: Optional[int] = None
    '\n    The line number on which this tag was encountered in the XML file\n    from which it was parsed.\n\n    If it was not parsed from an XML file, L{None}.\n    '
    columnNumber: Optional[int] = None
    '\n    The column number at which this tag was encountered in the XML file\n    from which it was parsed.\n\n    If it was not parsed from an XML file, L{None}.\n    '

@attr.s(hash=False, eq=False, repr=False, auto_attribs=True)
class Tag:
    """
    A L{Tag} represents an XML tags with a tag name, attributes, and children.
    A L{Tag} can be constructed using the special L{twisted.web.template.tags}
    object, or it may be constructed directly with a tag name. L{Tag}s have a
    special method, C{__call__}, which makes representing trees of XML natural
    using pure python syntax.
    """
    tagName: Union[bytes, str]
    '\n    The name of the represented element.\n\n    For a tag like C{<div></div>}, this would be C{"div"}.\n    '
    attributes: Dict[Union[bytes, str], 'Flattenable'] = attr.ib(factory=dict)
    'The attributes of the element.'
    children: List['Flattenable'] = attr.ib(factory=list)
    'The contents of this C{Tag}.'
    render: Optional[str] = None
    '\n    The name of the render method to use for this L{Tag}.\n\n    This name will be looked up at render time by the\n    L{twisted.web.template.Element} doing the rendering,\n    via L{twisted.web.template.Element.lookupRenderMethod},\n    to determine which method to call.\n    '
    filename: Optional[str] = None
    '\n    The name of the XML file from which this tag was parsed.\n\n    If it was not parsed from an XML file, L{None}.\n    '
    lineNumber: Optional[int] = None
    '\n    The line number on which this tag was encountered in the XML file\n    from which it was parsed.\n\n    If it was not parsed from an XML file, L{None}.\n    '
    columnNumber: Optional[int] = None
    '\n    The column number at which this tag was encountered in the XML file\n    from which it was parsed.\n\n    If it was not parsed from an XML file, L{None}.\n    '
    slotData: Optional[Dict[str, 'Flattenable']] = attr.ib(init=False, default=None)
    '\n    The data which can fill slots.\n\n    If present, a dictionary mapping slot names to renderable values.\n    The values in this dict might be anything that can be present as\n    the child of a L{Tag}: strings, lists, L{Tag}s, generators, etc.\n    '

    def fillSlots(self, **slots: 'Flattenable') -> 'Tag':
        if False:
            return 10
        '\n        Remember the slots provided at this position in the DOM.\n\n        During the rendering of children of this node, slots with names in\n        C{slots} will be rendered as their corresponding values.\n\n        @return: C{self}. This enables the idiom C{return tag.fillSlots(...)} in\n            renderers.\n        '
        if self.slotData is None:
            self.slotData = {}
        self.slotData.update(slots)
        return self

    def __call__(self, *children: 'Flattenable', **kw: 'Flattenable') -> 'Tag':
        if False:
            return 10
        '\n        Add children and change attributes on this tag.\n\n        This is implemented using __call__ because it then allows the natural\n        syntax::\n\n          table(tr1, tr2, width="100%", height="50%", border="1")\n\n        Children may be other tag instances, strings, functions, or any other\n        object which has a registered flatten.\n\n        Attributes may be \'transparent\' tag instances (so that\n        C{a(href=transparent(data="foo", render=myhrefrenderer))} works),\n        strings, functions, or any other object which has a registered\n        flattener.\n\n        If the attribute is a python keyword, such as \'class\', you can add an\n        underscore to the name, like \'class_\'.\n\n        There is one special keyword argument, \'render\', which will be used as\n        the name of the renderer and saved as the \'render\' attribute of this\n        instance, rather than the DOM \'render\' attribute in the attributes\n        dictionary.\n        '
        self.children.extend(children)
        for (k, v) in kw.items():
            if k[-1] == '_':
                k = k[:-1]
            if k == 'render':
                if not isinstance(v, str):
                    raise TypeError(f'Value for "render" attribute must be str, got {v!r}')
                self.render = v
            else:
                self.attributes[k] = v
        return self

    def _clone(self, obj: 'Flattenable', deep: bool) -> 'Flattenable':
        if False:
            for i in range(10):
                print('nop')
        '\n        Clone a C{Flattenable} object; used by L{Tag.clone}.\n\n        Note that both lists and tuples are cloned into lists.\n\n        @param obj: an object with a clone method, a list or tuple, or something\n            which should be immutable.\n\n        @param deep: whether to continue cloning child objects; i.e. the\n            contents of lists, the sub-tags within a tag.\n\n        @return: a clone of C{obj}.\n        '
        if hasattr(obj, 'clone'):
            return obj.clone(deep)
        elif isinstance(obj, (list, tuple)):
            return [self._clone(x, deep) for x in obj]
        elif isgenerator(obj):
            warn('Cloning a Tag which contains a generator is unsafe, since the generator can be consumed only once; this is deprecated since Twisted 21.7.0 and will raise an exception in the future', DeprecationWarning)
            return obj
        elif iscoroutine(obj):
            warn('Cloning a Tag which contains a coroutine is unsafe, since the coroutine can run only once; this is deprecated since Twisted 21.7.0 and will raise an exception in the future', DeprecationWarning)
            return obj
        else:
            return obj

    def clone(self, deep: bool=True) -> 'Tag':
        if False:
            while True:
                i = 10
        "\n        Return a clone of this tag. If deep is True, clone all of this tag's\n        children. Otherwise, just shallow copy the children list without copying\n        the children themselves.\n        "
        if deep:
            newchildren = [self._clone(x, True) for x in self.children]
        else:
            newchildren = self.children[:]
        newattrs = self.attributes.copy()
        for key in newattrs.keys():
            newattrs[key] = self._clone(newattrs[key], True)
        newslotdata = None
        if self.slotData:
            newslotdata = self.slotData.copy()
            for key in newslotdata:
                newslotdata[key] = self._clone(newslotdata[key], True)
        newtag = Tag(self.tagName, attributes=newattrs, children=newchildren, render=self.render, filename=self.filename, lineNumber=self.lineNumber, columnNumber=self.columnNumber)
        newtag.slotData = newslotdata
        return newtag

    def clear(self) -> 'Tag':
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear any existing children from this tag.\n        '
        self.children = []
        return self

    def __repr__(self) -> str:
        if False:
            return 10
        rstr = ''
        if self.attributes:
            rstr += ', attributes=%r' % self.attributes
        if self.children:
            rstr += ', children=%r' % self.children
        return f'Tag({self.tagName!r}{rstr})'
voidElements = ('img', 'br', 'hr', 'base', 'meta', 'link', 'param', 'area', 'input', 'col', 'basefont', 'isindex', 'frame', 'command', 'embed', 'keygen', 'source', 'track', 'wbs')

@attr.s(hash=False, eq=False, repr=False, auto_attribs=True)
class CDATA:
    """
    A C{<![CDATA[]]>} block from a template.  Given a separate representation in
    the DOM so that they may be round-tripped through rendering without losing
    information.
    """
    data: str
    'The data between "C{<![CDATA[}" and "C{]]>}".'

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'CDATA({self.data!r})'

@attr.s(hash=False, eq=False, repr=False, auto_attribs=True)
class Comment:
    """
    A C{<!-- -->} comment from a template.  Given a separate representation in
    the DOM so that they may be round-tripped through rendering without losing
    information.
    """
    data: str
    'The data between "C{<!--}" and "C{-->}".'

    def __repr__(self) -> str:
        if False:
            return 10
        return f'Comment({self.data!r})'

@attr.s(hash=False, eq=False, repr=False, auto_attribs=True)
class CharRef:
    """
    A numeric character reference.  Given a separate representation in the DOM
    so that non-ASCII characters may be output as pure ASCII.

    @since: 12.0
    """
    ordinal: int
    'The ordinal value of the unicode character to which this object refers.'

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'CharRef(%d)' % (self.ordinal,)