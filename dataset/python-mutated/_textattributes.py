"""
This module provides some common functionality for the manipulation of
formatting states.

Defining the mechanism by which text containing character attributes is
constructed begins by subclassing L{CharacterAttributesMixin}.

Defining how a single formatting state is to be serialized begins by
subclassing L{_FormattingStateMixin}.

Serializing a formatting structure is done with L{flatten}.

@see: L{twisted.conch.insults.helper._FormattingState}
@see: L{twisted.conch.insults.text._CharacterAttributes}
@see: L{twisted.words.protocols.irc._FormattingState}
@see: L{twisted.words.protocols.irc._CharacterAttributes}
"""
from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin

class _Attribute(FancyEqMixin):
    """
    A text attribute.

    Indexing a text attribute with a C{str} or another text attribute adds that
    object as a child, indexing with a C{list} or C{tuple} adds the elements as
    children; in either case C{self} is returned.

    @type children: C{list}
    @ivar children: Child attributes.
    """
    compareAttributes: ClassVar[Sequence[str]] = ('children',)

    def __init__(self):
        if False:
            while True:
                i = 10
        self.children = []

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<{type(self).__name__} {vars(self)!r}>'

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        assert isinstance(item, (list, tuple, _Attribute, str))
        if isinstance(item, (list, tuple)):
            self.children.extend(item)
        else:
            self.children.append(item)
        return self

    def serialize(self, write, attrs=None, attributeRenderer='toVT102'):
        if False:
            i = 10
            return i + 15
        "\n        Serialize the text attribute and its children.\n\n        @param write: C{callable}, taking one C{str} argument, called to output\n            a single text attribute at a time.\n\n        @param attrs: A formatting state instance used to determine how to\n            serialize the attribute children.\n\n        @type attributeRenderer: C{str}\n        @param attributeRenderer: Name of the method on I{attrs} that should be\n            called to render the attributes during serialization. Defaults to\n            C{'toVT102'}.\n        "
        if attrs is None:
            attrs = DefaultFormattingState()
        for ch in self.children:
            if isinstance(ch, _Attribute):
                ch.serialize(write, attrs.copy(), attributeRenderer)
            else:
                renderMeth = getattr(attrs, attributeRenderer)
                write(renderMeth())
                write(ch)

class _NormalAttr(_Attribute):
    """
    A text attribute for normal text.
    """

    def serialize(self, write, attrs, attributeRenderer):
        if False:
            while True:
                i = 10
        attrs.__init__()
        _Attribute.serialize(self, write, attrs, attributeRenderer)

class _OtherAttr(_Attribute):
    """
    A text attribute for text with formatting attributes.

    The unary minus operator returns the inverse of this attribute, where that
    makes sense.

    @type attrname: C{str}
    @ivar attrname: Text attribute name.

    @ivar attrvalue: Text attribute value.
    """
    compareAttributes = ('attrname', 'attrvalue', 'children')

    def __init__(self, attrname, attrvalue):
        if False:
            return 10
        _Attribute.__init__(self)
        self.attrname = attrname
        self.attrvalue = attrvalue

    def __neg__(self):
        if False:
            return 10
        result = _OtherAttr(self.attrname, not self.attrvalue)
        result.children.extend(self.children)
        return result

    def serialize(self, write, attrs, attributeRenderer):
        if False:
            i = 10
            return i + 15
        attrs = attrs._withAttribute(self.attrname, self.attrvalue)
        _Attribute.serialize(self, write, attrs, attributeRenderer)

class _ColorAttr(_Attribute):
    """
    Generic color attribute.

    @param color: Color value.

    @param ground: Foreground or background attribute name.
    """
    compareAttributes = ('color', 'ground', 'children')

    def __init__(self, color, ground):
        if False:
            for i in range(10):
                print('nop')
        _Attribute.__init__(self)
        self.color = color
        self.ground = ground

    def serialize(self, write, attrs, attributeRenderer):
        if False:
            print('Hello World!')
        attrs = attrs._withAttribute(self.ground, self.color)
        _Attribute.serialize(self, write, attrs, attributeRenderer)

class _ForegroundColorAttr(_ColorAttr):
    """
    Foreground color attribute.
    """

    def __init__(self, color):
        if False:
            return 10
        _ColorAttr.__init__(self, color, 'foreground')

class _BackgroundColorAttr(_ColorAttr):
    """
    Background color attribute.
    """

    def __init__(self, color):
        if False:
            i = 10
            return i + 15
        _ColorAttr.__init__(self, color, 'background')

class _ColorAttribute:
    """
    A color text attribute.

    Attribute access results in a color value lookup, by name, in
    I{_ColorAttribute.attrs}.

    @type ground: L{_ColorAttr}
    @param ground: Foreground or background color attribute to look color names
        up from.

    @param attrs: Mapping of color names to color values.
    @type attrs: Dict like object.
    """

    def __init__(self, ground, attrs):
        if False:
            i = 10
            return i + 15
        self.ground = ground
        self.attrs = attrs

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        try:
            return self.ground(self.attrs[name])
        except KeyError:
            raise AttributeError(name)

class CharacterAttributesMixin:
    """
    Mixin for character attributes that implements a C{__getattr__} method
    returning a new C{_NormalAttr} instance when attempting to access
    a C{'normal'} attribute; otherwise a new C{_OtherAttr} instance is returned
    for names that appears in the C{'attrs'} attribute.
    """

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        if name == 'normal':
            return _NormalAttr()
        if name in self.attrs:
            return _OtherAttr(name, True)
        raise AttributeError(name)

class DefaultFormattingState(FancyEqMixin):
    """
    A character attribute that does nothing, thus applying no attributes to
    text.
    """
    compareAttributes: ClassVar[Sequence[str]] = ('_dummy',)
    _dummy = 0

    def copy(self):
        if False:
            print('Hello World!')
        '\n        Make a copy of this formatting state.\n\n        @return: A formatting state instance.\n        '
        return type(self)()

    def _withAttribute(self, name, value):
        if False:
            print('Hello World!')
        '\n        Add a character attribute to a copy of this formatting state.\n\n        @param name: Attribute name to be added to formatting state.\n\n        @param value: Attribute value.\n\n        @return: A formatting state instance with the new attribute.\n        '
        return self.copy()

    def toVT102(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Emit a VT102 control sequence that will set up all the attributes this\n        formatting state has set.\n\n        @return: A string containing VT102 control sequences that mimic this\n            formatting state.\n        '
        return ''

class _FormattingStateMixin(DefaultFormattingState):
    """
    Mixin for the formatting state/attributes of a single character.
    """

    def copy(self):
        if False:
            i = 10
            return i + 15
        c = DefaultFormattingState.copy(self)
        c.__dict__.update(vars(self))
        return c

    def _withAttribute(self, name, value):
        if False:
            while True:
                i = 10
        if getattr(self, name) != value:
            attr = self.copy()
            attr._subtracting = not value
            setattr(attr, name, value)
            return attr
        else:
            return self.copy()

def flatten(output, attrs, attributeRenderer='toVT102'):
    if False:
        while True:
            i = 10
    '\n    Serialize a sequence of characters with attribute information\n\n    The resulting string can be interpreted by compatible software so that the\n    contained characters are displayed and, for those attributes which are\n    supported by the software, the attributes expressed. The exact result of\n    the serialization depends on the behavior of the method specified by\n    I{attributeRenderer}.\n\n    For example, if your terminal is VT102 compatible, you might run\n    this for a colorful variation on the "hello world" theme::\n\n        from twisted.conch.insults.text import flatten, attributes as A\n        from twisted.conch.insults.helper import CharacterAttribute\n        print(flatten(\n            A.normal[A.bold[A.fg.red[\'He\'], A.fg.green[\'ll\'], A.fg.magenta[\'o\'], \' \',\n                            A.fg.yellow[\'Wo\'], A.fg.blue[\'rl\'], A.fg.cyan[\'d!\']]],\n            CharacterAttribute()))\n\n    @param output: Object returned by accessing attributes of the\n        module-level attributes object.\n\n    @param attrs: A formatting state instance used to determine how to\n        serialize C{output}.\n\n    @type attributeRenderer: C{str}\n    @param attributeRenderer: Name of the method on I{attrs} that should be\n        called to render the attributes during serialization. Defaults to\n        C{\'toVT102\'}.\n\n    @return: A string expressing the text and display attributes specified by\n        L{output}.\n    '
    flattened: List[str] = []
    output.serialize(flattened.append, attrs, attributeRenderer)
    return ''.join(flattened)
__all__ = ['flatten', 'DefaultFormattingState', 'CharacterAttributesMixin']