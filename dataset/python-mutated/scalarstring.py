from pipenv.vendor.ruamel.yaml.anchor import Anchor
from typing import Text, Any, Dict, List
from pipenv.vendor.ruamel.yaml.compat import SupportsIndex
__all__ = ['ScalarString', 'LiteralScalarString', 'FoldedScalarString', 'SingleQuotedScalarString', 'DoubleQuotedScalarString', 'PlainScalarString', 'PreservedScalarString']

class ScalarString(str):
    __slots__ = Anchor.attrib

    def __new__(cls, *args: Any, **kw: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        anchor = kw.pop('anchor', None)
        ret_val = str.__new__(cls, *args, **kw)
        if anchor is not None:
            ret_val.yaml_set_anchor(anchor, always_dump=True)
        return ret_val

    def replace(self, old: Any, new: Any, maxreplace: SupportsIndex=-1) -> Any:
        if False:
            return 10
        return type(self)(str.replace(self, old, new, maxreplace))

    @property
    def anchor(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, Anchor.attrib):
            setattr(self, Anchor.attrib, Anchor())
        return getattr(self, Anchor.attrib)

    def yaml_anchor(self, any: bool=False) -> Any:
        if False:
            i = 10
            return i + 15
        if not hasattr(self, Anchor.attrib):
            return None
        if any or self.anchor.always_dump:
            return self.anchor
        return None

    def yaml_set_anchor(self, value: Any, always_dump: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.anchor.value = value
        self.anchor.always_dump = always_dump

class LiteralScalarString(ScalarString):
    __slots__ = 'comment'
    style = '|'

    def __new__(cls, value: Text, anchor: Any=None) -> Any:
        if False:
            i = 10
            return i + 15
        return ScalarString.__new__(cls, value, anchor=anchor)
PreservedScalarString = LiteralScalarString

class FoldedScalarString(ScalarString):
    __slots__ = ('fold_pos', 'comment')
    style = '>'

    def __new__(cls, value: Text, anchor: Any=None) -> Any:
        if False:
            while True:
                i = 10
        return ScalarString.__new__(cls, value, anchor=anchor)

class SingleQuotedScalarString(ScalarString):
    __slots__ = ()
    style = "'"

    def __new__(cls, value: Text, anchor: Any=None) -> Any:
        if False:
            print('Hello World!')
        return ScalarString.__new__(cls, value, anchor=anchor)

class DoubleQuotedScalarString(ScalarString):
    __slots__ = ()
    style = '"'

    def __new__(cls, value: Text, anchor: Any=None) -> Any:
        if False:
            print('Hello World!')
        return ScalarString.__new__(cls, value, anchor=anchor)

class PlainScalarString(ScalarString):
    __slots__ = ()
    style = ''

    def __new__(cls, value: Text, anchor: Any=None) -> Any:
        if False:
            print('Hello World!')
        return ScalarString.__new__(cls, value, anchor=anchor)

def preserve_literal(s: Text) -> Text:
    if False:
        return 10
    return LiteralScalarString(s.replace('\r\n', '\n').replace('\r', '\n'))

def walk_tree(base: Any, map: Any=None) -> None:
    if False:
        while True:
            i = 10
    "\n    the routine here walks over a simple yaml tree (recursing in\n    dict values and list items) and converts strings that\n    have multiple lines to literal scalars\n\n    You can also provide an explicit (ordered) mapping for multiple transforms\n    (first of which is executed):\n        map = ruamel.compat.ordereddict\n        map['\n'] = preserve_literal\n        map[':'] = SingleQuotedScalarString\n        walk_tree(data, map=map)\n    "
    from collections.abc import MutableMapping, MutableSequence
    if map is None:
        map = {'\n': preserve_literal}
    if isinstance(base, MutableMapping):
        for k in base:
            v: Text = base[k]
            if isinstance(v, str):
                for ch in map:
                    if ch in v:
                        base[k] = map[ch](v)
                        break
            else:
                walk_tree(v, map=map)
    elif isinstance(base, MutableSequence):
        for (idx, elem) in enumerate(base):
            if isinstance(elem, str):
                for ch in map:
                    if ch in elem:
                        base[idx] = map[ch](elem)
                        break
            else:
                walk_tree(elem, map=map)