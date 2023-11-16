"""
You cannot subclass bool, and this is necessary for round-tripping anchored
bool values (and also if you want to preserve the original way of writing)

bool.__bases__ is type 'int', so that is what is used as the basis for ScalarBoolean as well.

You can use these in an if statement, but not when testing equivalence
"""
from pipenv.vendor.ruamel.yaml.anchor import Anchor
from typing import Text, Any, Dict, List
__all__ = ['ScalarBoolean']

class ScalarBoolean(int):

    def __new__(cls: Any, *args: Any, **kw: Any) -> Any:
        if False:
            i = 10
            return i + 15
        anchor = kw.pop('anchor', None)
        b = int.__new__(cls, *args, **kw)
        if anchor is not None:
            b.yaml_set_anchor(anchor, always_dump=True)
        return b

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
            for i in range(10):
                print('nop')
        self.anchor.value = value
        self.anchor.always_dump = always_dump