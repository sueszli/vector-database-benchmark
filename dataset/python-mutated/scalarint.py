from pipenv.vendor.ruamel.yaml.anchor import Anchor
from typing import Text, Any, Dict, List
__all__ = ['ScalarInt', 'BinaryInt', 'OctalInt', 'HexInt', 'HexCapsInt', 'DecimalInt']

class ScalarInt(int):

    def __new__(cls: Any, *args: Any, **kw: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        width = kw.pop('width', None)
        underscore = kw.pop('underscore', None)
        anchor = kw.pop('anchor', None)
        v = int.__new__(cls, *args, **kw)
        v._width = width
        v._underscore = underscore
        if anchor is not None:
            v.yaml_set_anchor(anchor, always_dump=True)
        return v

    def __iadd__(self, a: Any) -> Any:
        if False:
            while True:
                i = 10
        x = type(self)(self + a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __ifloordiv__(self, a: Any) -> Any:
        if False:
            while True:
                i = 10
        x = type(self)(self // a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __imul__(self, a: Any) -> Any:
        if False:
            return 10
        x = type(self)(self * a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __ipow__(self, a: Any) -> Any:
        if False:
            i = 10
            return i + 15
        x = type(self)(self ** a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __isub__(self, a: Any) -> Any:
        if False:
            while True:
                i = 10
        x = type(self)(self - a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    @property
    def anchor(self) -> Any:
        if False:
            return 10
        if not hasattr(self, Anchor.attrib):
            setattr(self, Anchor.attrib, Anchor())
        return getattr(self, Anchor.attrib)

    def yaml_anchor(self, any: bool=False) -> Any:
        if False:
            return 10
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

class BinaryInt(ScalarInt):

    def __new__(cls, value: Any, width: Any=None, underscore: Any=None, anchor: Any=None) -> Any:
        if False:
            i = 10
            return i + 15
        return ScalarInt.__new__(cls, value, width=width, underscore=underscore, anchor=anchor)

class OctalInt(ScalarInt):

    def __new__(cls, value: Any, width: Any=None, underscore: Any=None, anchor: Any=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return ScalarInt.__new__(cls, value, width=width, underscore=underscore, anchor=anchor)

class HexInt(ScalarInt):
    """uses lower case (a-f)"""

    def __new__(cls, value: Any, width: Any=None, underscore: Any=None, anchor: Any=None) -> Any:
        if False:
            return 10
        return ScalarInt.__new__(cls, value, width=width, underscore=underscore, anchor=anchor)

class HexCapsInt(ScalarInt):
    """uses upper case (A-F)"""

    def __new__(cls, value: Any, width: Any=None, underscore: Any=None, anchor: Any=None) -> Any:
        if False:
            print('Hello World!')
        return ScalarInt.__new__(cls, value, width=width, underscore=underscore, anchor=anchor)

class DecimalInt(ScalarInt):
    """needed if anchor"""

    def __new__(cls, value: Any, width: Any=None, underscore: Any=None, anchor: Any=None) -> Any:
        if False:
            return 10
        return ScalarInt.__new__(cls, value, width=width, underscore=underscore, anchor=anchor)