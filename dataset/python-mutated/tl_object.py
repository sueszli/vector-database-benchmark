from io import BytesIO
from json import dumps
from typing import cast, List, Any, Union, Dict
from ..all import objects

class TLObject:
    __slots__: List[str] = []
    QUALNAME = 'Base'

    @classmethod
    def read(cls, b: BytesIO, *args: Any) -> Any:
        if False:
            i = 10
            return i + 15
        return cast(TLObject, objects[int.from_bytes(b.read(4), 'little')]).read(b, *args)

    def write(self, *args: Any) -> bytes:
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    def default(obj: 'TLObject') -> Union[str, Dict[str, str]]:
        if False:
            while True:
                i = 10
        if isinstance(obj, bytes):
            return repr(obj)
        return {'_': obj.QUALNAME, **{attr: getattr(obj, attr) for attr in obj.__slots__ if getattr(obj, attr) is not None}}

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return dumps(self, indent=4, default=TLObject.default, ensure_ascii=False)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'QUALNAME'):
            return repr(self)
        return 'pyrogram.raw.{}({})'.format(self.QUALNAME, ', '.join((f'{attr}={repr(getattr(self, attr))}' for attr in self.__slots__ if getattr(self, attr) is not None)))

    def __eq__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        for attr in self.__slots__:
            try:
                if getattr(self, attr) != getattr(other, attr):
                    return False
            except AttributeError:
                return False
        return True

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.write())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            return 10
        pass