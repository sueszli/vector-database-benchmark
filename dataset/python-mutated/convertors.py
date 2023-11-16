import math
import typing
import uuid
T = typing.TypeVar('T')

class Convertor(typing.Generic[T]):
    regex: typing.ClassVar[str] = ''

    def convert(self, value: str) -> T:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def to_string(self, value: T) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class StringConvertor(Convertor[str]):
    regex = '[^/]+'

    def convert(self, value: str) -> str:
        if False:
            while True:
                i = 10
        return value

    def to_string(self, value: str) -> str:
        if False:
            print('Hello World!')
        value = str(value)
        assert '/' not in value, 'May not contain path separators'
        assert value, 'Must not be empty'
        return value

class PathConvertor(Convertor[str]):
    regex = '.*'

    def convert(self, value: str) -> str:
        if False:
            return 10
        return str(value)

    def to_string(self, value: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return str(value)

class IntegerConvertor(Convertor[int]):
    regex = '[0-9]+'

    def convert(self, value: str) -> int:
        if False:
            return 10
        return int(value)

    def to_string(self, value: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        value = int(value)
        assert value >= 0, 'Negative integers are not supported'
        return str(value)

class FloatConvertor(Convertor[float]):
    regex = '[0-9]+(\\.[0-9]+)?'

    def convert(self, value: str) -> float:
        if False:
            return 10
        return float(value)

    def to_string(self, value: float) -> str:
        if False:
            for i in range(10):
                print('nop')
        value = float(value)
        assert value >= 0.0, 'Negative floats are not supported'
        assert not math.isnan(value), 'NaN values are not supported'
        assert not math.isinf(value), 'Infinite values are not supported'
        return ('%0.20f' % value).rstrip('0').rstrip('.')

class UUIDConvertor(Convertor[uuid.UUID]):
    regex = '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'

    def convert(self, value: str) -> uuid.UUID:
        if False:
            for i in range(10):
                print('nop')
        return uuid.UUID(value)

    def to_string(self, value: uuid.UUID) -> str:
        if False:
            return 10
        return str(value)
CONVERTOR_TYPES: typing.Dict[str, Convertor[typing.Any]] = {'str': StringConvertor(), 'path': PathConvertor(), 'int': IntegerConvertor(), 'float': FloatConvertor(), 'uuid': UUIDConvertor()}

def register_url_convertor(key: str, convertor: Convertor[typing.Any]) -> None:
    if False:
        i = 10
        return i + 15
    CONVERTOR_TYPES[key] = convertor