""" Vectorization related data types used by dataspecs.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, Union
from ...util.dataclasses import NotRequired, Unspecified, dataclass
from ..serialization import AnyRep, Deserializer, Serializable, Serializer
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from ...models.expressions import Expression
    from ...models.transforms import Transform
__all__ = ('Expr', 'Field', 'Value', 'expr', 'field', 'value')

@dataclass
class Value(Serializable):
    value: Any
    transform: NotRequired[Transform] = Unspecified
    units: NotRequired[str] = Unspecified

    def to_serializable(self, serializer: Serializer) -> AnyRep:
        if False:
            i = 10
            return i + 15
        return serializer.encode_struct(type='value', value=self.value, transform=self.transform, units=self.units)

    @classmethod
    def from_serializable(cls, rep: dict[str, AnyRep], deserializer: Deserializer) -> Value:
        if False:
            return 10
        if 'value' not in rep:
            deserializer.error("expected 'value' field")
        value = deserializer.decode(rep['value'])
        transform = deserializer.decode(rep['transform']) if 'transform' in rep else Unspecified
        units = deserializer.decode(rep['units']) if 'units' in rep else Unspecified
        return Value(value, transform, units)

    def __getitem__(self, key: str) -> Any:
        if False:
            return 10
        if key == 'value':
            return self.value
        elif key == 'transform' and self.transform is not Unspecified:
            return self.transform
        elif key == 'units' and self.units is not Unspecified:
            return self.units
        else:
            raise KeyError(f"key '{key}' not found")

@dataclass
class Field(Serializable):
    field: str
    transform: NotRequired[Transform] = Unspecified
    units: NotRequired[str] = Unspecified

    def to_serializable(self, serializer: Serializer) -> AnyRep:
        if False:
            i = 10
            return i + 15
        return serializer.encode_struct(type='field', field=self.field, transform=self.transform, units=self.units)

    @classmethod
    def from_serializable(cls, rep: dict[str, AnyRep], deserializer: Deserializer) -> Field:
        if False:
            for i in range(10):
                print('nop')
        if 'field' not in rep:
            deserializer.error("expected 'field' field")
        field = deserializer.decode(rep['field'])
        transform = deserializer.decode(rep['transform']) if 'transform' in rep else Unspecified
        units = deserializer.decode(rep['units']) if 'units' in rep else Unspecified
        return Field(field, transform, units)

    def __getitem__(self, key: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if key == 'field':
            return self.field
        elif key == 'transform' and self.transform is not Unspecified:
            return self.transform
        elif key == 'units' and self.units is not Unspecified:
            return self.units
        else:
            raise KeyError(f"key '{key}' not found")

@dataclass
class Expr(Serializable):
    expr: Expression
    transform: NotRequired[Transform] = Unspecified
    units: NotRequired[str] = Unspecified

    def to_serializable(self, serializer: Serializer) -> AnyRep:
        if False:
            for i in range(10):
                print('nop')
        return serializer.encode_struct(type='expr', expr=self.expr, transform=self.transform, units=self.units)

    @classmethod
    def from_serializable(cls, rep: dict[str, AnyRep], deserializer: Deserializer) -> Expr:
        if False:
            print('Hello World!')
        if 'expr' not in rep:
            deserializer.error("expected 'expr' field")
        expr = deserializer.decode(rep['expr'])
        transform = deserializer.decode(rep['transform']) if 'transform' in rep else Unspecified
        units = deserializer.decode(rep['units']) if 'units' in rep else Unspecified
        return Expr(expr, transform, units)

    def __getitem__(self, key: str) -> Any:
        if False:
            i = 10
            return i + 15
        if key == 'expr':
            return self.expr
        elif key == 'transform' and self.transform is not Unspecified:
            return self.transform
        elif key == 'units' and self.units is not Unspecified:
            return self.units
        else:
            raise KeyError(f"key '{key}' not found")
Vectorized: TypeAlias = Union[Value, Field, Expr]
value = Value
field = Field
expr = Expr
Deserializer.register('value', Value.from_serializable)
Deserializer.register('field', Field.from_serializable)
Deserializer.register('expr', Expr.from_serializable)