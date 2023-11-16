from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.utils.module_loading import qualname
if TYPE_CHECKING:
    import decimal
    from airflow.serialization.serde import U
serializers = ['decimal.Decimal']
deserializers = serializers
__version__ = 1

def serialize(o: object) -> tuple[U, str, int, bool]:
    if False:
        while True:
            i = 10
    from decimal import Decimal
    if not isinstance(o, Decimal):
        return ('', '', 0, False)
    name = qualname(o)
    (_, _, exponent) = o.as_tuple()
    if isinstance(exponent, int) and exponent >= 0:
        return (int(o), name, __version__, True)
    return (float(o), name, __version__, True)

def deserialize(classname: str, version: int, data: object) -> decimal.Decimal:
    if False:
        for i in range(10):
            print('nop')
    from decimal import Decimal
    if version > __version__:
        raise TypeError(f'serialized {version} of {classname} > {__version__}')
    if classname != qualname(Decimal):
        raise TypeError(f'{classname} != {qualname(Decimal)}')
    return Decimal(str(data))