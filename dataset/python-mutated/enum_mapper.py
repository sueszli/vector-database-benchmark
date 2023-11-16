"""Helper class to convert between Home Assistant and ESPHome enum values."""
from typing import Generic, TypeVar, overload
from aioesphomeapi import APIIntEnum
_EnumT = TypeVar('_EnumT', bound=APIIntEnum)
_ValT = TypeVar('_ValT')

class EsphomeEnumMapper(Generic[_EnumT, _ValT]):
    """Helper class to convert between hass and esphome enum values."""

    def __init__(self, mapping: dict[_EnumT, _ValT]) -> None:
        if False:
            return 10
        'Construct a EsphomeEnumMapper.'
        augmented_mapping: dict[_EnumT | None, _ValT | None] = mapping
        augmented_mapping[None] = None
        self._mapping = augmented_mapping
        self._inverse: dict[_ValT, _EnumT] = {v: k for (k, v) in mapping.items()}

    @overload
    def from_esphome(self, value: _EnumT) -> _ValT:
        if False:
            return 10
        ...

    @overload
    def from_esphome(self, value: _EnumT | None) -> _ValT | None:
        if False:
            for i in range(10):
                print('nop')
        ...

    def from_esphome(self, value: _EnumT | None) -> _ValT | None:
        if False:
            while True:
                i = 10
        'Convert from an esphome int representation to a hass string.'
        return self._mapping[value]

    def from_hass(self, value: _ValT) -> _EnumT:
        if False:
            print('Hello World!')
        'Convert from a hass string to a esphome int representation.'
        return self._inverse[value]