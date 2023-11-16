from typing import Any
from enum import EnumMeta, Enum

class CaseInsensitiveEnumMeta(EnumMeta):
    """Enum metaclass to allow for interoperability with case-insensitive strings.

    Consuming this metaclass in an SDK should be done in the following manner:

    .. code-block:: python

        from enum import Enum
        from azure.core import CaseInsensitiveEnumMeta

        class MyCustomEnum(str, Enum, metaclass=CaseInsensitiveEnumMeta):
            FOO = 'foo'
            BAR = 'bar'

    """

    def __getitem__(cls, name: str) -> Any:
        if False:
            return 10
        return super(CaseInsensitiveEnumMeta, cls).__getitem__(name.upper())

    def __getattr__(cls, name: str) -> Enum:
        if False:
            while True:
                i = 10
        "Return the enum member matching `name`.\n\n        We use __getattr__ instead of descriptors or inserting into the enum\n        class' __dict__ in order to support `name` and `value` being both\n        properties for enum members (which live in the class' __dict__) and\n        enum members themselves.\n\n        :param str name: The name of the enum member to retrieve.\n        :rtype: ~azure.core.CaseInsensitiveEnumMeta\n        :return: The enum member matching `name`.\n        :raises AttributeError: If `name` is not a valid enum member.\n        "
        try:
            return cls._member_map_[name.upper()]
        except KeyError as err:
            raise AttributeError(name) from err