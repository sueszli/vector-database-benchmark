"""Tools for manipulating DBRefs (references to MongoDB documents)."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Optional
from bson._helpers import _getstate_slots, _setstate_slots
from bson.son import SON

class DBRef:
    """A reference to a document stored in MongoDB."""
    __slots__ = ('__collection', '__id', '__database', '__kwargs')
    __getstate__ = _getstate_slots
    __setstate__ = _setstate_slots
    _type_marker = 100

    def __init__(self, collection: str, id: Any, database: Optional[str]=None, _extra: Optional[Mapping[str, Any]]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize a new :class:`DBRef`.\n\n        Raises :class:`TypeError` if `collection` or `database` is not\n        an instance of :class:`str`. `database` is optional and allows\n        references to documents to work across databases. Any additional\n        keyword arguments will create additional fields in the resultant\n        embedded document.\n\n        :Parameters:\n          - `collection`: name of the collection the document is stored in\n          - `id`: the value of the document\'s ``"_id"`` field\n          - `database` (optional): name of the database to reference\n          - `**kwargs` (optional): additional keyword arguments will\n            create additional, custom fields\n\n        .. seealso:: The MongoDB documentation on `dbrefs <https://dochub.mongodb.org/core/dbrefs>`_.\n        '
        if not isinstance(collection, str):
            raise TypeError('collection must be an instance of str')
        if database is not None and (not isinstance(database, str)):
            raise TypeError('database must be an instance of str')
        self.__collection = collection
        self.__id = id
        self.__database = database
        kwargs.update(_extra or {})
        self.__kwargs = kwargs

    @property
    def collection(self) -> str:
        if False:
            return 10
        "Get the name of this DBRef's collection."
        return self.__collection

    @property
    def id(self) -> Any:
        if False:
            while True:
                i = 10
        "Get this DBRef's _id."
        return self.__id

    @property
    def database(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        "Get the name of this DBRef's database.\n\n        Returns None if this DBRef doesn't specify a database.\n        "
        return self.__database

    def __getattr__(self, key: Any) -> Any:
        if False:
            i = 10
            return i + 15
        try:
            return self.__kwargs[key]
        except KeyError:
            raise AttributeError(key) from None

    def as_doc(self) -> SON[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Get the SON document representation of this DBRef.\n\n        Generally not needed by application developers\n        '
        doc = SON([('$ref', self.collection), ('$id', self.id)])
        if self.database is not None:
            doc['$db'] = self.database
        doc.update(self.__kwargs)
        return doc

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        extra = ''.join([f', {k}={v!r}' for (k, v) in self.__kwargs.items()])
        if self.database is None:
            return f'DBRef({self.collection!r}, {self.id!r}{extra})'
        return f'DBRef({self.collection!r}, {self.id!r}, {self.database!r}{extra})'

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(other, DBRef):
            us = (self.__database, self.__collection, self.__id, self.__kwargs)
            them = (other.__database, other.__collection, other.__id, other.__kwargs)
            return us == them
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Get a hash value for this :class:`DBRef`.'
        return hash((self.__collection, self.__id, self.__database, tuple(sorted(self.__kwargs.items()))))

    def __deepcopy__(self, memo: Any) -> DBRef:
        if False:
            return 10
        'Support function for `copy.deepcopy()`.'
        return DBRef(deepcopy(self.__collection, memo), deepcopy(self.__id, memo), deepcopy(self.__database, memo), deepcopy(self.__kwargs, memo))