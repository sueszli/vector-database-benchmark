from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from litestar.repository.exceptions import NotFoundError
if TYPE_CHECKING:
    from litestar.repository.filters import FilterTypes
T = TypeVar('T')
CollectionT = TypeVar('CollectionT')

class AbstractSyncRepository(Generic[T], metaclass=ABCMeta):
    """Interface for persistent data interaction."""
    model_type: type[T]
    'Type of object represented by the repository.'
    id_attribute: Any = 'id'
    'Name of the primary identifying attribute on :attr:`model_type`.'

    def __init__(self, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Repository constructors accept arbitrary kwargs.'
        super().__init__(**kwargs)

    @abstractmethod
    def add(self, data: T) -> T:
        if False:
            return 10
        'Add ``data`` to the collection.\n\n        Args:\n            data: Instance to be added to the collection.\n\n        Returns:\n            The added instance.\n        '

    @abstractmethod
    def add_many(self, data: list[T]) -> list[T]:
        if False:
            print('Hello World!')
        'Add multiple ``data`` to the collection.\n\n        Args:\n            data: Instances to be added to the collection.\n\n        Returns:\n            The added instances.\n        '

    @abstractmethod
    def count(self, *filters: FilterTypes, **kwargs: Any) -> int:
        if False:
            return 10
        'Get the count of records returned by a query.\n\n        Args:\n            *filters: Types for specific filtering operations.\n            **kwargs: Instance attribute value filters.\n\n        Returns:\n            The count of instances\n        '

    @abstractmethod
    def delete(self, item_id: Any) -> T:
        if False:
            i = 10
            return i + 15
        'Delete instance identified by ``item_id``.\n\n        Args:\n            item_id: Identifier of instance to be deleted.\n\n        Returns:\n            The deleted instance.\n\n        Raises:\n            NotFoundError: If no instance found identified by ``item_id``.\n        '

    @abstractmethod
    def delete_many(self, item_ids: list[Any]) -> list[T]:
        if False:
            return 10
        'Delete multiple instances identified by list of IDs ``item_ids``.\n\n        Args:\n            item_ids: list of Identifiers to be deleted.\n\n        Returns:\n            The deleted instances.\n        '

    @abstractmethod
    def exists(self, *filters: FilterTypes, **kwargs: Any) -> bool:
        if False:
            i = 10
            return i + 15
        'Return true if the object specified by ``kwargs`` exists.\n\n        Args:\n            *filters: Types for specific filtering operations.\n            **kwargs: Identifier of the instance to be retrieved.\n\n        Returns:\n            True if the instance was found.  False if not found.\n\n        '

    @abstractmethod
    def get(self, item_id: Any, **kwargs: Any) -> T:
        if False:
            i = 10
            return i + 15
        'Get instance identified by ``item_id``.\n\n        Args:\n            item_id: Identifier of the instance to be retrieved.\n            **kwargs: Additional arguments\n\n        Returns:\n            The retrieved instance.\n\n        Raises:\n            NotFoundError: If no instance found identified by ``item_id``.\n        '

    @abstractmethod
    def get_one(self, **kwargs: Any) -> T:
        if False:
            while True:
                i = 10
        'Get an instance specified by the ``kwargs`` filters if it exists.\n\n        Args:\n            **kwargs: Instance attribute value filters.\n\n        Returns:\n            The retrieved instance.\n\n        Raises:\n            NotFoundError: If no instance found identified by ``kwargs``.\n        '

    @abstractmethod
    def get_or_create(self, **kwargs: Any) -> tuple[T, bool]:
        if False:
            for i in range(10):
                print('nop')
        'Get an instance specified by the ``kwargs`` filters if it exists or create it.\n\n        Args:\n            **kwargs: Instance attribute value filters.\n\n        Returns:\n            A tuple that includes the retrieved or created instance, and a boolean on whether the record was created or not\n        '

    @abstractmethod
    def get_one_or_none(self, **kwargs: Any) -> T | None:
        if False:
            i = 10
            return i + 15
        'Get an instance if it exists or None.\n\n        Args:\n            **kwargs: Instance attribute value filters.\n\n        Returns:\n            The retrieved instance or None.\n        '

    @abstractmethod
    def update(self, data: T) -> T:
        if False:
            return 10
        'Update instance with the attribute values present on ``data``.\n\n        Args:\n            data: An instance that should have a value for :attr:`id_attribute <AbstractAsyncRepository.id_attribute>` that exists in the\n                collection.\n\n        Returns:\n            The updated instance.\n\n        Raises:\n            NotFoundError: If no instance found with same identifier as ``data``.\n        '

    @abstractmethod
    def update_many(self, data: list[T]) -> list[T]:
        if False:
            return 10
        'Update multiple instances with the attribute values present on instances in ``data``.\n\n        Args:\n            data: A list of instance that should have a value for :attr:`id_attribute <AbstractAsyncRepository.id_attribute>` that exists in the\n                collection.\n\n        Returns:\n            a list of the updated instances.\n\n        Raises:\n            NotFoundError: If no instance found with same identifier as ``data``.\n        '

    @abstractmethod
    def upsert(self, data: T) -> T:
        if False:
            print('Hello World!')
        "Update or create instance.\n\n        Updates instance with the attribute values present on ``data``, or creates a new instance if\n        one doesn't exist.\n\n        Args:\n            data: Instance to update existing, or be created. Identifier used to determine if an\n                existing instance exists is the value of an attribute on ``data`` named as value of\n                :attr:`id_attribute <AbstractAsyncRepository.id_attribute>`.\n\n        Returns:\n            The updated or created instance.\n\n        Raises:\n            NotFoundError: If no instance found with same identifier as ``data``.\n        "

    @abstractmethod
    def upsert_many(self, data: list[T]) -> list[T]:
        if False:
            while True:
                i = 10
        "Update or create multiple instances.\n\n        Update instances with the attribute values present on ``data``, or create a new instance if\n        one doesn't exist.\n\n        Args:\n            data: Instances to update or created. Identifier used to determine if an\n                existing instance exists is the value of an attribute on ``data`` named as value of\n                :attr:`id_attribute <AbstractAsyncRepository.id_attribute>`.\n\n        Returns:\n            The updated or created instances.\n\n        Raises:\n            NotFoundError: If no instance found with same identifier as ``data``.\n        "

    @abstractmethod
    def list_and_count(self, *filters: FilterTypes, **kwargs: Any) -> tuple[list[T], int]:
        if False:
            return 10
        'List records with total count.\n\n        Args:\n            *filters: Types for specific filtering operations.\n            **kwargs: Instance attribute value filters.\n\n        Returns:\n            a tuple containing The list of instances, after filtering applied, and a count of records returned by query, ignoring pagination.\n        '

    @abstractmethod
    def list(self, *filters: FilterTypes, **kwargs: Any) -> list[T]:
        if False:
            i = 10
            return i + 15
        'Get a list of instances, optionally filtered.\n\n        Args:\n            *filters: filters for specific filtering operations\n            **kwargs: Instance attribute value filters.\n\n        Returns:\n            The list of instances, after filtering applied\n        '

    @abstractmethod
    def filter_collection_by_kwargs(self, collection: CollectionT, /, **kwargs: Any) -> CollectionT:
        if False:
            return 10
        "Filter the collection by kwargs.\n\n        Has ``AND`` semantics where multiple kwargs name/value pairs are provided.\n\n        Args:\n            collection: the objects to be filtered\n            **kwargs: key/value pairs such that objects remaining in the collection after filtering\n                have the property that their attribute named ``key`` has value equal to ``value``.\n\n\n        Returns:\n            The filtered objects\n\n        Raises:\n            RepositoryError: if a named attribute doesn't exist on :attr:`model_type <AbstractAsyncRepository.model_type>`.\n        "

    @staticmethod
    def check_not_found(item_or_none: T | None) -> T:
        if False:
            while True:
                i = 10
        'Raise :class:`NotFoundError` if ``item_or_none`` is ``None``.\n\n        Args:\n            item_or_none: Item (:class:`T <T>`) to be tested for existence.\n\n        Returns:\n            The item, if it exists.\n        '
        if item_or_none is None:
            raise NotFoundError('No item found when one was expected')
        return item_or_none

    @classmethod
    def get_id_attribute_value(cls, item: T | type[T], id_attribute: str | None=None) -> Any:
        if False:
            i = 10
            return i + 15
        'Get value of attribute named as :attr:`id_attribute <AbstractAsyncRepository.id_attribute>` on ``item``.\n\n        Args:\n            item: Anything that should have an attribute named as :attr:`id_attribute <AbstractAsyncRepository.id_attribute>` value.\n            id_attribute: Allows customization of the unique identifier to use for model fetching.\n                Defaults to `None`, but can reference any surrogate or candidate key for the table.\n\n        Returns:\n            The value of attribute on ``item`` named as :attr:`id_attribute <AbstractAsyncRepository.id_attribute>`.\n        '
        return getattr(item, id_attribute if id_attribute is not None else cls.id_attribute)

    @classmethod
    def set_id_attribute_value(cls, item_id: Any, item: T, id_attribute: str | None=None) -> T:
        if False:
            return 10
        'Return the ``item`` after the ID is set to the appropriate attribute.\n\n        Args:\n            item_id: Value of ID to be set on instance\n            item: Anything that should have an attribute named as :attr:`id_attribute <AbstractAsyncRepository.id_attribute>` value.\n            id_attribute: Allows customization of the unique identifier to use for model fetching.\n                Defaults to `None`, but can reference any surrogate or candidate key for the table.\n\n        Returns:\n            Item with ``item_id`` set to :attr:`id_attribute <AbstractAsyncRepository.id_attribute>`\n        '
        setattr(item, id_attribute if id_attribute is not None else cls.id_attribute, item_id)
        return item