from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec, TypeAlias
    P = ParamSpec('P')
    R = TypeVar('R')
    MutableDataT = TypeVar('MutableDataT', bound='MutableData')
    DataGetter: TypeAlias = Callable[[MutableDataT, Any], Any]
InnerMutableDataT = TypeVar('InnerMutableDataT', bound='dict[str, Any] | list[Any]')

class Mutation:
    ABBR: str

class MutationSet(Mutation):
    """
    Setting a value.
    This mutation is used for MutableDictLikeData and MutableListLikeData.
    """
    ABBR = 'S'

    def __init__(self, key, value):
        if False:
            return 10
        self.key = key
        self.value = value

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'MutationSet({self.key}, {self.value})'

class MutationDel(Mutation):
    """
    Deleting a value.
    This mutation is used for MutableDictLikeData and MutableListLikeData.
    """
    ABBR = 'D'

    def __init__(self, key):
        if False:
            for i in range(10):
                print('nop')
        self.key = key

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'MutationDel({self.key})'

class MutationNew(Mutation):
    """
    Adding a new value.
    This mutation is only used for MutableDictLikeData.
    """
    ABBR = 'N'

    def __init__(self, key, value):
        if False:
            while True:
                i = 10
        self.key = key
        self.value = value

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'MutationNew({self.key}, {self.value})'

class MutationInsert(Mutation):
    """
    Inserting a value.
    This mutation is only used for MutableListLikeData.
    """
    ABBR = 'I'

    def __init__(self, index, value):
        if False:
            while True:
                i = 10
        self.index = index
        self.value = value

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'MutationInsert({self.index}, {self.value})'

class MutationPermutate(Mutation):
    """
    Permutating all the values.
    This mutation is only used for MutableListLikeData.
    """
    ABBR = 'P'

    def __init__(self, permutation):
        if False:
            print('Hello World!')
        self.permutation = permutation

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'MutationPermutate({self.permutation})'

def record_mutation(mutation_fn: Callable[Concatenate[MutableDataT, P], Mutation]) -> Callable[Concatenate[MutableDataT, P], None]:
    if False:
        while True:
            i = 10

    def wrapper(self, *args: P.args, **kwargs: P.kwargs):
        if False:
            print('Hello World!')
        mutation = mutation_fn(self, *args, **kwargs)
        self.records.append(mutation)
    return wrapper

class MutableData(Generic[InnerMutableDataT]):
    """
    An intermediate data structure between data and variable, it records all the mutations.
    """
    read_cache: InnerMutableDataT

    class Empty:

        def __repr__(self):
            if False:
                while True:
                    i = 10
            return 'Empty()'

    def __init__(self, data: Any, getter: DataGetter):
        if False:
            while True:
                i = 10
        self.original_data = data
        self.getter = getter
        self.records: list[Mutation] = []

    def is_empty(self, value):
        if False:
            i = 10
            return i + 15
        return isinstance(value, MutableData.Empty)

    @property
    def version(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.records)

    @property
    def has_changed(self):
        if False:
            return 10
        return self.version != 0

    def rollback(self, version: int):
        if False:
            for i in range(10):
                print('nop')
        assert version <= self.version
        self.records[:] = self.records[:version]

    def get(self, key):
        if False:
            return 10
        raise NotImplementedError()

    def set(self, key, value):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def apply(self, mutation: Mutation, write_cache: InnerMutableDataT):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def reproduce(self, version: int | None=None) -> InnerMutableDataT:
        if False:
            for i in range(10):
                print('nop')
        if version is None:
            version = self.version
        write_cache = self.read_cache.copy()
        for mutation in self.records[:version]:
            self.apply(mutation, write_cache)
        return write_cache

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        records_abbrs = ''.join([mutation.ABBR for mutation in self.records])
        return f'{self.__class__.__name__}({records_abbrs})'

class MutableDictLikeData(MutableData['dict[str, Any]']):

    def __init__(self, data: Any, getter: DataGetter):
        if False:
            i = 10
            return i + 15
        super().__init__(data, getter)
        self.read_cache = {}

    def clear_read_cache(self):
        if False:
            for i in range(10):
                print('nop')
        self.read_cache.clear()

    def get(self, key: Any):
        if False:
            while True:
                i = 10
        write_cache = self.reproduce(self.version)
        if key not in write_cache:
            self.read_cache[key] = self.getter(self, key)
        return self.reproduce(self.version)[key]

    def get_all(self):
        if False:
            while True:
                i = 10
        original_keys = list(self.original_data.keys())
        for mutation in self.records:
            if isinstance(mutation, MutationNew):
                original_keys.append(mutation.key)
            elif isinstance(mutation, MutationDel):
                original_keys.remove(mutation.key)
        return {key: self.get(key) for key in original_keys}

    @record_mutation
    def set(self, key: Any, value: Any) -> Mutation:
        if False:
            i = 10
            return i + 15
        is_new = False
        if self.is_empty(self.get(key)):
            is_new = True
        return MutationSet(key, value) if not is_new else MutationNew(key, value)

    @record_mutation
    def delete(self, key):
        if False:
            return 10
        return MutationDel(key)

    def apply(self, mutation: Mutation, write_cache: dict[str, Any]):
        if False:
            i = 10
            return i + 15
        if isinstance(mutation, MutationNew):
            write_cache[mutation.key] = mutation.value
        elif isinstance(mutation, MutationSet):
            write_cache[mutation.key] = mutation.value
        elif isinstance(mutation, MutationDel):
            write_cache[mutation.key] = MutableData.Empty()
        else:
            raise ValueError(f'Unknown mutation type {mutation}')

    def reproduce(self, version: int | None=None):
        if False:
            i = 10
            return i + 15
        if version is None:
            version = self.version
        write_cache = self.read_cache.copy()
        for mutation in self.records[:version]:
            self.apply(mutation, write_cache)
        return write_cache

class MutableListLikeData(MutableData['list[Any]']):

    def __init__(self, data: Any, getter: DataGetter):
        if False:
            return 10
        super().__init__(data, getter)
        self.read_cache = [self.getter(self, idx) for idx in range(len(self.original_data))]

    def clear_read_cache(self):
        if False:
            i = 10
            return i + 15
        self.read_cache[:] = []

    @property
    def length(self):
        if False:
            return 10
        return len(self.reproduce())

    def get(self, key):
        if False:
            return 10
        write_cache = self.reproduce(self.version)
        return write_cache[key]

    def get_all(self) -> list[Any]:
        if False:
            return 10
        items = self.reproduce(self.version)
        return items

    @record_mutation
    def set(self, key: int, value: Any):
        if False:
            for i in range(10):
                print('nop')
        return MutationSet(self._regularize_index(key), value)

    @record_mutation
    def delete(self, key: int):
        if False:
            return 10
        return MutationDel(self._regularize_index(key))

    @record_mutation
    def insert(self, index: int, value: Any):
        if False:
            print('Hello World!')
        return MutationInsert(self._regularize_index(index), value)

    @record_mutation
    def permutate(self, permutation: list[int]):
        if False:
            return 10
        return MutationPermutate(permutation)

    def _regularize_index(self, index: int):
        if False:
            while True:
                i = 10
        if index < 0:
            index += self.length
        return index

    def apply(self, mutation: Mutation, write_cache: list[Any]):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(mutation, MutationSet):
            write_cache[mutation.key] = mutation.value
        elif isinstance(mutation, MutationDel):
            write_cache[:] = write_cache[:mutation.key] + write_cache[mutation.key + 1:]
        elif isinstance(mutation, MutationInsert):
            write_cache.insert(mutation.index, mutation.value)
        elif isinstance(mutation, MutationPermutate):
            write_cache[:] = [write_cache[i] for i in mutation.permutation]
        else:
            raise ValueError(f'Unknown mutation type {mutation}')