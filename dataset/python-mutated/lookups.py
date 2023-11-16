from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import srsly
from preshed.bloom import BloomFilter
from .errors import Errors
from .strings import get_string_id
from .util import SimpleFrozenDict, ensure_path, load_language_data, registry
UNSET = object()

def load_lookups(lang: str, tables: List[str], strict: bool=True) -> 'Lookups':
    if False:
        for i in range(10):
            print('nop')
    'Load the data from the spacy-lookups-data package for a given language,\n    if available. Returns an empty `Lookups` container if there\'s no data or if the package\n    is not installed.\n\n    lang (str): The language code (corresponds to entry point exposed by\n        the spacy-lookups-data package).\n    tables (List[str]): Name of tables to load, e.g. ["lemma_lookup", "lemma_exc"]\n    strict (bool): Whether to raise an error if a table doesn\'t exist.\n    RETURNS (Lookups): The lookups container containing the loaded tables.\n    '
    lookups = Lookups()
    if lang not in registry.lookups:
        if strict and len(tables) > 0:
            raise ValueError(Errors.E955.format(table=', '.join(tables), lang=lang))
        return lookups
    data = registry.lookups.get(lang)
    for table in tables:
        if table not in data:
            if strict:
                raise ValueError(Errors.E955.format(table=table, lang=lang))
            language_data = {}
        else:
            language_data = load_language_data(data[table])
        lookups.add_table(table, language_data)
    return lookups

class Table(OrderedDict):
    """A table in the lookups. Subclass of builtin dict that implements a
    slightly more consistent and unified API.

    Includes a Bloom filter to speed up missed lookups.
    """

    @classmethod
    def from_dict(cls, data: dict, name: Optional[str]=None) -> 'Table':
        if False:
            print('Hello World!')
        'Initialize a new table from a dict.\n\n        data (dict): The dictionary.\n        name (str): Optional table name for reference.\n\n        DOCS: https://spacy.io/api/lookups#table.from_dict\n        '
        self = cls(name=name)
        self.update(data)
        return self

    def __init__(self, name: Optional[str]=None, data: Optional[dict]=None) -> None:
        if False:
            return 10
        'Initialize a new table.\n\n        name (str): Optional table name for reference.\n        data (dict): Initial data, used to hint Bloom Filter.\n\n        DOCS: https://spacy.io/api/lookups#table.init\n        '
        OrderedDict.__init__(self)
        self.name = name
        self.default_size = 1000000.0
        size = max(len(data), 1) if data is not None else self.default_size
        self.bloom = BloomFilter.from_error_rate(size)
        if data:
            self.update(data)

    def __setitem__(self, key: Union[str, int], value: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Set new key/value pair. String keys will be hashed.\n\n        key (str / int): The key to set.\n        value: The value to set.\n        '
        key = get_string_id(key)
        OrderedDict.__setitem__(self, key, value)
        self.bloom.add(key)

    def set(self, key: Union[str, int], value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set new key/value pair. String keys will be hashed.\n        Same as table[key] = value.\n\n        key (str / int): The key to set.\n        value: The value to set.\n        '
        self[key] = value

    def __getitem__(self, key: Union[str, int]) -> Any:
        if False:
            print('Hello World!')
        'Get the value for a given key. String keys will be hashed.\n\n        key (str / int): The key to get.\n        RETURNS: The value.\n        '
        key = get_string_id(key)
        return OrderedDict.__getitem__(self, key)

    def get(self, key: Union[str, int], default: Optional[Any]=None) -> Any:
        if False:
            return 10
        'Get the value for a given key. String keys will be hashed.\n\n        key (str / int): The key to get.\n        default: The default value to return.\n        RETURNS: The value.\n        '
        key = get_string_id(key)
        return OrderedDict.get(self, key, default)

    def __contains__(self, key: Union[str, int]) -> bool:
        if False:
            while True:
                i = 10
        'Check whether a key is in the table. String keys will be hashed.\n\n        key (str / int): The key to check.\n        RETURNS (bool): Whether the key is in the table.\n        '
        key = get_string_id(key)
        if key not in self.bloom:
            return False
        return OrderedDict.__contains__(self, key)

    def to_bytes(self) -> bytes:
        if False:
            print('Hello World!')
        'Serialize table to a bytestring.\n\n        RETURNS (bytes): The serialized table.\n\n        DOCS: https://spacy.io/api/lookups#table.to_bytes\n        '
        data = {'name': self.name, 'dict': dict(self.items()), 'bloom': self.bloom.to_bytes()}
        return srsly.msgpack_dumps(data)

    def from_bytes(self, bytes_data: bytes) -> 'Table':
        if False:
            while True:
                i = 10
        'Load a table from a bytestring.\n\n        bytes_data (bytes): The data to load.\n        RETURNS (Table): The loaded table.\n\n        DOCS: https://spacy.io/api/lookups#table.from_bytes\n        '
        loaded = srsly.msgpack_loads(bytes_data)
        data = loaded.get('dict', {})
        self.name = loaded['name']
        self.bloom = BloomFilter().from_bytes(loaded['bloom'])
        self.clear()
        self.update(data)
        return self

class Lookups:
    """Container for large lookup tables and dictionaries, e.g. lemmatization
    data or tokenizer exception lists. Lookups are available via vocab.lookups,
    so they can be accessed before the pipeline components are applied (e.g.
    in the tokenizer and lemmatizer), as well as within the pipeline components
    via doc.vocab.lookups.
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize the Lookups object.\n\n        DOCS: https://spacy.io/api/lookups#init\n        '
        self._tables: Dict[str, Table] = {}

    def __contains__(self, name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if the lookups contain a table of a given name. Delegates to\n        Lookups.has_table.\n\n        name (str): Name of the table.\n        RETURNS (bool): Whether a table of that name is in the lookups.\n        '
        return self.has_table(name)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'RETURNS (int): The number of tables in the lookups.'
        return len(self._tables)

    @property
    def tables(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'RETURNS (List[str]): Names of all tables in the lookups.'
        return list(self._tables.keys())

    def add_table(self, name: str, data: dict=SimpleFrozenDict()) -> Table:
        if False:
            while True:
                i = 10
        'Add a new table to the lookups. Raises an error if the table exists.\n\n        name (str): Unique name of table.\n        data (dict): Optional data to add to the table.\n        RETURNS (Table): The newly added table.\n\n        DOCS: https://spacy.io/api/lookups#add_table\n        '
        if name in self.tables:
            raise ValueError(Errors.E158.format(name=name))
        table = Table(name=name, data=data)
        self._tables[name] = table
        return table

    def set_table(self, name: str, table: Table) -> None:
        if False:
            return 10
        'Set a table.\n\n        name (str): Name of the table to set.\n        table (Table): The Table to set.\n\n        DOCS: https://spacy.io/api/lookups#set_table\n        '
        self._tables[name] = table

    def get_table(self, name: str, default: Any=UNSET) -> Table:
        if False:
            return 10
        "Get a table. Raises an error if the table doesn't exist and no\n        default value is provided.\n\n        name (str): Name of the table.\n        default (Any): Optional default value to return if table doesn't exist.\n        RETURNS (Table): The table.\n\n        DOCS: https://spacy.io/api/lookups#get_table\n        "
        if name not in self._tables:
            if default == UNSET:
                raise KeyError(Errors.E159.format(name=name, tables=self.tables))
            return default
        return self._tables[name]

    def remove_table(self, name: str) -> Table:
        if False:
            for i in range(10):
                print('nop')
        "Remove a table. Raises an error if the table doesn't exist.\n\n        name (str): Name of the table to remove.\n        RETURNS (Table): The removed table.\n\n        DOCS: https://spacy.io/api/lookups#remove_table\n        "
        if name not in self._tables:
            raise KeyError(Errors.E159.format(name=name, tables=self.tables))
        return self._tables.pop(name)

    def has_table(self, name: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if the lookups contain a table of a given name.\n\n        name (str): Name of the table.\n        RETURNS (bool): Whether a table of that name exists.\n\n        DOCS: https://spacy.io/api/lookups#has_table\n        '
        return name in self._tables

    def to_bytes(self, **kwargs) -> bytes:
        if False:
            return 10
        'Serialize the lookups to a bytestring.\n\n        RETURNS (bytes): The serialized Lookups.\n\n        DOCS: https://spacy.io/api/lookups#to_bytes\n        '
        return srsly.msgpack_dumps(self._tables)

    def from_bytes(self, bytes_data: bytes, **kwargs) -> 'Lookups':
        if False:
            i = 10
            return i + 15
        'Load the lookups from a bytestring.\n\n        bytes_data (bytes): The data to load.\n        RETURNS (Lookups): The loaded Lookups.\n\n        DOCS: https://spacy.io/api/lookups#from_bytes\n        '
        self._tables = {}
        for (key, value) in srsly.msgpack_loads(bytes_data).items():
            self._tables[key] = Table(key, value)
        return self

    def to_disk(self, path: Union[str, Path], filename: str='lookups.bin', **kwargs) -> None:
        if False:
            while True:
                i = 10
        "Save the lookups to a directory as lookups.bin. Expects a path to a\n        directory, which will be created if it doesn't exist.\n\n        path (str / Path): The file path.\n\n        DOCS: https://spacy.io/api/lookups#to_disk\n        "
        path = ensure_path(path)
        if not path.exists():
            path.mkdir()
        filepath = path / filename
        with filepath.open('wb') as file_:
            file_.write(self.to_bytes())

    def from_disk(self, path: Union[str, Path], filename: str='lookups.bin', **kwargs) -> 'Lookups':
        if False:
            while True:
                i = 10
        "Load lookups from a directory containing a lookups.bin. Will skip\n        loading if the file doesn't exist.\n\n        path (str / Path): The directory path.\n        RETURNS (Lookups): The loaded lookups.\n\n        DOCS: https://spacy.io/api/lookups#from_disk\n        "
        path = ensure_path(path)
        filepath = path / filename
        if filepath.exists():
            with filepath.open('rb') as file_:
                data = file_.read()
            return self.from_bytes(data)
        return self