from __future__ import annotations
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, DefaultDict, Generator, Iterable, List, Sequence, Union
    import dedupe.predicates
    from dedupe._typing import Data, Record, RecordID
    from dedupe.index import Index
    Docs = Union[Iterable[str], Iterable[Iterable[str]]]
    IndexList = DefaultDict[str, List[dedupe.predicates.IndexPredicate]]
logger = logging.getLogger(__name__)

def index_list() -> IndexList:
    if False:
        while True:
            i = 10
    return defaultdict(list)

class Fingerprinter(object):
    """Takes in a record and returns all blocks that record belongs to"""

    def __init__(self, predicates: Iterable[dedupe.predicates.Predicate]) -> None:
        if False:
            print('Hello World!')
        self.predicates = predicates
        self.index_fields: dict[str, IndexList]
        self.index_fields = defaultdict(index_list)
        '\n        A dictionary of all the fingerprinter methods that use an\n        index of data field values. The keys are the field names,\n        which can be useful to know for indexing the data.\n        '
        self.index_predicates = []
        for full_predicate in predicates:
            for predicate in full_predicate:
                if hasattr(predicate, 'index'):
                    self.index_fields[predicate.field][predicate.type].append(predicate)
                    self.index_predicates.append(predicate)

    def __call__(self, records: Iterable[Record], target: bool=False) -> Generator[tuple[str, RecordID], None, None]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate the predicates for records. Yields tuples of (predicate,\n        record_id).\n\n        Args:\n            records: A sequence of tuples of (record_id,\n                  record_dict). Can often be created by\n                  `data_dict.items()`.\n            target: Indicates whether the data should be treated as\n                    the target data. This effects the behavior of\n                    search predicates. If `target` is set to\n                    `True`, an search predicate will return the\n                    value itself. If `target` is set to `False` the\n                    search predicate will return all possible\n                    values within the specified search distance.\n\n                    Let\'s say we have a\n                    `LevenshteinSearchPredicate` with an associated\n                    distance of `1` on a `"name"` field; and we\n                    have a record like `{"name": "thomas"}`. If the\n                    `target` is set to `True` then the predicate\n                    will return `"thomas"`.  If `target` is set to\n                    `False`, then the blocker could return\n                    `"thomas"`, `"tomas"`, and `"thoms"`. By using\n                    the `target` argument on one of your datasets,\n                    you will dramatically reduce the total number\n                    of comparisons without a loss of accuracy.\n\n        .. code:: python\n\n           > data = [(1, {\'name\' : \'bob\'}), (2, {\'name\' : \'suzanne\'})]\n           > blocked_ids = deduper.fingerprinter(data)\n           > print list(blocked_ids)\n           [(\'foo:1\', 1), ..., (\'bar:1\', 100)]\n\n        '
        start_time = time.perf_counter()
        predicates = [(':' + str(i), predicate) for (i, predicate) in enumerate(self.predicates)]
        for (i, record) in enumerate(records):
            (record_id, instance) = record
            for (pred_id, predicate) in predicates:
                block_keys = predicate(instance, target=target)
                for block_key in block_keys:
                    yield (block_key + pred_id, record_id)
            if i and i % 10000 == 0:
                logger.info('%(iteration)d, %(elapsed)f2 seconds', {'iteration': i, 'elapsed': time.perf_counter() - start_time})

    def reset_indices(self) -> None:
        if False:
            return 10
        '\n        Fingeprinter indicdes can take up a lot of memory. If you are\n        done with blocking, the method will reset the indices to free up.\n        If you need to block again, the data will need to be re-indexed.\n        '
        for predicate in self.index_predicates:
            predicate.reset()

    def index(self, docs: Docs, field: str) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Add docs to the indices used by fingerprinters.\n\n        Some fingerprinter methods depend upon having an index of\n        values that a field may have in the data. This method adds\n        those values to the index. If you don't have any fingerprinter\n        methods that use an index, this method will do nothing.\n\n        Args:\n            docs: an iterator of values from your data to index. While\n                  not required, it is recommended that docs be a unique\n                  set of of those values. Indexing can be an expensive\n                  operation.\n            field: fieldname or key associated with the values you are\n                   indexing\n\n        "
        indices = extractIndices(self.index_fields[field])
        for doc in docs:
            if doc:
                for (_, index, preprocess) in indices:
                    index.index(preprocess(doc))
        for (index_type, index, _) in indices:
            index.initSearch()
            for predicate in self.index_fields[field][index_type]:
                logger.debug('Canopy: %s', str(predicate))
                predicate.index = index
                predicate.bust_cache()

    def unindex(self, docs: Docs, field: str) -> None:
        if False:
            print('Hello World!')
        'Remove docs from indices used by fingerprinters\n\n        Args:\n            docs: an iterator of values from your data to remove. While\n                  not required, it is recommended that docs be a unique\n                  set of of those values. Indexing can be an expensive\n                  operation.\n            field: fieldname or key associated with the values you are\n                   unindexing\n        '
        indices = extractIndices(self.index_fields[field])
        for doc in docs:
            if doc:
                for (_, index, preprocess) in indices:
                    try:
                        index.unindex(preprocess(doc))
                    except KeyError:
                        pass
        for (index_type, index, _) in indices:
            index.initSearch()
            for predicate in self.index_fields[field][index_type]:
                logger.debug('Canopy: %s', str(predicate))
                predicate.index = index
                predicate.bust_cache()

    def index_all(self, data: Data) -> None:
        if False:
            return 10
        for field in self.index_fields:
            unique_fields = {record[field] for record in data.values() if record[field]}
            self.index(unique_fields, field)

def extractIndices(index_fields: IndexList) -> Sequence[tuple[str, Index, Callable[[Any], Any]]]:
    if False:
        for i in range(10):
            print('nop')
    indices = []
    for (index_type, predicates) in index_fields.items():
        predicate = predicates[0]
        index = predicate.index
        preprocess = predicate.preprocess
        if predicate.index is None:
            index = predicate.initIndex()
        assert index is not None
        indices.append((index_type, index, preprocess))
    return indices