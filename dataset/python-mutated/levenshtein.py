from typing import Dict, List
import Levenshtein_search
from .core import Enumerator
from .index import Index

class LevenshteinIndex(Index):
    _doc_to_id: Dict[str, int]

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.index_key = Levenshtein_search.populate_wordset(-1, [])
        self._doc_to_id = Enumerator(start=1)

    def index(self, doc: str) -> None:
        if False:
            while True:
                i = 10
        if doc not in self._doc_to_id:
            self._doc_to_id[doc]
            Levenshtein_search.add_string(self.index_key, doc)

    def unindex(self, doc: str) -> None:
        if False:
            i = 10
            return i + 15
        del self._doc_to_id[doc]
        Levenshtein_search.clear_wordset(self.index_key)
        self.index_key = Levenshtein_search.populate_wordset(-1, list(self._doc_to_id))

    def initSearch(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def search(self, doc: str, threshold: int=0) -> List[int]:
        if False:
            print('Hello World!')
        matching_docs = Levenshtein_search.lookup(self.index_key, doc, threshold)
        if matching_docs:
            return [self._doc_to_id[match] for (match, _, _) in matching_docs]
        else:
            return []

    def __del__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        Levenshtein_search.clear_wordset(self.index_key)