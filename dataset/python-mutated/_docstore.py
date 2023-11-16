"""DocumentStore."""
from pathlib import Path
from typing import Dict, Optional, Union
from azure.ai.generative.index._documents import Document, StaticDocument
from azure.ai.generative.index._utils.logging import get_logger
logger = get_logger(__name__)

class FileBasedDocstore:
    """Simple docstore which serializes to file and loads into memory."""

    def __init__(self, _dict: Optional[Dict[str, Document]]=None):
        if False:
            while True:
                i = 10
        'Initialize with dict.'
        self._dict = _dict if _dict is not None else {}

    def add(self, texts: Dict[str, Document]) -> None:
        if False:
            print('Hello World!')
        '\n        Add texts to in memory dictionary.\n\n        Args:\n        ----\n            texts: dictionary of id -> document.\n\n        Returns:\n        -------\n            None\n        '
        overlapping = set(texts).intersection(self._dict)
        if overlapping:
            raise ValueError(f'Tried to add ids that already exist: {overlapping}')
        self._dict = {**self._dict, **texts}

    def delete(self, ids: list) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Deleting IDs from in memory dictionary.'
        overlapping = set(ids).intersection(self._dict)
        if not overlapping:
            raise ValueError(f'Tried to delete ids that does not  exist: {ids}')
        for _id in ids:
            self._dict.pop(_id)

    def search(self, search: str) -> Union[Document, str]:
        if False:
            print('Hello World!')
        '\n        Search via direct lookup.\n\n        Args:\n        ----\n            search: id of a document to search for.\n\n        Returns:\n        -------\n            Document if found, else error message.\n        '
        if search not in self._dict:
            return f'ID {search} not found.'
        else:
            return self._dict[search]

    def save(self, output_path: str):
        if False:
            return 10
        '\n        Save to JSONL file.\n\n        Args:\n        ----\n            output_path: folder to save doctore contents in.\n        '
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / 'docs.jsonl').open('w', encoding='utf-8') as f:
            for doc in self._dict.values():
                json_line = doc.dumps()
                f.write(json_line + '\n')

    @classmethod
    def load(cls, input_path: str) -> 'FileBasedDocstore':
        if False:
            print('Hello World!')
        'Load from JSONL file.'
        from fsspec.core import url_to_fs
        (fs, uri) = url_to_fs(input_path)
        documents = {}
        with fs.open(f"{input_path.rstrip('/')}/docs.jsonl") as f:
            for line in f:
                document = StaticDocument.loads(line.strip())
                documents[document.document_id] = document
        return cls(documents)