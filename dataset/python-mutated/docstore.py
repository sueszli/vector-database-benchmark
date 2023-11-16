"""Langchain compatible Docstore which serializes to jsonl."""
from typing import Dict, Union
from azure.ai.generative.index._docstore import FileBasedDocstore
from azure.ai.generative.index._embeddings import WrappedLangChainDocument
from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document as LangChainDocument

class FileBasedDocStore(Docstore, AddableMixin):
    """Simple docstore which serializes to file and loads into memory."""

    def __init__(self, docstore: FileBasedDocstore):
        if False:
            i = 10
            return i + 15
        'Initialize with azure.ai.generative.index._docstore.FileBasedDocstore.'
        self.docstore = docstore

    def add(self, texts: Dict[str, LangChainDocument]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Add texts to in memory dictionary.\n\n        Args:\n        ----\n            texts: dictionary of id -> document.\n\n        Returns:\n        -------\n            None\n        '
        return self.docstore.add({k: WrappedLangChainDocument(v) for (k, v) in texts.items()})

    def delete(self, ids: list) -> None:
        if False:
            print('Hello World!')
        'Deleting IDs from in memory dictionary.'
        return self.docstore.delete(ids)

    def search(self, search: str) -> Union[LangChainDocument, str]:
        if False:
            return 10
        '\n        Search via direct lookup.\n\n        Args:\n        ----\n            search: id of a document to search for.\n\n        Returns:\n        -------\n            Document if found, else error message.\n        '
        doc = self.docstore.search(search)
        return LangChainDocument(page_content=doc.page_content, metadata=doc.metadata) if doc else doc

    def save(self, output_path: str):
        if False:
            print('Hello World!')
        '\n        Save to JSONL file.\n\n        Args:\n        ----\n            output_path: folder to save doctore contents in.\n        '
        return self.docstore.save(output_path)

    @classmethod
    def load(cls, input_path: str) -> 'FileBasedDocstore':
        if False:
            while True:
                i = 10
        'Load from JSONL file.'
        return FileBasedDocStore.load(input_path)