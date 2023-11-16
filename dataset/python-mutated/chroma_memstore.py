from .memstore import MemStore
import chromadb
from chromadb.config import Settings
import hashlib

class ChromaMemStore:
    """
    A class used to represent a Memory Store
    """

    def __init__(self, store_path: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the MemStore with a given store path.\n\n        Args:\n            store_path (str): The path to the store.\n        '
        self.client = chromadb.PersistentClient(path=store_path, settings=Settings(anonymized_telemetry=False))

    def add(self, task_id: str, document: str, metadatas: dict) -> None:
        if False:
            while True:
                i = 10
        '\n        Add a document to the MemStore.\n\n        Args:\n            task_id (str): The ID of the task.\n            document (str): The document to be added.\n            metadatas (dict): The metadata of the document.\n        '
        doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
        collection = self.client.get_or_create_collection(task_id)
        collection.add(documents=[document], metadatas=[metadatas], ids=[doc_id])

    def query(self, task_id: str, query: str, filters: dict=None, document_search: dict=None) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Query the MemStore.\n\n        Args:\n            task_id (str): The ID of the task.\n            query (str): The query string.\n            filters (dict, optional): The filters to be applied. Defaults to None.\n            search_string (str, optional): The search string. Defaults to None.\n\n        Returns:\n            dict: The query results.\n        '
        collection = self.client.get_or_create_collection(task_id)
        kwargs = {'query_texts': [query], 'n_results': 10}
        if filters:
            kwargs['where'] = filters
        if document_search:
            kwargs['where_document'] = document_search
        return collection.query(**kwargs)

    def get(self, task_id: str, doc_ids: list=None, filters: dict=None) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Get documents from the MemStore.\n\n        Args:\n            task_id (str): The ID of the task.\n            doc_ids (list, optional): The IDs of the documents to be retrieved. Defaults to None.\n            filters (dict, optional): The filters to be applied. Defaults to None.\n\n        Returns:\n            dict: The retrieved documents.\n        '
        collection = self.client.get_or_create_collection(task_id)
        kwargs = {}
        if doc_ids:
            kwargs['ids'] = doc_ids
        if filters:
            kwargs['where'] = filters
        return collection.get(**kwargs)

    def update(self, task_id: str, doc_ids: list, documents: list, metadatas: list):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update documents in the MemStore.\n\n        Args:\n            task_id (str): The ID of the task.\n            doc_ids (list): The IDs of the documents to be updated.\n            documents (list): The updated documents.\n            metadatas (list): The updated metadata.\n        '
        collection = self.client.get_or_create_collection(task_id)
        collection.update(ids=doc_ids, documents=documents, metadatas=metadatas)

    def delete(self, task_id: str, doc_id: str):
        if False:
            i = 10
            return i + 15
        '\n        Delete a document from the MemStore.\n\n        Args:\n            task_id (str): The ID of the task.\n            doc_id (str): The ID of the document to be deleted.\n        '
        collection = self.client.get_or_create_collection(task_id)
        collection.delete(ids=[doc_id])
if __name__ == '__main__':
    print('#############################################')
    mem = ChromaMemStore('.agent_mem_store')
    task_id = 'test_task'
    document = 'This is a another new test document.'
    metadatas = {'metadata': 'test_metadata'}
    mem.add(task_id, document, metadatas)
    task_id = 'test_task'
    document = 'The quick brown fox jumps over the lazy dog.'
    metadatas = {'metadata': 'test_metadata'}
    mem.add(task_id, document, metadatas)
    task_id = 'test_task'
    document = 'AI is a new technology that will change the world.'
    metadatas = {'timestamp': 1623936000}
    mem.add(task_id, document, metadatas)
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    query = 'test'
    filters = {'metadata': {'$eq': 'test'}}
    search_string = {'$contains': 'test'}
    doc_ids = [doc_id]
    documents = ['This is an updated test document.']
    updated_metadatas = {'metadata': 'updated_test_metadata'}
    print('Query:')
    print(mem.query(task_id, query))
    print('Get:')
    print(mem.get(task_id))
    print('Update:')
    print(mem.update(task_id, doc_ids, documents, updated_metadatas))
    print('Delete:')
    print(mem.delete(task_id, doc_ids[0]))