import abc
import hashlib
import chromadb
from chromadb.config import Settings

class MemStore(abc.ABC):
    """
    An abstract class that represents a Memory Store
    """

    @abc.abstractmethod
    def __init__(self, store_path: str):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the MemStore with a given store path.\n\n        Args:\n            store_path (str): The path to the store.\n        '
        pass

    @abc.abstractmethod
    def add_task_memory(self, task_id: str, document: str, metadatas: dict) -> None:
        if False:
            while True:
                i = 10
        '\n        Add a document to the current tasks MemStore.\n        This function calls the base version with the task_id as the collection_name.\n\n        Args:\n            task_id (str): The ID of the task.\n            document (str): The document to be added.\n            metadatas (dict): The metadata of the document.\n        '
        self.add(collection_name=task_id, document=document, metadatas=metadatas)

    @abc.abstractmethod
    def query_task_memory(self, task_id: str, query: str, filters: dict=None, document_search: dict=None) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Query the current tasks MemStore.\n        This function calls the base version with the task_id as the collection_name.\n\n        Args:\n            task_id (str): The ID of the task.\n            query (str): The query string.\n            filters (dict, optional): The filters to be applied. Defaults to None.\n            document_search (dict, optional): The search string. Defaults to None.\n\n        Returns:\n            dict: The query results.\n        '
        return self.query(collection_name=task_id, query=query, filters=filters, document_search=document_search)

    @abc.abstractmethod
    def get_task_memory(self, task_id: str, doc_ids: list=None, filters: dict=None) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get documents from the current tasks MemStore.\n        This function calls the base version with the task_id as the collection_name.\n\n        Args:\n            task_id (str): The ID of the task.\n            doc_ids (list, optional): The IDs of the documents to be retrieved. Defaults to None.\n            filters (dict, optional): The filters to be applied. Defaults to None.\n\n        Returns:\n            dict: The retrieved documents.\n        '
        return self.get(collection_name=task_id, doc_ids=doc_ids, filters=filters)

    @abc.abstractmethod
    def update_task_memory(self, task_id: str, doc_ids: list, documents: list, metadatas: list):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update documents in the current tasks MemStore.\n        This function calls the base version with the task_id as the collection_name.\n\n        Args:\n            task_id (str): The ID of the task.\n            doc_ids (list): The IDs of the documents to be updated.\n            documents (list): The updated documents.\n            metadatas (list): The updated metadata.\n        '
        self.update(collection_name=task_id, doc_ids=doc_ids, documents=documents, metadatas=metadatas)

    @abc.abstractmethod
    def delete_task_memory(self, task_id: str, doc_id: str):
        if False:
            return 10
        '\n        Delete a document from the current tasks MemStore.\n        This function calls the base version with the task_id as the collection_name.\n\n        Args:\n            task_id (str): The ID of the task.\n            doc_id (str): The ID of the document to be deleted.\n        '
        self.delete(collection_name=task_id, doc_id=doc_id)

    @abc.abstractmethod
    def add(self, collection_name: str, document: str, metadatas: dict) -> None:
        if False:
            print('Hello World!')
        "\n        Add a document to the current collection's MemStore.\n\n        Args:\n            collection_name (str): The name of the collection.\n            document (str): The document to be added.\n            metadatas (dict): The metadata of the document.\n        "
        pass

    @abc.abstractmethod
    def query(self, collection_name: str, query: str, filters: dict=None, document_search: dict=None) -> dict:
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def get(self, collection_name: str, doc_ids: list=None, filters: dict=None) -> dict:
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def update(self, collection_name: str, doc_ids: list, documents: list, metadatas: list):
        if False:
            return 10
        pass

    @abc.abstractmethod
    def delete(self, collection_name: str, doc_id: str):
        if False:
            return 10
        pass