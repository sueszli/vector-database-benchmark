import os


from chunk import Chunk
from loguru import logger
import chromadb

class ChromaVDB():
    """
    ChromaDB implementation of the Vector DB
    Docs: https://docs.trychroma.com/
    """
    def __init__(self, path="./"):
        logger.info("Initializing ChromaVDB. Verify only one instance of this class is created.")
        self.client = chromadb.PersistentClient(path=path)

    def _sanitize_repo_name(self, repo_name: str) -> str:
        return repo_name.replace("/", "_")

    def _get_or_create_repo(self, repo_name: str):
        repo_name = self._sanitize_repo_name(repo_name)
        return self.client.get_or_create_collection(repo_name)

    def add(self, repo_name: str, chunk: Chunk):
        repo_name = self._sanitize_repo_name(repo_name)
        collection = self._get_or_create_repo(repo_name)
        collection.add(
            ids = repo_name + "_" + str(chunk.start_index)+ "_" + str(chunk.end_index),
            embeddings = chunk.embeddings,
            metadatas = {
                "filename": chunk.filename,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            },
            documents = chunk.code
        )
        logger.info("Added chunk to repo: " + repo_name)

    def add_multiple(self, repo_name: str,  chunks: list[Chunk]):
        repo_name = self._sanitize_repo_name(repo_name)
        embeddings = []
        metadata = []
        documents = []
        ids = []

        for chunk in chunks:
            embeddings.append(chunk.embeddings)
            metadata.append({
                "filename": chunk.filename,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            })
            documents.append(chunk.code)
            ids.append(repo_name + "_" + str(chunk.start_index)+ "_" + str(chunk.end_index))

        collection = self._get_or_create_repo(repo_name)
        collection.add(
            embeddings = embeddings,
            metadatas = metadata,
            documents = documents,
            ids = ids
        )

    def query(self, repo_name: str,  query_embedding: list, n_results=10, where={}) -> list[Chunk]:
        repo_name = self._sanitize_repo_name(repo_name)
        collection = self._get_or_create_repo(repo_name)

        query_res = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where
        )
        # query_res looks like- {'ids': [['test_integration_repo_0_10']], 'distances': [[0.0]],
        #                           'metadatas': [[{'end_index': 10, 'filename': 'test_integration.py', 'start_index': 0}]],
        #                            'embeddings': None, 'documents': [['print("Hello, World!")']]}

        ret = []
        if query_res:
            for i in range(len(query_res["ids"])):
                chunk = Chunk()
                chunk.filename = query_res["metadatas"][0][i]["filename"]
                chunk.code = query_res["documents"][0][i]
                chunk.start_index = query_res["metadatas"][0][i]["start_index"]
                chunk.end_index = query_res["metadatas"][0][i]["end_index"] #Whats up with these [0]s??
                ret.append(chunk)

        return ret


    def delete_repo(self, repo_name: str):
        repo_name = self._sanitize_repo_name(repo_name)
        try:
            self.client.delete_collection(repo_name)
            logger.critical("Deleted repo: " + repo_name)
        except ValueError:
            pass #Collection doesn't exist, so we don't need to delete it
