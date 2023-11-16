import logging
from pathlib import Path
from typing import List, Literal, Optional, Union
from haystack.nodes.ranker.base import BaseRanker
from haystack.schema import Document
from haystack.lazy_imports import LazyImport
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from sentence_transformers import SentenceTransformer
    from haystack.modeling.utils import initialize_device_settings

class DiversityRanker(BaseRanker):
    """
    Implements a document ranking algorithm that orders documents in such a way as to maximize the overall diversity
    of the documents.
    """

    def __init__(self, model_name_or_path: Union[str, Path]='all-MiniLM-L6-v2', top_k: Optional[int]=None, use_gpu: Optional[bool]=True, devices: Optional[List[Union[str, 'torch.device']]]=None, similarity: Literal['dot_product', 'cosine']='dot_product'):
        if False:
            while True:
                i = 10
        '\n        Initialize a DiversityRanker.\n\n        :param model_name_or_path: Path to a pretrained sentence-transformers model.\n        :param top_k: The maximum number of documents to return.\n        :param use_gpu: Whether to use GPU (if available). If no GPUs are available, it falls back on a CPU.\n        :param devices: List of torch devices (for example, cuda:0, cpu, mps) to limit inference to specific devices.\n        :param similarity: Whether to use dot product or cosine similarity. Can be set to "dot_product" (default) or "cosine".\n        '
        torch_and_transformers_import.check()
        super().__init__()
        self.top_k = top_k
        (self.devices, _) = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)
        self.model = SentenceTransformer(model_name_or_path, device=str(self.devices[0]))
        self.similarity = similarity

    def predict(self, query: str, documents: List[Document], top_k: Optional[int]=None) -> List[Document]:
        if False:
            print('Hello World!')
        '\n        Rank the documents based on their diversity and return the top_k documents.\n\n        :param query: The query.\n        :param documents: A list of Document objects that should be ranked.\n        :param top_k: The maximum number of documents to return.\n\n        :return: A list of top_k documents ranked based on diversity.\n        '
        if query is None or len(query) == 0:
            raise ValueError('Query is empty')
        if documents is None or len(documents) == 0:
            raise ValueError('No documents to choose from')
        top_k = top_k or self.top_k
        diversity_sorted = self.greedy_diversity_order(query=query, documents=documents)
        return diversity_sorted[:top_k]

    def greedy_diversity_order(self, query: str, documents: List[Document]) -> List[Document]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Orders the given list of documents to maximize diversity. The algorithm first calculates embeddings for\n        each document and the query. It starts by selecting the document that is semantically closest to the query.\n        Then, for each remaining document, it selects the one that, on average, is least similar to the already\n        selected documents. This process continues until all documents are selected, resulting in a list where\n        each subsequent document contributes the most to the overall diversity of the selected set.\n\n        :param query: The search query.\n        :param documents: The list of Document objects to be ranked.\n\n        :return: A list of documents ordered to maximize diversity.\n        '
        doc_embeddings: torch.Tensor = self.model.encode([d.content for d in documents], convert_to_tensor=True)
        query_embedding: torch.Tensor = self.model.encode([query], convert_to_tensor=True)
        if self.similarity == 'dot_product':
            doc_embeddings /= torch.norm(doc_embeddings, p=2, dim=-1).unsqueeze(-1)
            query_embedding /= torch.norm(query_embedding, p=2, dim=-1).unsqueeze(-1)
        n = len(documents)
        selected: List[int] = []
        query_doc_sim: torch.Tensor = query_embedding @ doc_embeddings.T
        selected.append(int(torch.argmax(query_doc_sim).item()))
        selected_sum = doc_embeddings[selected[0]] / n
        while len(selected) < n:
            similarities = selected_sum @ doc_embeddings.T
            similarities[selected] = torch.inf
            index_unselected = int(torch.argmin(similarities).item())
            selected.append(index_unselected)
            selected_sum += doc_embeddings[index_unselected] / n
        ranked_docs: List[Document] = [documents[i] for i in selected]
        return ranked_docs

    def predict_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[float]=None, batch_size: Optional[int]=None) -> Union[List[Document], List[List[Document]]]:
        if False:
            while True:
                i = 10
        '\n        Rank the documents based on their diversity and return the top_k documents.\n\n        :param queries: The queries.\n        :param documents: A list (or a list of lists) of Document objects that should be ranked.\n        :param top_k: The maximum number of documents to return.\n        :param batch_size: The number of documents to process in one batch.\n\n        :return: A list (or a list of lists) of top_k documents ranked based on diversity.\n        '
        if queries is None or len(queries) == 0:
            raise ValueError('No queries to choose from')
        if documents is None or len(documents) == 0:
            raise ValueError('No documents to choose from')
        if len(documents) > 0 and isinstance(documents[0], Document):
            if len(queries) != 1:
                raise ValueError('Number of queries must be 1 if a single list of Documents is provided.')
            return self.predict(query=queries[0], documents=documents, top_k=top_k)
        else:
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise ValueError('Number of queries must be equal to number of provided Document lists.')
            results = []
            for (query, cur_docs) in zip(queries, documents):
                results.append(self.predict(query=query, documents=cur_docs, top_k=top_k))
            return results