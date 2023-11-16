from typing import Optional, Union, List
import logging
from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker
logger = logging.getLogger(__name__)

class LostInTheMiddleRanker(BaseRanker):
    """
    The LostInTheMiddleRanker implements a ranker that reorders documents based on the "lost in the middle" order.
    "Lost in the Middle: How Language Models Use Long Contexts" paper by Liu et al. aims to lay out paragraphs into LLM
    context so that the relevant paragraphs are at the beginning or end of the input context, while the least relevant
    information is in the middle of the context.

    See https://arxiv.org/abs/2307.03172 for more details.
    """

    def __init__(self, word_count_threshold: Optional[int]=None, top_k: Optional[int]=None):
        if False:
            while True:
                i = 10
        "\n        Creates an instance of LostInTheMiddleRanker.\n\n        If 'word_count_threshold' is specified, this ranker includes all documents up until the point where adding\n        another document would exceed the 'word_count_threshold'. The last document that causes the threshold to\n        be breached will be included in the resulting list of documents, but all subsequent documents will be\n        discarded.\n\n        :param word_count_threshold: The maximum total number of words across all documents selected by the ranker.\n        :param top_k: The maximum number of documents to return.\n        "
        super().__init__()
        if isinstance(word_count_threshold, int) and word_count_threshold <= 0:
            raise ValueError(f'Invalid value for word_count_threshold: {word_count_threshold}. word_count_threshold must be a positive integer.')
        self.word_count_threshold = word_count_threshold
        self.top_k = top_k

    def reorder_documents(self, documents: List[Document]) -> List[Document]:
        if False:
            print('Hello World!')
        '\n        Ranks documents based on the "lost in the middle" order. Assumes that all documents are ordered by relevance.\n\n        :param documents: List of Documents to merge.\n        :return: Documents in the "lost in the middle" order.\n        '
        if not documents:
            return []
        if len(documents) == 1:
            return documents
        if any((not doc.content_type == 'text' for doc in documents)):
            raise ValueError('Some provided documents are not textual; LostInTheMiddleRanker can process only text.')
        word_count = 0
        document_index = list(range(len(documents)))
        lost_in_the_middle_indices = [0]
        if self.word_count_threshold:
            word_count = len(documents[0].content.split())
            if word_count >= self.word_count_threshold:
                return [documents[0]]
        for doc_idx in document_index[1:]:
            insertion_index = len(lost_in_the_middle_indices) // 2 + len(lost_in_the_middle_indices) % 2
            lost_in_the_middle_indices.insert(insertion_index, doc_idx)
            if self.word_count_threshold:
                word_count += len(documents[doc_idx].content.split())
                if word_count >= self.word_count_threshold:
                    break
        return [documents[idx] for idx in lost_in_the_middle_indices]

    def predict(self, query: str, documents: List[Document], top_k: Optional[int]=None) -> List[Document]:
        if False:
            i = 10
            return i + 15
        '\n        Reranks documents based on the "lost in the middle" order.\n\n        :param query: The query to reorder documents for (ignored).\n        :param documents: List of Documents to reorder.\n        :param top_k: The number of documents to return.\n\n        :return: The reordered documents.\n        '
        top_k = top_k or self.top_k
        documents_to_reorder = documents[:top_k] if top_k else documents
        ranked_docs = self.reorder_documents(documents=documents_to_reorder)
        return ranked_docs

    def predict_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[int]=None, batch_size: Optional[int]=None) -> Union[List[Document], List[List[Document]]]:
        if False:
            while True:
                i = 10
        '\n        Reranks batch of documents based on the "lost in the middle" order.\n\n        :param queries: The queries to reorder documents for (ignored).\n        :param documents: List of Documents to reorder.\n        :param top_k: The number of documents to return.\n        :param batch_size: The number of queries to process in one batch (ignored).\n\n        :return: The reordered documents.\n        '
        if len(documents) > 0 and isinstance(documents[0], Document):
            return self.predict(query='', documents=documents, top_k=top_k)
        else:
            results = []
            for cur_docs in documents:
                assert isinstance(cur_docs, list)
                results.append(self.predict(query='', documents=cur_docs, top_k=top_k))
            return results