import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from haystack.nodes.sampler.base import BaseSampler
from haystack.schema import Document
from haystack.lazy_imports import LazyImport
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from sentence_transformers import CrossEncoder
    from haystack.modeling.utils import initialize_device_settings

class TopPSampler(BaseSampler):
    """
    Filters documents based on the cumulative probability of the similarity scores between the
    query and the documents using top p sampling.

    Top p sampling selects a subset of the most relevant data points from a larger set of data. The technique
    involves calculating the cumulative probability of the scores of each data point, and then
    selecting the top p percent of data points with the highest cumulative probability.

    In the context of TopPSampler, the `run()` method takes in a query and a set of documents,
    calculates the similarity scores between the query and the documents, and then filters
    the documents based on the cumulative probability of these scores. The TopPSampler provides a
    way to efficiently select the most relevant documents based on their similarity to a given query.

    Usage example:

    ```python
    prompt_node = PromptNode(
        "text-davinci-003",
        api_key=openai_key,
        max_length=256,
        default_prompt_template="question-answering-with-document-scores",
    )
    retriever = WebRetriever(api_key=os.environ.get("SERPERDEV_API_KEY"), mode="preprocessed_documents")
    sampler = TopPSampler(top_p=0.95)

    pipeline = WebQAPipeline(retriever=retriever, prompt_node=prompt_node, sampler=sampler)
    print(pipeline.run(query="What's the secret of the Universe?"))
    ```
    """

    def __init__(self, model_name_or_path: Union[str, Path]='cross-encoder/ms-marco-MiniLM-L-6-v2', top_p: Optional[float]=1.0, strict: Optional[bool]=False, score_field: Optional[str]='score', use_gpu: Optional[bool]=True, devices: Optional[List[Union[str, 'torch.device']]]=None):
        if False:
            while True:
                i = 10
        "\n        Initialize a TopPSampler.\n\n        :param model_name_or_path: Path to a pretrained sentence-transformers model.\n        :param top_p: Cumulative probability threshold for filtering the documents (usually between 0.9 and 0.99).\n        :param strict: If `top_p` is set to a low value and sampler returned no documents, then setting `strict` to\n        `False` ensures at least one document is returned. If `strict` is set to `True`, then no documents are returned.\n        :param score_field: The name of the field that should be used to store the scores a document's meta data.\n        :param use_gpu: Whether to use GPU (if available). If no GPUs are available, it falls back on a CPU.\n        :param devices: List of torch devices (for example, cuda:0, cpu, mps) to limit inference to specific devices.\n        "
        torch_and_transformers_import.check()
        super().__init__()
        self.top_p = top_p
        self.score_field = score_field
        self.strict = strict
        (self.devices, _) = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)
        self.cross_encoder = CrossEncoder(model_name_or_path, device=str(self.devices[0]))

    def predict(self, query: str, documents: List[Document], top_p: Optional[float]=None) -> List[Document]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of documents filtered using `top_p`, based on the similarity scores between the query and the\n        documents whose cumulative probability is less than or equal to `top_p`.\n\n        :param query: Query string.\n        :param documents: List of Documents.\n        :param top_p: Cumulative probability threshold for filtering the documents. If not provided, the top_p value\n        set during TopPSampler initialization is used.\n        :return: List of Documents sorted by (desc.) similarity with the query.\n        '
        if top_p is None:
            top_p = self.top_p if self.top_p else 1.0
        if not documents:
            return []
        query_doc_pairs = [[query, doc.content] for doc in documents]
        similarity_scores = self.cross_encoder.predict(query_doc_pairs)
        probs = np.exp(similarity_scores) / np.sum(np.exp(similarity_scores))
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        top_p_indices = np.where(cumulative_probs <= top_p)[0]
        original_indices = np.argsort(probs)[::-1][top_p_indices]
        selected_docs = [documents[i] for i in original_indices]
        if not selected_docs and (not self.strict):
            highest_prob_indices = np.argsort(probs)[::-1]
            selected_docs = [documents[highest_prob_indices[0]]]
        if self.score_field:
            for (idx, doc) in enumerate(selected_docs):
                doc.meta[self.score_field] = str(sorted_probs[idx])
        return selected_docs

    def predict_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]], top_p: Optional[float]=None, batch_size: Optional[int]=None) -> Union[List[Document], List[List[Document]]]:
        if False:
            print('Hello World!')
        '\n         - If you provide a list containing a single query...\n\n            - ... and a single list of Documents, the single list of Documents is re-ranked based on the\n              supplied query.\n            - ... and a list of lists of Documents, each list of Documents is re-ranked individually based on the\n              supplied query.\n\n\n        - If you provide a list of multiple queries, provide a list of lists of Documents. Each list of Documents\n        is re-ranked based on its corresponding query.\n        '
        if top_p is None:
            top_p = self.top_p
        if len(queries) == 1 and isinstance(documents[0], Document):
            return self.predict(queries[0], documents, top_p)
        if len(queries) == 1 and isinstance(documents[0], list):
            return [self.predict(queries[0], docs, top_p) for docs in documents]
        if len(queries) > 1 and isinstance(documents[0], list):
            return [self.predict(query, docs, top_p) for (query, docs) in zip(queries, documents)]
        raise ValueError(f"The following queries {queries} and documents {documents} were provided as input but it seems they're not valid.Check the documentation of this method for valid parameters and their types.")