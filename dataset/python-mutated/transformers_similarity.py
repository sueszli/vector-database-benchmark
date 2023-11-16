import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from haystack.preview import ComponentError, Document, component, default_to_dict
from haystack.preview.lazy_imports import LazyImport
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]==4.34.1'") as torch_and_transformers_import:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

@component
class TransformersSimilarityRanker:
    """
    Ranks documents based on query similarity.
    It uses a pre-trained cross-encoder model (from Hugging Face Hub) to embed the query and documents.

    Usage example:
    ```
    from haystack.preview import Document
    from haystack.preview.components.rankers import TransformersSimilarityRanker

    ranker = TransformersSimilarityRanker()
    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "City in Germany"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    assert len(docs) == 2
    assert docs[0].content == "Berlin"
    ```
    """

    def __init__(self, model_name_or_path: Union[str, Path]='cross-encoder/ms-marco-MiniLM-L-6-v2', device: str='cpu', token: Union[bool, str, None]=None, top_k: int=10):
        if False:
            print('Hello World!')
        '\n        Creates an instance of TransformersSimilarityRanker.\n\n        :param model_name_or_path: The name or path of a pre-trained cross-encoder model\n            from Hugging Face Hub.\n        :param device: torch device (for example, cuda:0, cpu, mps) to limit model inference to a specific device.\n        :param token: The API token used to download private models from Hugging Face.\n            If this parameter is set to `True`, then the token generated when running\n            `transformers-cli login` (stored in ~/.huggingface) will be used.\n        :param top_k: The maximum number of documents to return per query.\n        '
        torch_and_transformers_import.check()
        self.model_name_or_path = model_name_or_path
        if top_k <= 0:
            raise ValueError(f'top_k must be > 0, but got {top_k}')
        self.top_k = top_k
        self.device = device
        self.token = token
        self.model = None
        self.tokenizer = None

    def _get_telemetry_data(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Data that is sent to Posthog for usage analytics.\n        '
        return {'model': str(self.model_name_or_path)}

    def warm_up(self):
        if False:
            return 10
        '\n        Warm up the model and tokenizer used in scoring the documents.\n        '
        if self.model_name_or_path and (not self.model):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, token=self.token)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, token=self.token)

    def to_dict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Serialize this component to a dictionary.\n        '
        return default_to_dict(self, device=self.device, model_name_or_path=self.model_name_or_path, token=self.token if not isinstance(self.token, str) else None, top_k=self.top_k)

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int]=None):
        if False:
            print('Hello World!')
        '\n        Returns a list of documents ranked by their similarity to the given query\n\n        :param query: Query string.\n        :param documents: List of Documents.\n        :param top_k: The maximum number of documents to return.\n        :return: List of Documents sorted by (desc.) similarity with the query.\n        '
        if not documents:
            return {'documents': []}
        if top_k is None:
            top_k = self.top_k
        elif top_k <= 0:
            raise ValueError(f'top_k must be > 0, but got {top_k}')
        if self.model_name_or_path and (not self.model):
            raise ComponentError(f"The component {self.__class__.__name__} not warmed up. Run 'warm_up()' before calling 'run()'.")
        query_doc_pairs = [[query, doc.content] for doc in documents]
        features = self.tokenizer(query_doc_pairs, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            similarity_scores = self.model(**features).logits.squeeze()
        (_, sorted_indices) = torch.sort(similarity_scores, descending=True)
        ranked_docs = []
        for sorted_index_tensor in sorted_indices:
            i = sorted_index_tensor.item()
            documents[i].score = similarity_scores[i].item()
            ranked_docs.append(documents[i])
        return {'documents': ranked_docs[:top_k]}