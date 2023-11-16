import logging
from typing import List, Optional
from haystack.preview import ComponentError, Document, component
from haystack.preview.lazy_imports import LazyImport
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install torch>=1.13'") as torch_import:
    import torch

@component
class TopPSampler:
    """
    Filters documents using top-p (nucleus) sampling based on their similarity scores' cumulative probability.

    Usage example:

    ```python
    from haystack.preview import Document
    from haystack.preview.components.samplers import TopPSampler

    sampler = TopPSampler(top_p=0.95)
    docs = [
        Document(text="Berlin", metadata={"similarity_score": -10.6}),
        Document(text="Belgrade", metadata={"similarity_score": -8.9}),
        Document(text="Sarajevo", metadata={"similarity_score": -4.6}),
    ]
    output = sampler.run(documents=docs)
    docs = output["documents"]
    assert len(docs) == 1
    assert docs[0].content == "Sarajevo"
    ```
    """

    def __init__(self, top_p: float=1.0, score_field: Optional[str]=None):
        if False:
            print('Hello World!')
        "\n        Creates an instance of TopPSampler.\n\n        :param top_p: Cumulative probability threshold (usually between 0.9 and 0.99).\n        :param score_field: Field name in a document's metadata containing the scores. Defaults to the Document score\n        if not provided.\n        "
        torch_import.check()
        self.top_p = top_p
        self.score_field = score_field

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], top_p: Optional[float]=None):
        if False:
            return 10
        '\n        Filter documents based on their similarity scores using top-p sampling.\n\n        :param documents: List of Documents to filter.\n        :param top_p: Cumulative probability threshold. Defaults to the value set during initialization or 1.0\n        if not set.\n        :return: List of filtered Documents.\n        '
        if not documents:
            return {'documents': []}
        top_p = top_p or self.top_p or 1.0
        if not 0 <= top_p <= 1:
            raise ComponentError(f'top_p must be between 0 and 1. Got {top_p}.')
        similarity_scores = torch.tensor(self._collect_scores(documents), dtype=torch.float32)
        probs = torch.nn.functional.softmax(similarity_scores, dim=-1)
        (sorted_probs, sorted_indices) = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        close_to_top_p = torch.isclose(cumulative_probs, torch.tensor(top_p, device=cumulative_probs.device), atol=1e-06)
        condition = (cumulative_probs <= top_p) | close_to_top_p
        top_p_indices = torch.where(torch.BoolTensor(condition))[0]
        original_indices = sorted_indices[top_p_indices]
        selected_docs = [documents[i.item()] for i in original_indices]
        if not selected_docs:
            logger.warning('Top-p sampling with p=%s resulted in no documents being selected. Returning the document with the highest similarity score.', top_p)
            highest_prob_indices = torch.argsort(probs, descending=True)
            selected_docs = [documents[int(highest_prob_indices[0].item())]]
        return {'documents': selected_docs}

    def _collect_scores(self, documents: List[Document]) -> List[float]:
        if False:
            while True:
                i = 10
        "\n        Collect the scores from the documents' metadata.\n        :param documents: List of Documents.\n        :return: List of scores.\n        "
        if self.score_field:
            missing_scores_docs = [d for d in documents if self.score_field not in d.meta]
            if missing_scores_docs:
                missing_scores_docs_ids = [d.id for d in missing_scores_docs if d.id]
                raise ComponentError(f"Score field '{self.score_field}' not found in metadata of documents with IDs: {missing_scores_docs_ids}.Make sure that all documents have a score field '{self.score_field}' in their metadata.")
            return [d.meta[self.score_field] for d in documents]
        else:
            missing_scores_docs = [d for d in documents if d.score is None]
            if missing_scores_docs:
                missing_scores_docs_ids = [d.id for d in missing_scores_docs if d.id]
                raise ComponentError(f"Ensure all documents have a valid score value. These docs  {missing_scores_docs_ids} don't.")
            return [d.score for d in documents]