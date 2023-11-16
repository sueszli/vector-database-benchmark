from typing import List, Optional, Union, Tuple, Iterator, Any
import logging
from pathlib import Path
from tqdm import tqdm
from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker
from haystack.lazy_imports import LazyImport
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from torch.nn import DataParallel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from haystack.modeling.utils import initialize_device_settings

class SentenceTransformersRanker(BaseRanker):
    """
    Sentence Transformer based pre-trained Cross-Encoder model for Document Re-ranking (https://huggingface.co/cross-encoder).
    Re-Ranking can be used on top of a retriever to boost the performance for document search.
    This is particularly useful if the retriever has a high recall but is bad in sorting the documents by relevance.

    SentenceTransformerRanker handles Cross-Encoder models
        - use a single logit as similarity score e.g.  cross-encoder/ms-marco-MiniLM-L-12-v2
        - use two output logits (no_answer, has_answer) e.g. deepset/gbert-base-germandpr-reranking
    https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#usage-with-transformers

    With a SentenceTransformersRanker, you can:
     - directly get predictions via predict()

    Usage example:

    ```python
    retriever = BM25Retriever(document_store=document_store)
    ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")
    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])
    ```
    """

    def __init__(self, model_name_or_path: Union[str, Path], model_version: Optional[str]=None, top_k: int=10, use_gpu: bool=True, devices: Optional[List[Union[str, 'torch.device']]]=None, batch_size: int=16, scale_score: bool=True, progress_bar: bool=True, use_auth_token: Optional[Union[str, bool]]=None, embed_meta_fields: Optional[List[str]]=None):
        if False:
            return 10
        '\n        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.\n        \'cross-encoder/ms-marco-MiniLM-L-12-v2\'.\n        See https://huggingface.co/cross-encoder for full list of available models\n        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.\n        :param top_k: The maximum number of documents to return\n        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.\n        :param batch_size: Number of documents to process at a time.\n        :param scale_score: The raw predictions will be transformed using a Sigmoid activation function in case the model\n                            only predicts a single label. For multi-label predictions, no scaling is applied. Set this\n                            to False if you do not want any scaling of the raw predictions.\n        :param progress_bar: Whether to show a progress bar while processing the documents.\n        :param use_auth_token: The API token used to download private models from Huggingface.\n                               If this parameter is set to `True`, then the token generated when running\n                               `transformers-cli login` (stored in ~/.huggingface) will be used.\n                               Additional information can be found here\n                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained\n        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.\n                        A list containing torch device objects and/or strings is supported (For example\n                        [torch.device(\'cuda:0\'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices\n                        parameter is not used and a single cpu device is used for inference.\n        :param embed_meta_fields: Concatenate the provided meta fields and into the text passage that is then used in\n            reranking. The original documents are returned so the concatenated metadata is not included in the returned documents.\n        '
        torch_and_transformers_import.check()
        super().__init__()
        self.top_k = top_k
        (self.devices, _) = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)
        self.progress_bar = progress_bar
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, revision=model_version, use_auth_token=use_auth_token)
        self.transformer_model.to(str(self.devices[0]))
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path, revision=model_version, use_auth_token=use_auth_token)
        self.transformer_model.eval()
        num_labels = self.transformer_model.num_labels
        self.activation_function: torch.nn.Module
        if num_labels == 1 and scale_score:
            self.activation_function = torch.nn.Sigmoid()
        else:
            self.activation_function = torch.nn.Identity()
        if len(self.devices) > 1:
            self.model = DataParallel(self.transformer_model, device_ids=self.devices)
        self.batch_size = batch_size
        self.embed_meta_fields = embed_meta_fields

    def predict(self, query: str, documents: List[Document], top_k: Optional[int]=None) -> List[Document]:
        if False:
            return 10
        '\n        Use loaded ranker model to re-rank the supplied list of Document.\n\n        Returns list of Document sorted by (desc.) similarity with the query.\n\n        :param query: Query string\n        :param documents: List of Document to be re-ranked\n        :param top_k: The maximum number of documents to return\n        :return: List of Document\n        '
        if top_k is None:
            top_k = self.top_k
        docs_with_meta_fields = self._add_meta_fields_to_docs(documents=documents, embed_meta_fields=self.embed_meta_fields)
        docs = [doc.content for doc in docs_with_meta_fields]
        features = self.transformer_tokenizer([query for _ in documents], docs, padding=True, truncation=True, return_tensors='pt').to(self.devices[0])
        with torch.inference_mode():
            similarity_scores = self.transformer_model(**features).logits
        logits_dim = similarity_scores.shape[1]
        sorted_scores_and_documents = sorted(zip(similarity_scores, documents), key=lambda similarity_document_tuple: similarity_document_tuple[0][-1] if logits_dim >= 2 else similarity_document_tuple[0], reverse=True)
        sorted_documents = self._add_scores_to_documents(sorted_scores_and_documents[:top_k], logits_dim)
        return sorted_documents

    def _add_scores_to_documents(self, sorted_scores_and_documents: List[Tuple[Any, Document]], logits_dim: int) -> List[Document]:
        if False:
            while True:
                i = 10
        '\n        Normalize and add scores to retrieved result documents.\n\n        :param sorted_scores_and_documents: List of score, Document Tuples.\n        :param logits_dim: Dimensionality of the returned scores.\n        '
        sorted_documents = []
        for (raw_score, doc) in sorted_scores_and_documents:
            if logits_dim >= 2:
                score = self.activation_function(raw_score)[-1]
            else:
                score = self.activation_function(raw_score)[0]
            doc.score = score.detach().cpu().numpy().tolist()
            sorted_documents.append(doc)
        return sorted_documents

    def predict_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[int]=None, batch_size: Optional[int]=None) -> Union[List[Document], List[List[Document]]]:
        if False:
            print('Hello World!')
        '\n        Use loaded ranker model to re-rank the supplied lists of Documents.\n\n        Returns lists of Documents sorted by (desc.) similarity with the corresponding queries.\n\n\n        - If you provide a list containing a single query...\n\n            - ... and a single list of Documents, the single list of Documents will be re-ranked based on the\n              supplied query.\n            - ... and a list of lists of Documents, each list of Documents will be re-ranked individually based on the\n              supplied query.\n\n\n        - If you provide a list of multiple queries...\n\n            - ... you need to provide a list of lists of Documents. Each list of Documents will be re-ranked based on\n              its corresponding query.\n\n        :param queries: Single query string or list of queries\n        :param documents: Single list of Documents or list of lists of Documents to be reranked.\n        :param top_k: The maximum number of documents to return per Document list.\n        :param batch_size: Number of Documents to process at a time.\n        '
        if top_k is None:
            top_k = self.top_k
        if batch_size is None:
            batch_size = self.batch_size
        (number_of_docs, all_queries, all_docs, single_list_of_docs) = self._preprocess_batch_queries_and_docs(queries=queries, documents=documents)
        all_docs_with_meta_fields = self._add_meta_fields_to_docs(documents=all_docs, embed_meta_fields=self.embed_meta_fields)
        batches = self._get_batches(all_queries=all_queries, all_docs=all_docs_with_meta_fields, batch_size=batch_size)
        pb = tqdm(total=len(all_docs_with_meta_fields), disable=not self.progress_bar, desc='Ranking')
        preds = []
        for (cur_queries, cur_docs) in batches:
            features = self.transformer_tokenizer(cur_queries, [doc.content for doc in cur_docs], padding=True, truncation=True, return_tensors='pt').to(self.devices[0])
            with torch.inference_mode():
                similarity_scores = self.transformer_model(**features).logits
                preds.extend(similarity_scores)
            pb.update(len(cur_docs))
        pb.close()
        logits_dim = similarity_scores.shape[1]
        if single_list_of_docs:
            sorted_scores_and_documents = sorted(zip(preds, documents), key=lambda similarity_document_tuple: similarity_document_tuple[0][-1] if logits_dim >= 2 else similarity_document_tuple[0], reverse=True)
            sorted_documents = [(score, doc) for (score, doc) in sorted_scores_and_documents if isinstance(doc, Document)]
            sorted_documents_with_scores = self._add_scores_to_documents(sorted_documents[:top_k], logits_dim)
            return sorted_documents_with_scores
        else:
            grouped_predictions = []
            left_idx = 0
            for number in number_of_docs:
                right_idx = left_idx + number
                grouped_predictions.append(preds[left_idx:right_idx])
                left_idx = right_idx
            result = []
            for (pred_group, doc_group) in zip(grouped_predictions, documents):
                sorted_scores_and_documents = sorted(zip(pred_group, doc_group), key=lambda similarity_document_tuple: similarity_document_tuple[0][-1] if logits_dim >= 2 else similarity_document_tuple[0], reverse=True)
                sorted_documents = [(score, doc) for (score, doc) in sorted_scores_and_documents if isinstance(doc, Document)]
                sorted_documents_with_scores = self._add_scores_to_documents(sorted_documents[:top_k], logits_dim)
                result.append(sorted_documents_with_scores)
            return result

    def _preprocess_batch_queries_and_docs(self, queries: List[str], documents: Union[List[Document], List[List[Document]]]) -> Tuple[List[int], List[str], List[Document], bool]:
        if False:
            print('Hello World!')
        number_of_docs = []
        all_queries = []
        all_docs: List[Document] = []
        single_list_of_docs = False
        if len(documents) > 0 and isinstance(documents[0], Document):
            if len(queries) != 1:
                raise HaystackError('Number of queries must be 1 if a single list of Documents is provided.')
            query = queries[0]
            number_of_docs = [len(documents)]
            all_queries = [query] * len(documents)
            all_docs = documents
            single_list_of_docs = True
        if len(documents) > 0 and isinstance(documents[0], list):
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError('Number of queries must be equal to number of provided Document lists.')
            for (query, cur_docs) in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f'cur_docs was of type {type(cur_docs)}, but expected a list of Documents.')
                number_of_docs.append(len(cur_docs))
                all_queries.extend([query] * len(cur_docs))
                all_docs.extend(cur_docs)
        return (number_of_docs, all_queries, all_docs, single_list_of_docs)

    @staticmethod
    def _get_batches(all_queries: List[str], all_docs: List[Document], batch_size: Optional[int]) -> Iterator[Tuple[List[str], List[Document]]]:
        if False:
            i = 10
            return i + 15
        if batch_size is None:
            yield (all_queries, all_docs)
            return
        else:
            for index in range(0, len(all_queries), batch_size):
                yield (all_queries[index:index + batch_size], all_docs[index:index + batch_size])