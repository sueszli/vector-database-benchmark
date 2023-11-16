import logging
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm
from haystack.nodes.base import BaseComponent
from haystack.nodes.question_generator import QuestionGenerator
from haystack.schema import Document
from haystack.lazy_imports import LazyImport
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from sentence_transformers import CrossEncoder
    from haystack.modeling.utils import initialize_device_settings

class PseudoLabelGenerator(BaseComponent):
    """
    PseudoLabelGenerator is a component that creates Generative Pseudo Labeling (GPL) training data for the
    training of dense retrievers.

    GPL is an unsupervised domain adaptation method for the training of dense retrievers. It is based on question
    generation and pseudo labelling with powerful cross-encoders. To train a domain-adapted model, it needs access
    to an unlabeled target corpus, usually through DocumentStore and a Retriever to mine for negatives.

    For more details, see [GPL](https://github.com/UKPLab/gpl).

    For example:

    ```python
    document_store = ElasticsearchDocumentStore(...)
    retriever = BM25Retriever(...)
    qg = QuestionGenerator(model_name_or_path="doc2query/msmarco-t5-base-v1")
    plg = PseudoLabelGenerator(qg, retriever)
    output, output_id = psg.run(documents=document_store.get_all_documents())
    ```

    Note:

        While the NLP researchers trained the default question
        [generation](https://huggingface.co/doc2query/msmarco-t5-base-v1) and the cross
        [encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) models on
        the English language corpus, we can also use the language-specific question generation and
        cross-encoder models in the target language of our choice to apply GPL to documents in languages
        other than English.

        As of this writing, the German language question
        [generation](https://huggingface.co/ml6team/mt5-small-german-query-generation) and the cross
        [encoder](https://huggingface.co/ml6team/cross-encoder-mmarco-german-distilbert-base) models are
        already available, as well as question [generation](https://huggingface.co/doc2query/msmarco-14langs-mt5-base-v1)
        and the cross [encoder](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
        models trained on fourteen languages.


    """
    outgoing_edges: int = 1

    def __init__(self, question_producer: Union[QuestionGenerator, List[Dict[str, str]]], retriever, cross_encoder_model_name_or_path: str='cross-encoder/ms-marco-MiniLM-L-6-v2', max_questions_per_document: int=3, top_k: int=50, batch_size: int=16, progress_bar: bool=True, use_auth_token: Optional[Union[str, bool]]=None, use_gpu: bool=True, devices: Optional[List[Union[str, 'torch.device']]]=None):
        if False:
            print('Hello World!')
        '\n        Loads the cross-encoder model and prepares PseudoLabelGenerator.\n\n        :param question_producer: The question producer used to generate questions or a list of already produced\n        questions/document pairs in a Dictionary format {"question": "question text ...", "document": "document text ..."}.\n        :type question_producer: Union[QuestionGenerator, List[Dict[str, str]]]\n        :param retriever: The Retriever used to query document stores.\n        :type retriever: BaseRetriever\n        :param cross_encoder_model_name_or_path: The path to the cross encoder model, defaults to\n        `cross-encoder/ms-marco-MiniLM-L-6-v2`.\n        :type cross_encoder_model_name_or_path: str (optional)\n        :param max_questions_per_document: The max number of questions generated per document, defaults to 3.\n        :type max_questions_per_document: int\n        :param top_k: The number of answers retrieved for each question, defaults to 50.\n        :type top_k: int (optional)\n        :param batch_size: The number of documents to process at a time.\n        :type batch_size: int (optional)\n        :param progress_bar: Whether to show a progress bar, defaults to True.\n        :type progress_bar: bool (optional)\n        :param use_auth_token: The API token used to download private models from Huggingface.\n                               If this parameter is set to `True`, then the token generated when running\n                               `transformers-cli login` (stored in ~/.huggingface) will be used.\n                               Additional information can be found here\n                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained\n        :type use_auth_token: Union[str, bool] (optional)\n        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit CrossEncoder inference to specific devices.\n                        A list containing torch device objects and/or strings is supported (For example\n                        [torch.device(\'cuda:0\'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices\n                        parameter is not used and a single cpu device is used for inference.\n        '
        torch_and_transformers_import.check()
        super().__init__()
        self.question_document_pairs = None
        self.question_generator = None
        if isinstance(question_producer, QuestionGenerator):
            self.question_generator = question_producer
        elif isinstance(question_producer, list) and len(question_producer) > 0:
            example = question_producer[0]
            if isinstance(example, dict) and 'question' in example and ('document' in example):
                self.question_document_pairs = question_producer
            else:
                raise ValueError("The question_producer list must contain dictionaries with keys 'question' and 'document'.")
        else:
            raise ValueError('Provide either a QuestionGenerator or a non-empty list of questions/document pairs.')
        (self.devices, _) = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning('Multiple devices are not supported in %s inference, using the first device %s.', self.__class__.__name__, self.devices[0])
        self.retriever = retriever
        self.cross_encoder = CrossEncoder(cross_encoder_model_name_or_path, device=str(self.devices[0]), tokenizer_args={'use_auth_token': use_auth_token}, automodel_args={'use_auth_token': use_auth_token})
        self.max_questions_per_document = max_questions_per_document
        self.top_k = top_k
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    def generate_questions(self, documents: List[Document], batch_size: Optional[int]=None) -> List[Dict[str, str]]:
        if False:
            return 10
        '\n        It takes a list of documents and generates a list of question-document pairs.\n\n        :param documents: A list of documents to generate questions from.\n        :type documents: List[Document]\n        :param batch_size: The number of documents to process at a time.\n        :type batch_size: Optional[int]\n        :return: A list of question-document pairs.\n        '
        question_doc_pairs: List[Dict[str, str]] = []
        if self.question_document_pairs:
            question_doc_pairs = self.question_document_pairs
        else:
            batch_size = batch_size if batch_size else self.batch_size
            questions: List[List[str]] = self.question_generator.generate_batch([d.content for d in documents], batch_size=batch_size)
            for (idx, question_list_per_doc) in enumerate(questions):
                for q in question_list_per_doc[:self.max_questions_per_document]:
                    question_doc_pairs.append({'question': q.strip(), 'document': documents[idx].content})
        return question_doc_pairs

    def mine_negatives(self, question_doc_pairs: List[Dict[str, str]], batch_size: Optional[int]=None) -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a list of question and positive document pairs, this function returns a list of question/positive document/negative document\n        dictionaries.\n\n        :param question_doc_pairs: A list of question/positive document pairs.\n        :type question_doc_pairs: List[Dict[str, str]]\n        :param batch_size: The number of queries to run in a batch.\n        :type batch_size: int (optional)\n        :return: A list of dictionaries, where each dictionary contains the question, positive document,\n                and negative document.\n        '
        question_pos_doc_neg_doc: List[Dict[str, str]] = []
        batch_size = batch_size if batch_size else self.batch_size
        for i in tqdm(range(0, len(question_doc_pairs), batch_size), disable=not self.progress_bar, desc='Mine negatives'):
            i_end = min(i + batch_size, len(question_doc_pairs))
            queries: List[str] = [e['question'] for e in question_doc_pairs[i:i_end]]
            pos_docs: List[str] = [e['document'] for e in question_doc_pairs[i:i_end]]
            docs: List[List[Document]] = self.retriever.retrieve_batch(queries=queries, top_k=self.top_k, batch_size=batch_size)
            for (question, pos_doc, top_docs) in zip(queries, pos_docs, docs):
                random.shuffle(top_docs)
                for doc_item in top_docs:
                    neg_doc = doc_item.content
                    if neg_doc != pos_doc:
                        question_pos_doc_neg_doc.append({'question': question, 'pos_doc': pos_doc, 'neg_doc': neg_doc})
                        break
        return question_pos_doc_neg_doc

    def generate_margin_scores(self, mined_negatives: List[Dict[str, str]], batch_size: Optional[int]=None) -> List[Dict]:
        if False:
            while True:
                i = 10
        "\n        Given a list of mined negatives, this function predicts the score margin between the positive and negative document using\n        the cross-encoder.\n\n        The function returns a list of examples, where each example is a dictionary with the following keys:\n\n        * question: The question string.\n        * pos_doc: Positive document string (the document containing the answer).\n        * neg_doc: Negative document string (the document that doesn't contain the answer).\n        * score: The margin between the score for question-positive document pair and the score for question-negative document pair.\n\n        :param mined_negatives: The list of mined negatives.\n        :type mined_negatives: List[Dict[str, str]]\n        :param batch_size: The number of mined negative lists to run in a batch.\n        :type batch_size: int (optional)\n        :return: A list of dictionaries, each of which has the following keys:\n            - question: The question string\n            - pos_doc: Positive document string\n            - neg_doc: Negative document string\n            - score: The score margin\n        "
        examples: List[Dict] = []
        batch_size = batch_size if batch_size else self.batch_size
        for i in tqdm(range(0, len(mined_negatives), batch_size), disable=not self.progress_bar, desc='Score margin'):
            negatives_batch = mined_negatives[i:i + batch_size]
            pb = []
            for item in negatives_batch:
                pb.append([item['question'], item['pos_doc']])
                pb.append([item['question'], item['neg_doc']])
            scores = self.cross_encoder.predict(pb)
            for (idx, item) in enumerate(negatives_batch):
                scores_idx = idx * 2
                score_margin = scores[scores_idx] - scores[scores_idx + 1]
                examples.append({'question': item['question'], 'pos_doc': item['pos_doc'], 'neg_doc': item['neg_doc'], 'score': score_margin})
        return examples

    def generate_pseudo_labels(self, documents: List[Document], batch_size: Optional[int]=None) -> Tuple[dict, str]:
        if False:
            print('Hello World!')
        "\n        Given a list of documents, this function generates a list of question-document pairs, mines for negatives, and\n        scores a positive/negative margin with cross-encoder. The output is the training data for the\n        adaptation of dense retriever models.\n\n        :param documents: List[Document] = The list of documents to mine negatives from.\n        :type documents: List[Document]\n        :param batch_size: The number of documents to process in a batch.\n        :type batch_size: Optional[int]\n        :return: A dictionary with a single key 'gpl_labels' representing a list of dictionaries, where each\n        dictionary contains the following keys:\n            - question: The question string.\n            - pos_doc: Positive document for the given question.\n            - neg_doc: Negative document for the given question.\n            - score: The margin between the score for question-positive document pair and the score for question-negative document pair.\n        "
        batch_size = batch_size if batch_size else self.batch_size
        question_doc_pairs = self.generate_questions(documents=documents, batch_size=batch_size)
        mined_negatives = self.mine_negatives(question_doc_pairs=question_doc_pairs, batch_size=batch_size)
        pseudo_labels: List[Dict[str, str]] = self.generate_margin_scores(mined_negatives, batch_size=batch_size)
        return ({'gpl_labels': pseudo_labels}, 'output_1')

    def run(self, documents: List[Document]) -> Tuple[dict, str]:
        if False:
            for i in range(10):
                print('nop')
        return self.generate_pseudo_labels(documents=documents)

    def run_batch(self, documents: Union[List[Document], List[List[Document]]]) -> Tuple[dict, str]:
        if False:
            for i in range(10):
                print('nop')
        flat_list_of_documents = []
        for sub_list_documents in documents:
            if isinstance(sub_list_documents, Iterable):
                flat_list_of_documents += sub_list_documents
            else:
                flat_list_of_documents.append(sub_list_documents)
        return self.generate_pseudo_labels(documents=flat_list_of_documents)