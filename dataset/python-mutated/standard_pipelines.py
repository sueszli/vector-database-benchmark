import logging
from abc import ABC
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from haystack.document_stores.base import BaseDocumentStore, FilterType
from haystack.nodes.answer_generator.base import BaseGenerator
from haystack.nodes.other.docs2answers import Docs2Answers
from haystack.nodes.other.document_merger import DocumentMerger
from haystack.nodes.question_generator.question_generator import QuestionGenerator
from haystack.nodes.reader.base import BaseReader
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever.web import WebRetriever
from haystack.nodes.summarizer.base import BaseSummarizer
from haystack.nodes.translator.base import BaseTranslator
from haystack.nodes import PreProcessor, TextConverter, PromptNode, Shaper, TopPSampler
from haystack.pipelines.base import Pipeline
from haystack.schema import Document, EvaluationResult, MultiLabel, Answer
logger = logging.getLogger(__name__)

class BaseStandardPipeline(ABC):
    """
    Base class for pre-made standard Haystack pipelines.
    This class does not inherit from Pipeline.
    """
    pipeline: Pipeline
    metrics_filter: Optional[Dict[str, List[str]]] = None

    def add_node(self, component, name: str, inputs: List[str]):
        if False:
            i = 10
            return i + 15
        '\n        Add a new node to the pipeline.\n\n        :param component: The object to be called when the data is passed to the node. It can be a Haystack component\n                          (like Retriever, Reader, or Generator) or a user-defined object that implements a run()\n                          method to process incoming data from predecessor node.\n        :param name: The name for the node. It must not contain any dots.\n        :param inputs: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name\n                       of node is sufficient. For instance, a \'BM25Retriever\' node would always output a single\n                       edge with a list of documents. It can be represented as ["BM25Retriever"].\n\n                       In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output\n                       must be specified explicitly as "QueryClassifier.output_2".\n        '
        self.pipeline.add_node(component=component, name=name, inputs=inputs)

    def get_node(self, name: str):
        if False:
            while True:
                i = 10
        '\n        Get a node from the Pipeline.\n\n        :param name: The name of the node.\n        '
        component = self.pipeline.get_node(name)
        return component

    def set_node(self, name: str, component):
        if False:
            i = 10
            return i + 15
        '\n        Set the component for a node in the Pipeline.\n\n        :param name: The name of the node.\n        :param component: The component object to be set at the node.\n        '
        self.pipeline.set_node(name, component)

    def draw(self, path: Path=Path('pipeline.png')):
        if False:
            print('Hello World!')
        '\n        Create a Graphviz visualization of the pipeline.\n\n        :param path: the path to save the image.\n        '
        self.pipeline.draw(path)

    def get_nodes_by_class(self, class_type) -> List[Any]:
        if False:
            print('Hello World!')
        '\n        Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).\n        This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.\n        Example:\n        ```python\n        from haystack.document_stores.base import BaseDocumentStore\n        INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)\n        res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)\n        ```\n        :return: List of components that are an instance of the requested class\n        '
        return self.pipeline.get_nodes_by_class(class_type)

    def get_document_store(self) -> Optional[BaseDocumentStore]:
        if False:
            while True:
                i = 10
        '\n        Return the document store object used in the current pipeline.\n\n        :return: Instance of DocumentStore or None\n        '
        return self.pipeline.get_document_store()

    def get_type(self) -> str:
        if False:
            print('Hello World!')
        '\n        Return the type of the pipeline.\n\n        :return: Type of the pipeline\n        '
        return self.pipeline.get_type()

    def eval(self, labels: List[MultiLabel], params: Optional[dict]=None, sas_model_name_or_path: Optional[str]=None, sas_batch_size: int=32, sas_use_gpu: bool=True, add_isolated_node_eval: bool=False, custom_document_id_field: Optional[str]=None, context_matching_min_length: int=100, context_matching_boost_split_overlaps: bool=True, context_matching_threshold: float=65.0) -> EvaluationResult:
        if False:
            i = 10
            return i + 15
        '\n        Evaluates the pipeline by running the pipeline once per query in debug mode\n        and putting together all data that is needed for evaluation, e.g. calculating metrics.\n\n        If you want to calculate SAS (Semantic Answer Similarity) metrics, you have to specify `sas_model_name_or_path`.\n\n        You will be able to control the scope within which an answer or a document is considered correct afterwards (See `document_scope` and `answer_scope` params in `EvaluationResult.calculate_metrics()`).\n        Some of these scopes require additional information that already needs to be specified during `eval()`:\n        - `custom_document_id_field` param to select a custom document ID from document\'s meta data for ID matching (only affects \'document_id\' scopes)\n        - `context_matching_...` param to fine-tune the fuzzy matching mechanism that determines whether some text contexts match each other (only affects \'context\' scopes, default values should work most of the time)\n\n        :param labels: The labels to evaluate on\n        :param params: Params for the `retriever` and `reader`. For instance,\n                       params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}\n        :param sas_model_name_or_path: SentenceTransformers semantic textual similarity model to be used for sas value calculation,\n                                    should be path or string pointing to downloadable models.\n        :param sas_batch_size: Number of prediction label pairs to encode at once by CrossEncoder or SentenceTransformer while calculating SAS.\n        :param sas_use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.\n                            Falls back to CPU if no GPU is available.\n        :param add_isolated_node_eval: Whether to additionally evaluate the reader based on labels as input instead of output of previous node in pipeline\n        :param custom_document_id_field: Custom field name within `Document`\'s `meta` which identifies the document and is being used as criterion for matching documents to labels during evaluation.\n                                         This is especially useful if you want to match documents on other criteria (e.g. file names) than the default document ids as these could be heavily influenced by preprocessing.\n                                         If not set (default) the `Document`\'s `id` is being used as criterion for matching documents to labels.\n        :param context_matching_min_length: The minimum string length context and candidate need to have in order to be scored.\n                           Returns 0.0 otherwise.\n        :param context_matching_boost_split_overlaps: Whether to boost split overlaps (e.g. [AB] <-> [BC]) that result from different preprocessing params.\n                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries\n                                 we cut the context on the same side, recalculate the score and take the mean of both.\n                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.\n        :param context_matching_threshold: Score threshold that candidates must surpass to be included into the result list. Range: [0,100]\n        '
        output = self.pipeline.eval(labels=labels, params=params, sas_model_name_or_path=sas_model_name_or_path, sas_batch_size=sas_batch_size, sas_use_gpu=sas_use_gpu, add_isolated_node_eval=add_isolated_node_eval, custom_document_id_field=custom_document_id_field, context_matching_boost_split_overlaps=context_matching_boost_split_overlaps, context_matching_min_length=context_matching_min_length, context_matching_threshold=context_matching_threshold)
        return output

    def eval_batch(self, labels: List[MultiLabel], params: Optional[dict]=None, sas_model_name_or_path: Optional[str]=None, sas_batch_size: int=32, sas_use_gpu: bool=True, add_isolated_node_eval: bool=False, custom_document_id_field: Optional[str]=None, context_matching_min_length: int=100, context_matching_boost_split_overlaps: bool=True, context_matching_threshold: float=65.0) -> EvaluationResult:
        if False:
            for i in range(10):
                print('nop')
        '\n         Evaluates the pipeline by running the pipeline once per query in the debug mode\n         and putting together all data that is needed for evaluation, for example, calculating metrics.\n\n        To calculate SAS (Semantic Answer Similarity) metrics, specify `sas_model_name_or_path`.\n\n         You can control the scope within which an Answer or a Document is considered correct afterwards (see `document_scope` and `answer_scope` params in `EvaluationResult.calculate_metrics()`).\n         For some of these scopes, you need to add the following information during `eval()`:\n         - `custom_document_id_field` parameter to select a custom document ID from document\'s metadata for ID matching (only affects \'document_id\' scopes).\n         - `context_matching_...` parameter to fine-tune the fuzzy matching mechanism that determines whether text contexts match each other (only affects \'context\' scopes, default values should work most of the time).\n\n         :param labels: The labels to evaluate on.\n         :param params: Parameters for the `retriever` and `reader`. For instance,\n                        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}.\n         :param sas_model_name_or_path: Sentence transformers semantic textual similarity model you want to use for the SAS value calculation.\n                                     It should be a path or a string pointing to downloadable models.\n         :param sas_batch_size: Number of prediction label pairs to encode at once by cross encoder or sentence transformer while calculating SAS.\n         :param sas_use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.\n                             Falls back to CPU if no GPU is available.\n         :param add_isolated_node_eval: Whether to additionally evaluate the reader based on labels as input, instead of the output of the previous node in the pipeline.\n         :param custom_document_id_field: Custom field name within `Document`\'s `meta` which identifies the document and is used as a criterion for matching documents to labels during evaluation.\n                                          This is especially useful if you want to match documents on other criteria (for example, file names) than the default document IDs, as these could be heavily influenced by preprocessing.\n                                          If not set, the default `Document`\'s `id` is used as the criterion for matching documents to labels.\n         :param context_matching_min_length: The minimum string length context and candidate need to have to be scored.\n                            Returns 0.0 otherwise.\n         :param context_matching_boost_split_overlaps: Whether to boost split overlaps (for example, [AB] <-> [BC]) that result from different preprocessing parameters.\n                                  If we detect that the score is near a half match and the matching part of the candidate is at its boundaries,\n                                  we cut the context on the same side, recalculate the score, and take the mean of both.\n                                  Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.\n         :param context_matching_threshold: Score threshold that candidates must surpass to be included into the result list. Range: [0,100]\n        '
        output = self.pipeline.eval_batch(labels=labels, params=params, sas_model_name_or_path=sas_model_name_or_path, sas_batch_size=sas_batch_size, sas_use_gpu=sas_use_gpu, add_isolated_node_eval=add_isolated_node_eval, custom_document_id_field=custom_document_id_field, context_matching_boost_split_overlaps=context_matching_boost_split_overlaps, context_matching_min_length=context_matching_min_length, context_matching_threshold=context_matching_threshold)
        return output

    def print_eval_report(self, eval_result: EvaluationResult, n_wrong_examples: int=3, metrics_filter: Optional[Dict[str, List[str]]]=None, document_scope: Literal['document_id', 'context', 'document_id_and_context', 'document_id_or_context', 'answer', 'document_id_or_answer']='document_id_or_answer', answer_scope: Literal['any', 'context', 'document_id', 'document_id_and_context']='any', wrong_examples_fields: Optional[List[str]]=None, max_characters_per_field: int=150):
        if False:
            return 10
        '\n        Prints evaluation report containing a metrics funnel and worst queries for further analysis.\n\n        :param eval_result: The evaluation result, can be obtained by running eval().\n        :param n_wrong_examples: The number of worst queries to show.\n        :param metrics_filter: The metrics to show per node. If None all metrics will be shown.\n        :param document_scope: A criterion for deciding whether documents are relevant or not.\n            You can select between:\n            - \'document_id\': Specifies that the document ID must match. You can specify a custom document ID through `pipeline.eval()`\'s `custom_document_id_field` param.\n                    A typical use case is Document Retrieval.\n            - \'context\': Specifies that the content of the document must match. Uses fuzzy matching (see `pipeline.eval()`\'s `context_matching_...` params).\n                    A typical use case is Document-Independent Passage Retrieval.\n            - \'document_id_and_context\': A Boolean operation specifying that both `\'document_id\' AND \'context\'` must match.\n                    A typical use case is Document-Specific Passage Retrieval.\n            - \'document_id_or_context\': A Boolean operation specifying that either `\'document_id\' OR \'context\'` must match.\n                    A typical use case is Document Retrieval having sparse context labels.\n            - \'answer\': Specifies that the document contents must include the answer. The selected `answer_scope` is enforced automatically.\n                    A typical use case is Question Answering.\n            - \'document_id_or_answer\' (default): A Boolean operation specifying that either `\'document_id\' OR \'answer\'` must match.\n                    This is intended to be a proper default value in order to support both main use cases:\n                    - Document Retrieval\n                    - Question Answering\n            The default value is \'document_id_or_answer\'.\n        :param answer_scope: Specifies the scope in which a matching answer is considered correct.\n            You can select between:\n            - \'any\' (default): Any matching answer is considered correct.\n            - \'context\': The answer is only considered correct if its context matches as well.\n                    Uses fuzzy matching (see `pipeline.eval()`\'s `context_matching_...` params).\n            - \'document_id\': The answer is only considered correct if its document ID matches as well.\n                    You can specify a custom document ID through `pipeline.eval()`\'s `custom_document_id_field` param.\n            - \'document_id_and_context\': The answer is only considered correct if its document ID and its context match as well.\n            The default value is \'any\'.\n            In Question Answering, to enforce that the retrieved document is considered correct whenever the answer is correct, set `document_scope` to \'answer\' or \'document_id_or_answer\'.\n        :param wrong_examples_fields: A list of field names to include in the worst samples. By default, "answer", "context", and "document_id" are used.\n        :param max_characters_per_field: The maximum number of characters per wrong example to show (per field).\n        '
        if wrong_examples_fields is None:
            wrong_examples_fields = ['answer', 'context', 'document_id']
        if metrics_filter is None:
            metrics_filter = self.metrics_filter
        self.pipeline.print_eval_report(eval_result=eval_result, n_wrong_examples=n_wrong_examples, metrics_filter=metrics_filter, document_scope=document_scope, answer_scope=answer_scope, wrong_examples_fields=wrong_examples_fields, max_characters_per_field=max_characters_per_field)

    def run_batch(self, queries: List[str], params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        '\n        Run a batch of queries through the pipeline.\n\n        :param queries: List of query strings.\n        :param params: Parameters for the individual nodes of the pipeline. For instance,\n                       `params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}`\n        :param debug: Whether the pipeline should instruct nodes to collect debug information\n                      about their execution. By default these include the input parameters\n                      they received and the output they generated.\n                      All debug information can then be found in the dict returned\n                      by this method under the key "_debug"\n        '
        output = self.pipeline.run_batch(queries=queries, params=params, debug=debug)
        return output

class ExtractiveQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Extractive Question Answering.
    """

    def __init__(self, reader: BaseReader, retriever: BaseRetriever):
        if False:
            print('Hello World!')
        '\n        :param reader: Reader instance\n        :param retriever: Retriever instance\n        '
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])
        self.pipeline.add_node(component=reader, name='Reader', inputs=['Retriever'])
        self.metrics_filter = {'Retriever': ['recall_single_hit']}

    def run(self, query: str, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            return 10
        '\n        :param query: The search query string.\n        :param params: Params for the `retriever` and `reader`. For instance,\n                       params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}\n        :param debug: Whether the pipeline should instruct nodes to collect debug information\n                      about their execution. By default these include the input parameters\n                      they received and the output they generated.\n                      All debug information can then be found in the dict returned\n                      by this method under the key "_debug"\n        '
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output

class WebQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Generative Question Answering performed based on Documents returned from a web search engine.
    """

    def __init__(self, retriever: WebRetriever, prompt_node: PromptNode, sampler: Optional[TopPSampler]=None, shaper: Optional[Shaper]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param retriever: The WebRetriever used for retrieving documents from a web search engine.\n        :param prompt_node: The PromptNode used for generating the answer based on retrieved documents.\n        :param shaper: The Shaper used for transforming the documents and scores into a format that can be used by the PromptNode. Optional.\n        '
        if not shaper:
            shaper = Shaper(func='join_documents_and_scores', inputs={'documents': 'documents'}, outputs=['documents'])
        if not sampler and retriever.mode != 'snippets':
            sampler = TopPSampler(top_p=0.95)
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])
        if sampler:
            self.pipeline.add_node(component=sampler, name='Sampler', inputs=['Retriever'])
            self.pipeline.add_node(component=shaper, name='Shaper', inputs=['Sampler'])
        else:
            self.pipeline.add_node(component=shaper, name='Shaper', inputs=['Retriever'])
        self.pipeline.add_node(component=prompt_node, name='PromptNode', inputs=['Shaper'])
        self.metrics_filter = {'Retriever': ['recall_single_hit']}

    def run(self, query: str, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param query: The search query string.\n        :param params: Params for the `Retriever`, `Sampler`, `Shaper`, and ``PromptNode. For instance,\n                       params={"Retriever": {"top_k": 3}, "Sampler": {"top_p": 0.8}}. See the API documentation of each node for available parameters and their descriptions.\n        :param debug: Whether the pipeline should instruct nodes to collect debug information\n                      about their execution. By default, these include the input parameters\n                      they received and the output they generated.\n                      YOu can then find all debug information in the dict thia method returns\n                      under the key "_debug".\n        '
        output = self.pipeline.run(query=query, params=params, debug=debug)
        output['answers'] = [Answer(answer=output['results'][0].split('\n')[-1], type='generative')]
        return output

class DocumentSearchPipeline(BaseStandardPipeline):
    """
    Pipeline for semantic document search.
    """

    def __init__(self, retriever: BaseRetriever):
        if False:
            while True:
                i = 10
        '\n        :param retriever: Retriever instance\n        '
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])

    def run(self, query: str, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            print('Hello World!')
        '\n        :param query: the query string.\n        :param params: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}\n        :param debug: Whether the pipeline should instruct nodes to collect debug information\n              about their execution. By default these include the input parameters\n              they received and the output they generated.\n              All debug information can then be found in the dict returned\n              by this method under the key "_debug"\n        '
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output

class GenerativeQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Generative Question Answering.
    """

    def __init__(self, generator: BaseGenerator, retriever: BaseRetriever):
        if False:
            i = 10
            return i + 15
        '\n        :param generator: Generator instance\n        :param retriever: Retriever instance\n        '
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])
        self.pipeline.add_node(component=generator, name='Generator', inputs=['Retriever'])

    def run(self, query: str, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param query: the query string.\n        :param params: params for the `retriever` and `generator`. For instance,\n                       params={"Retriever": {"top_k": 10}, "Generator": {"top_k": 5}}\n        :param debug: Whether the pipeline should instruct nodes to collect debug information\n              about their execution. By default these include the input parameters\n              they received and the output they generated.\n              All debug information can then be found in the dict returned\n              by this method under the key "_debug"\n        '
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output

class SearchSummarizationPipeline(BaseStandardPipeline):
    """
    Pipeline that retrieves documents for a query and then summarizes those documents.
    """

    def __init__(self, summarizer: BaseSummarizer, retriever: BaseRetriever, generate_single_summary: bool=False, return_in_answer_format: bool=False):
        if False:
            while True:
                i = 10
        '\n        :param summarizer: Summarizer instance\n        :param retriever: Retriever instance\n        :param generate_single_summary: Whether to generate a single summary for all documents or one summary per document.\n        :param return_in_answer_format: Whether the results should be returned as documents (False) or in the answer\n                                        format used in other QA pipelines (True). With the latter, you can use this\n                                        pipeline as a "drop-in replacement" for other QA pipelines.\n        '
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])
        if generate_single_summary is True:
            document_merger = DocumentMerger()
            self.pipeline.add_node(component=document_merger, name='Document Merger', inputs=['Retriever'])
            self.pipeline.add_node(component=summarizer, name='Summarizer', inputs=['Document Merger'])
        else:
            self.pipeline.add_node(component=summarizer, name='Summarizer', inputs=['Retriever'])
        self.return_in_answer_format = return_in_answer_format

    def run(self, query: str, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param query: the query string.\n        :param params: params for the `retriever` and `summarizer`. For instance,\n                       params={"Retriever": {"top_k": 10}, "Summarizer": {"generate_single_summary": True}}\n        :param debug: Whether the pipeline should instruct nodes to collect debug information\n              about their execution. By default these include the input parameters\n              they received and the output they generated.\n              All debug information can then be found in the dict returned\n              by this method under the key "_debug"\n        '
        output = self.pipeline.run(query=query, params=params, debug=debug)
        if self.return_in_answer_format:
            results: Dict = {'query': query, 'answers': []}
            docs = deepcopy(output['documents'])
            for doc in docs:
                cur_answer = {'query': query, 'answer': doc.meta.pop('summary'), 'document_id': doc.id, 'context': doc.content, 'score': None, 'offset_start': None, 'offset_end': None, 'meta': doc.meta}
                results['answers'].append(cur_answer)
        else:
            results = output
        return results

    def run_batch(self, queries: List[str], params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            while True:
                i = 10
        '\n        Run a batch of queries through the pipeline.\n\n        :param queries: List of query strings.\n        :param params: Parameters for the individual nodes of the pipeline. For instance,\n                       `params={"Retriever": {"top_k": 10}, "Summarizer": {"generate_single_summary": True}}`\n        :param debug: Whether the pipeline should instruct nodes to collect debug information\n                      about their execution. By default these include the input parameters\n                      they received and the output they generated.\n                      All debug information can then be found in the dict returned\n                      by this method under the key "_debug"\n        '
        output = self.pipeline.run_batch(queries=queries, params=params, debug=debug)
        if self.return_in_answer_format:
            results: Dict = {'queries': queries, 'answers': []}
            docs = deepcopy(output['documents'])
            for (query, cur_docs) in zip(queries, docs):
                cur_answers = []
                for doc in cur_docs:
                    cur_answer = {'query': query, 'answer': doc.meta.pop('summary'), 'document_id': doc.id, 'context': doc.content, 'score': None, 'offset_start': None, 'offset_end': None, 'meta': doc.meta}
                    cur_answers.append(cur_answer)
                results['answers'].append(cur_answers)
        else:
            results = output
        return results

class FAQPipeline(BaseStandardPipeline):
    """
    Pipeline for finding similar FAQs using semantic document search.
    """

    def __init__(self, retriever: BaseRetriever):
        if False:
            return 10
        '\n        :param retriever: Retriever instance\n        '
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])
        self.pipeline.add_node(component=Docs2Answers(), name='Docs2Answers', inputs=['Retriever'])

    def run(self, query: str, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param query: the query string.\n        :param params: params for the `retriever`. For instance, params={"Retriever": {"top_k": 10}}\n        :param debug: Whether the pipeline should instruct nodes to collect debug information\n              about their execution. By default these include the input parameters\n              they received and the output they generated.\n              All debug information can then be found in the dict returned\n              by this method under the key "_debug"\n        '
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output

class TranslationWrapperPipeline(BaseStandardPipeline):
    """
    Takes an existing search pipeline and adds one "input translation node" after the Query and one
    "output translation" node just before returning the results
    """

    def __init__(self, input_translator: BaseTranslator, output_translator: BaseTranslator, pipeline: BaseStandardPipeline):
        if False:
            print('Hello World!')
        '\n        Wrap a given `pipeline` with the `input_translator` and `output_translator`.\n\n        :param input_translator: A Translator node that shall translate the input query from language A to B\n        :param output_translator: A Translator node that shall translate the pipeline results from language B to A\n        :param pipeline: The pipeline object (e.g. ExtractiveQAPipeline) you want to "wrap".\n                         Note that pipelines with split or merge nodes are currently not supported.\n        '
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=input_translator, name='InputTranslator', inputs=['Query'])
        if isinstance(pipeline, QuestionAnswerGenerationPipeline):
            setattr(output_translator, 'run', output_translator.run_batch)
        if hasattr(pipeline, 'pipeline'):
            graph = pipeline.pipeline.graph
        else:
            graph = pipeline.graph
        previous_node_name = ['InputTranslator']
        for node in graph.nodes:
            if node == 'Query':
                continue
            if graph.nodes[node]['inputs'] and len(graph.nodes[node]['inputs']) > 1:
                raise AttributeError('Split and merge nodes are not supported currently')
            self.pipeline.add_node(name=node, component=graph.nodes[node]['component'], inputs=previous_node_name)
            previous_node_name = [node]
        self.pipeline.add_node(component=output_translator, name='OutputTranslator', inputs=previous_node_name)

    def run(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        output = self.pipeline.run(**kwargs)
        return output

    def run_batch(self, **kwargs):
        if False:
            i = 10
            return i + 15
        output = self.pipeline.run_batch(**kwargs)
        return output

class QuestionGenerationPipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes documents as input and generates
    questions that it thinks can be answered by the documents.
    """

    def __init__(self, question_generator: QuestionGenerator):
        if False:
            print('Hello World!')
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=question_generator, name='QuestionGenerator', inputs=['Query'])

    def run(self, documents, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            print('Hello World!')
        output = self.pipeline.run(documents=documents, params=params, debug=debug)
        return output

    def run_batch(self, documents: Union[List[Document], List[List[Document]]], params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        output = self.pipeline.run_batch(documents=documents, params=params, debug=debug)
        return output

class RetrieverQuestionGenerationPipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes a query as input, performs retrieval, and then generates
    questions that it thinks can be answered by the retrieved documents.
    """

    def __init__(self, retriever: BaseRetriever, question_generator: QuestionGenerator):
        if False:
            for i in range(10):
                print('nop')
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])
        self.pipeline.add_node(component=question_generator, name='QuestionGenerator', inputs=['Retriever'])

    def run(self, query: str, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            return 10
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output

class QuestionAnswerGenerationPipeline(BaseStandardPipeline):
    """
    This is a pipeline which takes a document as input, generates questions that the model thinks can be answered by
    this document, and then performs question answering of this questions using that single document.
    """

    def __init__(self, question_generator: QuestionGenerator, reader: BaseReader):
        if False:
            i = 10
            return i + 15
        setattr(question_generator, 'run', self.formatting_wrapper(question_generator.run))
        setattr(reader, 'run', reader.run_batch)
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=question_generator, name='QuestionGenerator', inputs=['Query'])
        self.pipeline.add_node(component=reader, name='Reader', inputs=['QuestionGenerator'])

    def formatting_wrapper(self, fn):
        if False:
            while True:
                i = 10

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            (output, output_stream) = fn(*args, **kwargs)
            questions = []
            documents = []
            for (generated_questions, doc) in zip(output['generated_questions'], output['documents']):
                questions.extend(generated_questions['questions'])
                documents.extend([[doc]] * len(generated_questions['questions']))
            kwargs['queries'] = questions
            kwargs['documents'] = documents
            return (kwargs, output_stream)
        return wrapper

    def run(self, documents: List[Document], params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            return 10
        output = self.pipeline.run(documents=documents, params=params, debug=debug)
        return output

class MostSimilarDocumentsPipeline(BaseStandardPipeline):

    def __init__(self, document_store: BaseDocumentStore):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize a Pipeline for finding the most similar documents to a given document.\n        This pipeline can be helpful if you already show a relevant document to your end users and they want to search for just similar ones.\n\n        :param document_store: Document Store instance with already stored embeddings.\n        '
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=document_store, name='DocumentStore', inputs=['Query'])
        self.document_store = document_store

    def run(self, document_ids: List[str], filters: Optional[FilterType]=None, top_k: int=5, index: Optional[str]=None):
        if False:
            print('Hello World!')
        "\n        :param document_ids: document ids\n        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain conditions\n        :param top_k: How many documents id to return against single document\n        :param index: Optionally specify the name of index to query the document from. If None, the DocumentStore's default index (self.index) will be used.\n        "
        self.document_store.return_embedding = True
        documents = self.document_store.get_documents_by_id(ids=document_ids, index=index)
        query_embs = [doc.embedding for doc in documents]
        similar_documents = self.document_store.query_by_embedding_batch(query_embs=query_embs, filters=filters, return_embedding=False, top_k=top_k, index=index)
        self.document_store.return_embedding = False
        return similar_documents

    def run_batch(self, document_ids: List[str], filters: Optional[FilterType]=None, top_k: int=5, index: Optional[str]=None):
        if False:
            while True:
                i = 10
        "\n        :param document_ids: document ids\n        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain conditions\n        :param top_k: How many documents id to return against single document\n        :param index: Optionally specify the name of index to query the document from. If None, the DocumentStore's default index (self.index) will be used.\n        "
        return self.run(document_ids=document_ids, filters=filters, top_k=top_k, index=index)

class TextIndexingPipeline(BaseStandardPipeline):

    def __init__(self, document_store: BaseDocumentStore, text_converter: Optional[TextConverter]=None, preprocessor: Optional[PreProcessor]=None):
        if False:
            i = 10
            return i + 15
        '\n        Initialize a basic Pipeline that converts text files into Documents and indexes them into a DocumentStore.\n\n        :param document_store: The DocumentStore to index the Documents into.\n        :param text_converter: A TextConverter object to be used in this pipeline for converting the text files into Documents.\n        :param preprocessor: A PreProcessor object to be used in this pipeline for preprocessing Documents.\n        '
        self.pipeline = Pipeline()
        self.document_store = document_store
        self.text_converter = text_converter or TextConverter()
        self.preprocessor = preprocessor or PreProcessor()
        self.pipeline.add_node(component=self.text_converter, name='TextConverter', inputs=['File'])
        self.pipeline.add_node(component=self.preprocessor, name='PreProcessor', inputs=['TextConverter'])
        self.pipeline.add_node(component=self.document_store, name='DocumentStore', inputs=['PreProcessor'])

    def run(self, file_path):
        if False:
            i = 10
            return i + 15
        return self.pipeline.run(file_paths=[file_path])

    def run_batch(self, file_paths):
        if False:
            return 10
        return self.pipeline.run_batch(file_paths=file_paths)