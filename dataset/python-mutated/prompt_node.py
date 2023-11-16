from collections import defaultdict
import copy
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from haystack.nodes.base import BaseComponent
from haystack.schema import Document, MultiLabel
from haystack.telemetry import send_event
from haystack.nodes.prompt.prompt_model import PromptModel
from haystack.nodes.prompt.prompt_template import PromptTemplate
from haystack.lazy_imports import LazyImport
with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_import:
    import torch
logger = logging.getLogger(__name__)

class PromptNode(BaseComponent):
    """
    The PromptNode class is the central abstraction in Haystack's large language model (LLM) support. PromptNode
    supports multiple NLP tasks out of the box. You can use it to perform tasks such as
    summarization, question answering, question generation, and more, using a single, unified model within the Haystack
    framework.

    One of the benefits of PromptNode is that you can use it to define and add additional prompt templates
     the model supports. Defining additional prompt templates makes it possible to extend the model's capabilities
    and use it for a broader range of NLP tasks in Haystack. Prompt engineers define templates
    for each NLP task and register them with PromptNode. The burden of defining templates for each task rests on
    the prompt engineers, not the users.

    Using an instance of the PromptModel class, you can create multiple PromptNodes that share the same model, saving
    the memory and time required to load the model multiple times.

    PromptNode also supports multiple model invocation layers:
    - Hugging Face transformers (all text2text-generation models)
    - OpenAI InstructGPT models
    - Azure OpenAI InstructGPT models

    But you're not limited to the models listed above, as you can register
    additional custom model invocation layers.

    We recommend using LLMs fine-tuned on a collection of datasets phrased as instructions, otherwise we find that the
    LLM does not "follow" prompt instructions well. The list of instruction-following models increases every month,
    and the current list includes: Flan, OpenAI InstructGPT, opt-iml, bloomz, and mt0 models.

    For more details, see [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node).
    """
    outgoing_edges: int = 1

    def __init__(self, model_name_or_path: Union[str, PromptModel]='google/flan-t5-base', default_prompt_template: Optional[Union[str, PromptTemplate]]=None, output_variable: Optional[str]=None, max_length: Optional[int]=100, api_key: Optional[str]=None, use_auth_token: Optional[Union[str, bool]]=None, use_gpu: Optional[bool]=None, devices: Optional[List[Union[str, 'torch.device']]]=None, stop_words: Optional[List[str]]=None, top_k: int=1, debug: Optional[bool]=False, model_kwargs: Optional[Dict]=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates a PromptNode instance.\n\n        :param model_name_or_path: The name of the model to use or an instance of the PromptModel.\n        :param default_prompt_template: The default prompt template to use for the model.\n        :param output_variable: The name of the output variable in which you want to store the inference results.\n            If not set, PromptNode uses PromptTemplate's output_variable. If PromptTemplate's output_variable is not set, the default name is `results`.\n        :param max_length: The maximum number of tokens the generated text output can have.\n        :param api_key: The API key to use for the model.\n        :param use_auth_token: The authentication token to use for the model.\n        :param use_gpu: Whether to use GPU or not.\n        :param devices: The devices to use for the model.\n        :param top_k: The number of independently generated texts to return per prompt. For example, if you set top_k=3, the model will generate three answers to the query.\n        :param stop_words: Stops text generation if any of the stop words is generated.\n        :param model_kwargs: Additional keyword arguments passed when loading the model specified in `model_name_or_path`.\n        :param debug: Whether to include the used prompts as debug information in the output under the key _debug.\n\n        Note that Azure OpenAI InstructGPT models require two additional parameters: azure_base_url (the URL for the\n        Azure OpenAI API endpoint, usually in the form `https://<your-endpoint>.openai.azure.com') and\n        azure_deployment_name (the name of the Azure OpenAI API deployment).\n        You should specify these parameters in the `model_kwargs` dictionary.\n\n        "
        send_event(event_name='PromptNode', event_properties={'llm.model_name_or_path': model_name_or_path, 'llm.default_prompt_template': default_prompt_template})
        super().__init__()
        self._default_template = None
        self.default_prompt_template = default_prompt_template
        self.output_variable: Optional[str] = output_variable
        self.model_name_or_path: Union[str, PromptModel] = model_name_or_path
        self.prompt_model: PromptModel
        self.stop_words: Optional[List[str]] = stop_words
        self.top_k: int = top_k
        self.debug = debug
        if isinstance(model_name_or_path, str):
            self.prompt_model = PromptModel(model_name_or_path=model_name_or_path, max_length=max_length, api_key=api_key, use_auth_token=use_auth_token, use_gpu=use_gpu, devices=devices, model_kwargs=model_kwargs)
        elif isinstance(model_name_or_path, PromptModel):
            self.prompt_model = model_name_or_path
        else:
            raise ValueError('model_name_or_path must be either a string or a PromptModel object')

    def __call__(self, *args, **kwargs) -> List[Any]:
        if False:
            return 10
        '\n        This method is invoked when the component is called directly, for example:\n        ```python\n            PromptNode pn = ...\n            sa = pn.set_default_prompt_template("sentiment-analysis")\n            sa(documents=[Document("I am in love and I feel great!")])\n        ```\n        '
        if 'prompt_template' in kwargs:
            prompt_template = kwargs['prompt_template']
            kwargs.pop('prompt_template')
            return self.prompt(prompt_template, *args, **kwargs)
        else:
            return self.prompt(self.default_prompt_template, *args, **kwargs)

    def prompt(self, prompt_template: Optional[Union[str, PromptTemplate]], *args, **kwargs) -> List[Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Prompts the model and represents the central API for the PromptNode. It takes a prompt template,\n        a list of non-keyword and keyword arguments, and returns a list of strings - the responses from the underlying model.\n\n        If you specify the optional prompt_template parameter, it takes precedence over the default PromptTemplate for this PromptNode.\n\n        :param prompt_template: The name or object of the optional PromptTemplate to use.\n        :return: A list of strings as model responses.\n        '
        results = []
        prompt_collector: List[Union[str, List[Dict[str, str]]]] = kwargs.pop('prompt_collector', [])
        kwargs = {**self._prepare_model_kwargs(), **kwargs}
        template_to_fill = self.get_prompt_template(prompt_template)
        if template_to_fill:
            for prompt in template_to_fill.fill(*args, **kwargs):
                kwargs_copy = template_to_fill.remove_template_params(copy.copy(kwargs))
                prompt = self.prompt_model._ensure_token_limit(prompt)
                prompt_collector.append(prompt)
                logger.debug('Prompt being sent to LLM with prompt %s and kwargs %s', prompt, kwargs_copy)
                output = self.prompt_model.invoke(prompt, **kwargs_copy)
                results.extend(output)
            kwargs['prompts'] = prompt_collector
            results = template_to_fill.post_process(results, **kwargs)
        else:
            for prompt in list(args):
                kwargs_copy = copy.copy(kwargs)
                prompt = self.prompt_model._ensure_token_limit(prompt)
                prompt_collector.append(prompt)
                logger.debug('Prompt being sent to LLM with prompt %s and kwargs %s ', prompt, kwargs_copy)
                output = self.prompt_model.invoke(prompt, **kwargs_copy)
                results.extend(output)
        return results

    @property
    def default_prompt_template(self):
        if False:
            return 10
        return self._default_template

    @default_prompt_template.setter
    def default_prompt_template(self, prompt_template: Union[str, PromptTemplate, None]):
        if False:
            print('Hello World!')
        '\n        Sets the default prompt template for the node.\n        :param prompt_template: The prompt template to be set as default.\n        :return: The current PromptNode object.\n        '
        self._default_template = self.get_prompt_template(prompt_template)

    def get_prompt_template(self, prompt_template: Union[str, PromptTemplate, None]=None) -> Optional[PromptTemplate]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Resolves a prompt template.\n\n        :param prompt_template: The prompt template to be resolved. You can choose between the following types:\n            - None: Returns the default prompt template.\n            - PromptTemplate: Returns the given prompt template object.\n            - str: Parses the string depending on its content:\n                - prompt template name: Returns the prompt template registered with the given name.\n                - prompt template yaml: Returns a prompt template specified by the given YAML.\n                - prompt text: Returns a copy of the default prompt template with the given prompt text.\n\n            :return: The prompt template object.\n        '
        prompt_template = prompt_template or self._default_template
        if prompt_template is None:
            return None
        if isinstance(prompt_template, PromptTemplate):
            return prompt_template
        output_parser = None
        if self.default_prompt_template:
            output_parser = self.default_prompt_template.output_parser
        return PromptTemplate(prompt_template, output_parser=output_parser)

    def prompt_template_params(self, prompt_template: str) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Returns the list of parameters for a prompt template.\n        :param prompt_template: The name of the prompt template.\n        :return: The list of parameters for the prompt template.\n        '
        template = self.get_prompt_template(prompt_template)
        if template:
            return list(template.prompt_params)
        return []

    def _prepare(self, query, file_paths, labels, documents, meta, invocation_context, prompt_template, generation_kwargs) -> Dict:
        if False:
            i = 10
            return i + 15
        '\n        Prepare prompt invocation.\n        '
        invocation_context = invocation_context or {}
        if query and 'query' not in invocation_context:
            invocation_context['query'] = query
        if file_paths and 'file_paths' not in invocation_context:
            invocation_context['file_paths'] = file_paths
        if labels and 'labels' not in invocation_context:
            invocation_context['labels'] = labels
        if documents and 'documents' not in invocation_context:
            invocation_context['documents'] = documents
        if meta and 'meta' not in invocation_context:
            invocation_context['meta'] = meta
        if 'prompt_template' not in invocation_context:
            invocation_context['prompt_template'] = self.get_prompt_template(prompt_template)
        if generation_kwargs:
            invocation_context.update(generation_kwargs)
        return invocation_context

    def run(self, query: Optional[str]=None, file_paths: Optional[List[str]]=None, labels: Optional[MultiLabel]=None, documents: Optional[List[Document]]=None, meta: Optional[dict]=None, invocation_context: Optional[Dict[str, Any]]=None, prompt_template: Optional[Union[str, PromptTemplate]]=None, generation_kwargs: Optional[Dict[str, Any]]=None) -> Tuple[Dict, str]:
        if False:
            return 10
        "\n        Runs the PromptNode on these input parameters. Returns the output of the prompt model.\n        The parameters `query`, `file_paths`, `labels`, `documents`, and `meta` are added to the invocation context\n        before invoking the prompt model. PromptNode uses these variables only if they are present as\n        parameters in the PromptTemplate.\n\n        :param query: The PromptNode usually ignores the query, unless it's used as a parameter in the\n        prompt template.\n        :param file_paths: The PromptNode usually ignores the file paths, unless they're used as a parameter\n        in the prompt template.\n        :param labels: The PromptNode usually ignores the labels, unless they're used as a parameter in the\n        prompt template.\n        :param documents: The documents to be used for the prompt.\n        :param meta: PromptNode usually ignores meta information, unless it's used as a parameter in the\n        PromptTemplate.\n        :param invocation_context: The invocation context to be used for the prompt.\n        :param prompt_template: The prompt template to use. You can choose between the following types:\n            - None: Use the default prompt template.\n            - PromptTemplate: Use the given prompt template object.\n            - str: Parses the string depending on its content:\n                - prompt template name: Uses the prompt template registered with the given name.\n                - prompt template yaml: Uses the prompt template specified by the given YAML.\n                - prompt text: Uses a copy of the default prompt template with the given prompt text.\n        :param generation_kwargs: The generation_kwargs are used to customize text generation for the underlying pipeline.\n        "
        prompt_collector: List[str] = []
        invocation_context = self._prepare(query, file_paths, labels, documents, meta, invocation_context, prompt_template, generation_kwargs)
        results = self(**invocation_context, prompt_collector=prompt_collector)
        prompt_template_resolved: PromptTemplate = invocation_context.pop('prompt_template')
        try:
            output_variable = self.output_variable or prompt_template_resolved.output_variable or 'results'
        except:
            output_variable = 'results'
        invocation_context[output_variable] = results
        invocation_context['prompts'] = prompt_collector
        final_result: Dict[str, Any] = {output_variable: results, 'invocation_context': invocation_context}
        if self.debug:
            final_result['_debug'] = {'prompts_used': prompt_collector}
        return (final_result, 'output_1')

    async def _aprompt(self, prompt_template: Optional[Union[str, PromptTemplate]], *args, **kwargs):
        """
        Async version of the actual prompt invocation.
        """
        results = []
        prompt_collector: List[Union[str, List[Dict[str, str]]]] = kwargs.pop('prompt_collector', [])
        kwargs = {**self._prepare_model_kwargs(), **kwargs}
        template_to_fill = self.get_prompt_template(prompt_template)
        if template_to_fill:
            for prompt in template_to_fill.fill(*args, **kwargs):
                kwargs_copy = template_to_fill.remove_template_params(copy.copy(kwargs))
                prompt = self.prompt_model._ensure_token_limit(prompt)
                prompt_collector.append(prompt)
                logger.debug('Prompt being sent to LLM with prompt %s and kwargs %s', prompt, kwargs_copy)
                output = await self.prompt_model.ainvoke(prompt, **kwargs_copy)
                results.extend(output)
            kwargs['prompts'] = prompt_collector
            results = template_to_fill.post_process(results, **kwargs)
        else:
            for prompt in list(args):
                kwargs_copy = copy.copy(kwargs)
                prompt = self.prompt_model._ensure_token_limit(prompt)
                prompt_collector.append(prompt)
                logger.debug('Prompt being sent to LLM with prompt %s and kwargs %s ', prompt, kwargs_copy)
                output = await self.prompt_model.ainvoke(prompt, **kwargs_copy)
                results.extend(output)
        return results

    async def arun(self, query: Optional[str]=None, file_paths: Optional[List[str]]=None, labels: Optional[MultiLabel]=None, documents: Optional[List[Document]]=None, meta: Optional[dict]=None, invocation_context: Optional[Dict[str, Any]]=None, prompt_template: Optional[Union[str, PromptTemplate]]=None, generation_kwargs: Optional[Dict[str, Any]]=None) -> Tuple[Dict, str]:
        """
        Drop-in replacement asyncio version of the `run` method, see there for documentation.
        """
        prompt_collector: List[str] = []
        invocation_context = self._prepare(query, file_paths, labels, documents, meta, invocation_context, prompt_template, generation_kwargs)
        results = await self._aprompt(prompt_collector=prompt_collector, **invocation_context)
        prompt_template_resolved: PromptTemplate = invocation_context.pop('prompt_template')
        try:
            output_variable = self.output_variable or prompt_template_resolved.output_variable or 'results'
        except:
            output_variable = 'results'
        invocation_context[output_variable] = results
        invocation_context['prompts'] = prompt_collector
        final_result: Dict[str, Any] = {output_variable: results, 'invocation_context': invocation_context}
        if self.debug:
            final_result['_debug'] = {'prompts_used': prompt_collector}
        return (final_result, 'output_1')

    def run_batch(self, queries: Optional[List[str]]=None, documents: Optional[Union[List[Document], List[List[Document]]]]=None, invocation_contexts: Optional[List[Dict[str, Any]]]=None, prompt_templates: Optional[List[Union[str, PromptTemplate]]]=None):
        if False:
            while True:
                i = 10
        '\n        Runs PromptNode in batch mode.\n\n        - If you provide a list containing a single query (or invocation context)...\n            - ... and a single list of Documents, the query is applied to each Document individually.\n            - ... and a list of lists of Documents, the query is applied to each list of Documents and the results\n              are aggregated per Document list.\n\n        - If you provide a list of multiple queries (or multiple invocation contexts)...\n            - ... and a single list of Documents, each query (or invocation context) is applied to each Document individually.\n            - ... and a list of lists of Documents, each query (or invocation context) is applied to its corresponding list of Documents\n              and the results are aggregated per query-Document pair.\n\n        - If you provide no Documents, then each query (or invocation context) is applied directly to the PromptTemplate.\n\n        :param queries: List of queries.\n        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.\n        :param invocation_contexts: List of invocation contexts.\n        :param prompt_templates: The prompt templates to use. You can choose between the following types:\n            - None: Use the default prompt template.\n            - PromptTemplate: Use the given prompt template object.\n            - str: Parses the string depending on its content:\n                - prompt template name: Uses the prompt template registered with the given name.\n                - prompt template yaml: Uuses the prompt template specified by the given YAML.\n                - prompt text: Uses a copy of the default prompt template with the given prompt text.\n        '
        inputs = PromptNode._flatten_inputs(queries, documents, invocation_contexts, prompt_templates)
        all_results: Dict[str, List] = defaultdict(list)
        for (query, docs, invocation_context, prompt_template) in zip(inputs['queries'], inputs['documents'], inputs['invocation_contexts'], inputs['prompt_templates']):
            prompt_template = self.get_prompt_template(self.default_prompt_template)
            output_variable = self.output_variable or prompt_template.output_variable or 'results'
            results = self.run(query=query, documents=docs, invocation_context=invocation_context, prompt_template=prompt_template)[0]
            all_results[output_variable].append(results[output_variable])
            all_results['invocation_contexts'].append(results['invocation_context'])
            if self.debug:
                all_results['_debug'].append(results['_debug'])
        return (all_results, 'output_1')

    def _prepare_model_kwargs(self):
        if False:
            i = 10
            return i + 15
        return {'stop_words': self.stop_words, 'top_k': self.top_k}

    @staticmethod
    def _flatten_inputs(queries: Optional[List[str]]=None, documents: Optional[Union[List[Document], List[List[Document]]]]=None, invocation_contexts: Optional[List[Dict[str, Any]]]=None, prompt_templates: Optional[List[Union[str, PromptTemplate]]]=None) -> Dict[str, List]:
        if False:
            return 10
        'Flatten and copy the queries, documents, and invocation contexts into lists of equal length.\n\n        - If you provide a list containing a single query (or invocation context)...\n            - ... and a single list of Documents, the query is applied to each Document individually.\n            - ... and a list of lists of Documents, the query is applied to each list of Documents and the results\n              are aggregated per Document list.\n\n        - If you provide a list of multiple queries (or multiple invocation contexts)...\n            - ... and a single list of Documents, each query (or invocation context) is applied to each Document individually.\n            - ... and a list of lists of Documents, each query (or invocation context) is applied to its corresponding list of Documents\n              and the results are aggregated per query-Document pair.\n\n        - If you provide no Documents, then each query (or invocation context) is applied to the PromptTemplate.\n\n        :param queries: List of queries.\n        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.\n        :param invocation_contexts: List of invocation contexts.\n        '
        input_queries: List[Any]
        input_invocation_contexts: List[Any]
        input_prompt_templates: List[Any]
        if queries is not None and invocation_contexts is not None:
            if len(queries) != len(invocation_contexts):
                raise ValueError('The input variables queries and invocation_contexts should have the same length.')
            input_queries = queries
            input_invocation_contexts = invocation_contexts
        elif queries is not None and invocation_contexts is None:
            input_queries = queries
            input_invocation_contexts = [None] * len(queries)
        elif queries is None and invocation_contexts is not None:
            input_queries = [None] * len(invocation_contexts)
            input_invocation_contexts = invocation_contexts
        else:
            input_queries = [None]
            input_invocation_contexts = [None]
        if prompt_templates is not None:
            if len(prompt_templates) != len(input_queries):
                raise ValueError('The input variables prompt_templates and queries should have the same length.')
            input_prompt_templates = prompt_templates
        else:
            input_prompt_templates = [None] * len(input_queries)
        multi_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list)
        single_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], Document)
        inputs: Dict[str, List] = defaultdict(list)
        if documents is not None:
            if single_docs_list:
                for (query, invocation_context, prompt_template) in zip(input_queries, input_invocation_contexts, input_prompt_templates):
                    for doc in documents:
                        inputs['queries'].append(query)
                        inputs['invocation_contexts'].append(invocation_context)
                        inputs['documents'].append([doc])
                        inputs['prompt_templates'].append(prompt_template)
            elif multi_docs_list:
                total_queries = input_queries.copy()
                total_invocation_contexts = input_invocation_contexts.copy()
                total_prompt_templates = input_prompt_templates.copy()
                if len(total_queries) == 1 and len(total_invocation_contexts) == 1 and (len(total_prompt_templates) == 1):
                    total_queries = input_queries * len(documents)
                    total_invocation_contexts = input_invocation_contexts * len(documents)
                    total_prompt_templates = input_prompt_templates * len(documents)
                if len(total_queries) != len(documents) or len(total_invocation_contexts) != len(documents) or len(total_prompt_templates) != len(documents):
                    raise ValueError('Number of queries must be equal to number of provided Document lists.')
                for (query, invocation_context, prompt_template, cur_docs) in zip(total_queries, total_invocation_contexts, total_prompt_templates, documents):
                    inputs['queries'].append(query)
                    inputs['invocation_contexts'].append(invocation_context)
                    inputs['documents'].append(cur_docs)
                    inputs['prompt_templates'].append(prompt_template)
        elif queries is not None or invocation_contexts is not None or prompt_templates is not None:
            for (query, invocation_context, prompt_template) in zip(input_queries, input_invocation_contexts, input_prompt_templates):
                inputs['queries'].append(query)
                inputs['invocation_contexts'].append(invocation_context)
                inputs['documents'].append([None])
                inputs['prompt_templates'].append(prompt_template)
        return inputs