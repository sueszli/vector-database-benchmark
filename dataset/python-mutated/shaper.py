from functools import reduce
import inspect
import re
from datetime import datetime
from string import Template
from typing import Literal, Optional, List, Dict, Any, Tuple, Union, Callable
import logging
from haystack.nodes.base import BaseComponent
from haystack.schema import Document, Answer, MultiLabel
logger = logging.getLogger(__name__)

def rename(value: Any) -> Any:
    if False:
        print('Hello World!')
    '\n    An identity function. You can use it to rename values in the invocation context without changing them.\n\n    Example:\n\n    ```python\n    assert rename(1) == 1\n    ```\n    '
    return value

def current_datetime(format: str='%H:%M:%S %d/%m/%y') -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Function that outputs the current time and/or date formatted according to the parameters.\n\n    Example:\n\n    ```python\n    assert current_datetime("%d.%m.%y %H:%M:%S") == 01.01.2023 12:30:10\n    ```\n    '
    return datetime.now().strftime(format)

def value_to_list(value: Any, target_list: List[Any]) -> List[Any]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Transforms a value into a list containing this value as many times as the length of the target list.\n\n    Example:\n\n    ```python\n    assert value_to_list(value=1, target_list=list(range(5))) == [1, 1, 1, 1, 1]\n    ```\n    '
    return [value] * len(target_list)

def join_lists(lists: List[List[Any]]) -> List[Any]:
    if False:
        return 10
    '\n    Joins the lists you pass to it into a single list.\n\n    Example:\n\n    ```python\n    assert join_lists(lists=[[1, 2, 3], [4, 5]]) == [1, 2, 3, 4, 5]\n    ```\n    '
    merged_list = []
    for inner_list in lists:
        merged_list += inner_list
    return merged_list

def join_strings(strings: List[str], delimiter: str=' ', str_replace: Optional[Dict[str, str]]=None) -> str:
    if False:
        return 10
    '\n    Transforms a list of strings into a single string. The content of this string\n    is the content of all of the original strings separated by the delimiter you specify.\n\n    Example:\n\n    ```python\n    assert join_strings(strings=["first", "second", "third"], delimiter=" - ", str_replace={"r": "R"}) == "fiRst - second - thiRd"\n    ```\n    '
    str_replace = str_replace or {}
    return delimiter.join([format_string(string, str_replace) for string in strings])

def format_string(string: str, str_replace: Optional[Dict[str, str]]=None) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Replaces strings.\n\n    Example:\n\n    ```python\n    assert format_string(string="first", str_replace={"r": "R"}) == "fiRst"\n    ```\n    '
    str_replace = str_replace or {}
    return reduce(lambda s, kv: s.replace(*kv), str_replace.items(), string)

def join_documents(documents: List[Document], delimiter: str=' ', pattern: Optional[str]=None, str_replace: Optional[Dict[str, str]]=None) -> List[Document]:
    if False:
        while True:
            i = 10
    '\n    Transforms a list of documents into a list containing a single document. The content of this document\n    is the joined result of all original documents, separated by the delimiter you specify.\n    Use regex in the `pattern` parameter to control how each document is represented.\n    You can use the following placeholders:\n    - $content: The content of the document.\n    - $idx: The index of the document in the list.\n    - $id: The ID of the document.\n    - $META_FIELD: The value of the metadata field called \'META_FIELD\'.\n\n    All metadata is dropped.\n\n    Example:\n\n    ```python\n    assert join_documents(\n        documents=[\n            Document(content="first"),\n            Document(content="second"),\n            Document(content="third")\n        ],\n        delimiter=" - ",\n        pattern="[$idx] $content",\n        str_replace={"r": "R"}\n    ) == [Document(content="[1] fiRst - [2] second - [3] thiRd")]\n    ```\n    '
    return [Document(content=join_documents_to_string(documents, delimiter, pattern, str_replace))]

def join_documents_and_scores(documents: List[Document]) -> Tuple[List[Document]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Transforms a list of documents with scores in their metadata into a list containing a single document.\n    The resulting document contains the scores and the contents of all the original documents.\n    All metadata is dropped.\n    Example:\n    ```python\n    assert join_documents_and_scores(\n        documents=[\n            Document(content="first", meta={"score": 0.9}),\n            Document(content="second", meta={"score": 0.7}),\n            Document(content="third", meta={"score": 0.5})\n        ],\n        delimiter=" - "\n    ) == ([Document(content="-[0.9] first\n -[0.7] second\n -[0.5] third")], )\n    ```\n    '
    content = '\n'.join([f"-[{round(float(doc.meta['score']), 2)}] {doc.content}" for doc in documents])
    return ([Document(content=content)],)

def format_document(document: Document, pattern: Optional[str]=None, str_replace: Optional[Dict[str, str]]=None, idx: Optional[int]=None) -> str:
    if False:
        print('Hello World!')
    '\n    Transforms a document into a single string.\n    Use regex in the `pattern` parameter to control how the document is represented.\n    You can use the following placeholders:\n    - $content: The content of the document.\n    - $idx: The index of the document in the list.\n    - $id: The ID of the document.\n    - $META_FIELD: The value of the metadata field called \'META_FIELD\'.\n\n    Example:\n\n    ```python\n    assert format_document(\n        document=Document(content="first"),\n        pattern="prefix [$idx] $content",\n        str_replace={"r": "R"},\n        idx=1,\n    ) == "prefix [1] fiRst"\n    ```\n    '
    str_replace = str_replace or {}
    pattern = pattern or '$content'
    template = Template(pattern)
    pattern_params = [match.groupdict().get('named', match.groupdict().get('braced')) for match in template.pattern.finditer(template.template)]
    meta_params = [param for param in pattern_params if param and param not in ['content', 'idx', 'id']]
    content = template.substitute({'idx': idx, 'content': reduce(lambda content, kv: content.replace(*kv), str_replace.items(), document.content), 'id': reduce(lambda id, kv: id.replace(*kv), str_replace.items(), document.id), **{k: reduce(lambda val, kv: val.replace(*kv), str_replace.items(), document.meta.get(k, '')) for k in meta_params}})
    return content

def format_answer(answer: Answer, pattern: Optional[str]=None, str_replace: Optional[Dict[str, str]]=None, idx: Optional[int]=None) -> str:
    if False:
        print('Hello World!')
    '\n    Transforms an answer into a single string.\n    Use regex in the `pattern` parameter to control how the answer is represented.\n    You can use the following placeholders:\n    - $answer: The answer text.\n    - $idx: The index of the answer in the list.\n    - $META_FIELD: The value of the metadata field called \'META_FIELD\'.\n\n    Example:\n\n    ```python\n    assert format_answer(\n        answer=Answer(answer="first"),\n        pattern="prefix [$idx] $answer",\n        str_replace={"r": "R"},\n        idx=1,\n    ) == "prefix [1] fiRst"\n    ```\n    '
    str_replace = str_replace or {}
    pattern = pattern or '$answer'
    template = Template(pattern)
    pattern_params = [match.groupdict().get('named', match.groupdict().get('braced')) for match in template.pattern.finditer(template.template)]
    meta_params = [param for param in pattern_params if param and param not in ['answer', 'idx']]
    meta = answer.meta or {}
    content = template.substitute({'idx': idx, 'answer': reduce(lambda content, kv: content.replace(*kv), str_replace.items(), answer.answer), **{k: reduce(lambda val, kv: val.replace(*kv), str_replace.items(), meta.get(k, '')) for k in meta_params}})
    return content

def join_documents_to_string(documents: List[Document], delimiter: str=' ', pattern: Optional[str]=None, str_replace: Optional[Dict[str, str]]=None) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Transforms a list of documents into a single string. The content of this string\n    is the joined result of all original documents separated by the delimiter you specify.\n    Use regex in the `pattern` parameter to control how the documents are represented.\n    You can use the following placeholders:\n    - $content: The content of the document.\n    - $idx: The index of the document in the list.\n    - $id: The ID of the document.\n    - $META_FIELD: The value of the metadata field called \'META_FIELD\'.\n\n    Example:\n\n    ```python\n    assert join_documents_to_string(\n        documents=[\n            Document(content="first"),\n            Document(content="second"),\n            Document(content="third")\n        ],\n        delimiter=" - ",\n        pattern="[$idx] $content",\n        str_replace={"r": "R"}\n    ) == "[1] fiRst - [2] second - [3] thiRd"\n    ```\n    '
    content = delimiter.join((format_document(doc, pattern, str_replace, idx=idx) for (idx, doc) in enumerate(documents, start=1)))
    return content

def strings_to_answers(strings: List[str], prompts: Optional[List[Union[str, List[Dict[str, str]]]]]=None, documents: Optional[List[Document]]=None, pattern: Optional[str]=None, reference_pattern: Optional[str]=None, reference_mode: Literal['index', 'id', 'meta']='index', reference_meta_field: Optional[str]=None) -> List[Answer]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Transforms a list of strings into a list of answers.\n    Specify `reference_pattern` to populate the answer\'s `document_ids` by extracting document references from the strings.\n\n    :param strings: The list of strings to transform.\n    :param prompts: The prompts used to generate the answers.\n    :param documents: The documents used to generate the answers.\n    :param pattern: The regex pattern to use for parsing the answer.\n        Examples:\n            `[^\\n]+$` will find "this is an answer" in string "this is an argument.\nthis is an answer".\n            `Answer: (.*)` will find "this is an answer" in string "this is an argument. Answer: this is an answer".\n        If None, the whole string is used as the answer. If not None, the first group of the regex is used as the answer. If there is no group, the whole match is used as the answer.\n    :param reference_pattern: The regex pattern to use for parsing the document references.\n        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".\n        If None, no parsing is done and all documents are referenced.\n    :param reference_mode: The mode used to reference documents. Supported modes are:\n        - index: the document references are the one-based index of the document in the list of documents.\n            Example: "this is an answer[1]" will reference the first document in the list of documents.\n        - id: the document references are the document IDs.\n            Example: "this is an answer[123]" will reference the document with id "123".\n        - meta: the document references are the value of a metadata field of the document.\n            Example: "this is an answer[123]" will reference the document with the value "123" in the metadata field specified by reference_meta_field.\n    :param reference_meta_field: The name of the metadata field to use for document references in reference_mode "meta".\n    :return: The list of answers.\n\n    Examples:\n\n    Without reference parsing:\n    ```python\n    assert strings_to_answers(strings=["first", "second", "third"], prompt="prompt", documents=[Document(id="123", content="content")]) == [\n            Answer(answer="first", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),\n            Answer(answer="second", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),\n            Answer(answer="third", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),\n        ]\n    ```\n\n    With reference parsing:\n    ```python\n    assert strings_to_answers(strings=["first[1]", "second[2]", "third[1][3]"], prompt="prompt",\n            documents=[Document(id="123", content="content"), Document(id="456", content="content"), Document(id="789", content="content")],\n            reference_pattern=r"\\[(\\d+)\\]",\n            reference_mode="index"\n        ) == [\n            Answer(answer="first", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),\n            Answer(answer="second", type="generative", document_ids=["456"], meta={"prompt": "prompt"}),\n            Answer(answer="third", type="generative", document_ids=["123", "789"], meta={"prompt": "prompt"}),\n        ]\n    ```\n    '
    if prompts:
        if len(prompts) == 1:
            documents_per_string: List[Optional[List[Document]]] = [documents] * len(strings)
            prompt_per_string: List[Optional[Union[str, List[Dict[str, str]]]]] = [prompts[0]] * len(strings)
        elif len(prompts) > 1 and len(strings) % len(prompts) == 0:
            if documents is not None and len(documents) != len(prompts):
                raise ValueError('The number of documents must match the number of prompts.')
            string_multiplier = len(strings) // len(prompts)
            documents_per_string = [[doc] for doc in documents for _ in range(string_multiplier)] if documents else [None] * len(strings)
            prompt_per_string = [prompt for prompt in prompts for _ in range(string_multiplier)]
        else:
            raise ValueError('The number of prompts must be one or a multiple of the number of strings.')
    else:
        documents_per_string = [documents] * len(strings)
        prompt_per_string = [None] * len(strings)
    answers = []
    for (string, prompt, _documents) in zip(strings, prompt_per_string, documents_per_string):
        answer = string_to_answer(string=string, prompt=prompt, documents=_documents, pattern=pattern, reference_pattern=reference_pattern, reference_mode=reference_mode, reference_meta_field=reference_meta_field)
        answers.append(answer)
    return answers

def string_to_answer(string: str, prompt: Optional[Union[str, List[Dict[str, str]]]], documents: Optional[List[Document]], pattern: Optional[str]=None, reference_pattern: Optional[str]=None, reference_mode: Literal['index', 'id', 'meta']='index', reference_meta_field: Optional[str]=None) -> Answer:
    if False:
        while True:
            i = 10
    '\n    Transforms a string into an answer.\n    Specify `reference_pattern` to populate the answer\'s `document_ids` by extracting document references from the string.\n\n    :param string: The string to transform.\n    :param prompt: The prompt used to generate the answer.\n    :param documents: The documents used to generate the answer.\n    :param pattern: The regex pattern to use for parsing the answer.\n        Examples:\n            `[^\\n]+$` will find "this is an answer" in string "this is an argument.\nthis is an answer".\n            `Answer: (.*)` will find "this is an answer" in string "this is an argument. Answer: this is an answer".\n        If None, the whole string is used as the answer. If not None, the first group of the regex is used as the answer. If there is no group, the whole match is used as the answer.\n    :param reference_pattern: The regex pattern to use for parsing the document references.\n        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".\n        If None, no parsing is done and all documents are referenced.\n    :param reference_mode: The mode used to reference documents. Supported modes are:\n        - index: the document references are the one-based index of the document in the list of documents.\n            Example: "this is an answer[1]" will reference the first document in the list of documents.\n        - id: the document references are the document IDs.\n            Example: "this is an answer[123]" will reference the document with id "123".\n        - meta: the document references are the value of a metadata field of the document.\n            Example: "this is an answer[123]" will reference the document with the value "123" in the metadata field specified by reference_meta_field.\n    :param reference_meta_field: The name of the metadata field to use for document references in reference_mode "meta".\n    :return: The answer\n    '
    if reference_mode == 'index':
        candidates = {str(idx): doc.id for (idx, doc) in enumerate(documents, start=1)} if documents else {}
    elif reference_mode == 'id':
        candidates = {doc.id: doc.id for doc in documents} if documents else {}
    elif reference_mode == 'meta':
        if not reference_meta_field:
            raise ValueError("reference_meta_field must be specified when reference_mode is 'meta'")
        candidates = {doc.meta[reference_meta_field]: doc.id for doc in documents if doc.meta.get(reference_meta_field)} if documents else {}
    else:
        raise ValueError(f'Invalid document_id_mode: {reference_mode}')
    if pattern:
        match = re.search(pattern, string)
        if match:
            if not match.lastindex:
                string = match.group(0)
            elif match.lastindex == 1:
                string = match.group(1)
            else:
                raise ValueError(f'Pattern must have at most one group: {pattern}')
        else:
            string = ''
    document_ids = parse_references(string=string, reference_pattern=reference_pattern, candidates=candidates)
    answer = Answer(answer=string, type='generative', document_ids=document_ids, meta={'prompt': prompt})
    return answer

def parse_references(string: str, reference_pattern: Optional[str]=None, candidates: Optional[Dict[str, str]]=None) -> Optional[List[str]]:
    if False:
        return 10
    '\n    Parses an answer string for document references and returns the document IDs of the referenced documents.\n\n    :param string: The string to parse.\n    :param reference_pattern: The regex pattern to use for parsing the document references.\n        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".\n        If None, no parsing is done and all candidate document IDs are returned.\n    :param candidates: A dictionary of candidates to choose from. The keys are the reference strings and the values are the document IDs.\n        If None, no parsing is done and None is returned.\n    :return: A list of document IDs.\n    '
    if not candidates:
        return None
    if not reference_pattern:
        return list(candidates.values())
    document_idxs = re.findall(reference_pattern, string)
    return [candidates[idx] for idx in document_idxs if idx in candidates]

def answers_to_strings(answers: List[Answer], pattern: Optional[str]=None, str_replace: Optional[Dict[str, str]]=None) -> List[str]:
    if False:
        while True:
            i = 10
    '\n    Extracts the content field of answers and returns a list of strings.\n\n    Example:\n\n    ```python\n    assert answers_to_strings(\n            answers=[\n                Answer(answer="first"),\n                Answer(answer="second"),\n                Answer(answer="third")\n            ],\n            pattern="[$idx] $answer",\n            str_replace={"r": "R"}\n        ) == ["[1] fiRst", "[2] second", "[3] thiRd"]\n    ```\n    '
    return [format_answer(answer, pattern, str_replace, idx) for (idx, answer) in enumerate(answers, start=1)]

def strings_to_documents(strings: List[str], meta: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]=None, id_hash_keys: Optional[List[str]]=None) -> List[Document]:
    if False:
        print('Hello World!')
    '\n    Transforms a list of strings into a list of documents. If you pass the metadata in a single\n    dictionary, all documents get the same metadata. If you pass the metadata as a list, the length of this list\n    must be the same as the length of the list of strings, and each document gets its own metadata.\n    You can specify `id_hash_keys` only once and it gets assigned to all documents.\n\n    Example:\n\n    ```python\n    assert strings_to_documents(\n            strings=["first", "second", "third"],\n            meta=[{"position": i} for i in range(3)],\n            id_hash_keys=[\'content\', \'meta]\n        ) == [\n            Document(content="first", metadata={"position": 1}, id_hash_keys=[\'content\', \'meta])]),\n            Document(content="second", metadata={"position": 2}, id_hash_keys=[\'content\', \'meta]),\n            Document(content="third", metadata={"position": 3}, id_hash_keys=[\'content\', \'meta])\n        ]\n    ```\n    '
    all_metadata: List[Optional[Dict[str, Any]]]
    if isinstance(meta, dict):
        all_metadata = [meta] * len(strings)
    elif isinstance(meta, list):
        if len(meta) != len(strings):
            raise ValueError(f'Not enough metadata dictionaries. strings_to_documents received {len(strings)} and {len(meta)} metadata dictionaries.')
        all_metadata = meta
    else:
        all_metadata = [None] * len(strings)
    return [Document(content=string, meta=m, id_hash_keys=id_hash_keys) for (string, m) in zip(strings, all_metadata)]

def documents_to_strings(documents: List[Document], pattern: Optional[str]=None, str_replace: Optional[Dict[str, str]]=None) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Extracts the content field of documents and returns a list of strings. Use regext in the `pattern` parameter to control how the documents are represented.\n\n    Example:\n\n    ```python\n    assert documents_to_strings(\n            documents=[\n                Document(content="first"),\n                Document(content="second"),\n                Document(content="third")\n            ],\n            pattern="[$idx] $content",\n            str_replace={"r": "R"}\n        ) == ["[1] fiRst", "[2] second", "[3] thiRd"]\n    ```\n    '
    return [format_document(doc, pattern, str_replace, idx) for (idx, doc) in enumerate(documents, start=1)]
REGISTERED_FUNCTIONS: Dict[str, Callable[..., Any]] = {'rename': rename, 'current_datetime': current_datetime, 'value_to_list': value_to_list, 'join_lists': join_lists, 'join_strings': join_strings, 'join_documents': join_documents, 'join_documents_and_scores': join_documents_and_scores, 'strings_to_answers': strings_to_answers, 'answers_to_strings': answers_to_strings, 'strings_to_documents': strings_to_documents, 'documents_to_strings': documents_to_strings}

class Shaper(BaseComponent):
    """
    Shaper is a component that can invoke arbitrary, registered functions on the invocation context
    (query, documents, and so on) of a pipeline. It then passes the new or modified variables further down the pipeline.

    Using YAML configuration, the Shaper component is initialized with functions to invoke on pipeline invocation
    context.

    For example, in the YAML snippet below:
    ```yaml
        components:
        - name: shaper
          type: Shaper
          params:
            func: value_to_list
            inputs:
                value: query
                target_list: documents
            output: [questions]
    ```
    the Shaper component is initialized with a directive to invoke function expand on the variable query and to store
    the result in the invocation context variable questions. All other invocation context variables are passed down
    the pipeline as they are.

    You can use multiple Shaper components in a pipeline to modify the invocation context as needed.

    Currently, `Shaper` supports the following functions:

    - `rename`
    - `value_to_list`
    - `join_lists`
    - `join_strings`
    - `format_string`
    - `join_documents`
    - `join_documents_and_scores`
    - `format_document`
    - `format_answer`
    - `join_documents_to_string`
    - `strings_to_answers`
    - `string_to_answer`
    - `parse_references`
    - `answers_to_strings`
    - `join_lists`
    - `strings_to_documents`
    - `documents_to_strings`

    See their descriptions in the code for details about their inputs, outputs, and other parameters.
    """
    outgoing_edges = 1

    def __init__(self, func: str, outputs: List[str], inputs: Optional[Dict[str, Union[List[str], str]]]=None, params: Optional[Dict[str, Any]]=None, publish_outputs: Union[bool, List[str]]=True):
        if False:
            i = 10
            return i + 15
        '\n        Initializes the Shaper component.\n\n        Some examples:\n\n        ```yaml\n        - name: shaper\n          type: Shaper\n          params:\n          func: value_to_list\n          inputs:\n            value: query\n            target_list: documents\n          outputs:\n            - questions\n        ```\n        This node takes the content of `query` and creates a list that contains the value of `query` `len(documents)` times.\n        This list is stored in the invocation context under the key `questions`.\n\n        ```yaml\n        - name: shaper\n          type: Shaper\n          params:\n          func: join_documents\n          inputs:\n            value: documents\n          params:\n            delimiter: \' - \'\n          outputs:\n            - documents\n        ```\n        This node overwrites the content of `documents` in the invocation context with a list containing a single Document\n        whose content is the concatenation of all the original Documents. So if `documents` contained\n        `[Document("A"), Document("B"), Document("C")]`, this shaper overwrites it with `[Document("A - B - C")]`\n\n        ```yaml\n        - name: shaper\n          type: Shaper\n          params:\n          func: join_strings\n          params:\n            strings: [\'a\', \'b\', \'c\']\n            delimiter: \' . \'\n          outputs:\n            - single_string\n\n        - name: shaper\n          type: Shaper\n          params:\n          func: strings_to_documents\n          inputs:\n            strings: single_string\n            metadata:\n              name: \'my_file.txt\'\n          outputs:\n            - single_document\n        ```\n        These two nodes, executed one after the other, first add a key in the invocation context called `single_string`\n        that contains `a . b . c`, and then create another key called `single_document` that contains instead\n        `[Document(content="a . b . c", metadata={\'name\': \'my_file.txt\'})]`.\n\n        :param func: The function to apply.\n        :param inputs: Maps the function\'s input kwargs to the key-value pairs in the invocation context.\n            For example, `value_to_list` expects the `value` and `target_list` parameters, so `inputs` might contain:\n            `{\'value\': \'query\', \'target_list\': \'documents\'}`. It doesn\'t need to contain all keyword args, see `params`.\n        :param params: Maps the function\'s input kwargs to some fixed values. For example, `value_to_list` expects\n            `value` and `target_list` parameters, so `params` might contain\n            `{\'value\': \'A\', \'target_list\': [1, 1, 1, 1]}` and the node\'s output is `["A", "A", "A", "A"]`.\n            It doesn\'t need to contain all keyword args, see `inputs`.\n            You can use params to provide fallback values for arguments of `run` that you\'re not sure exist.\n            So if you need `query` to exist, you can provide a fallback value in the params, which will be used only if `query`\n            is not passed to this node by the pipeline.\n        :param outputs: The key to store the outputs in the invocation context. The length of the outputs must match\n            the number of outputs produced by the function invoked.\n        :param publish_outputs: Controls whether to publish the outputs to the pipeline\'s output.\n            Set `True` (default value) to publishes all outputs or `False` to publish None.\n            E.g. if `outputs = ["documents"]` result for `publish_outputs = True` looks like\n            ```python\n                {\n                    "invocation_context": {\n                        "documents": [...]\n                    },\n                    "documents": [...]\n                }\n            ```\n            For `publish_outputs = False` result looks like\n            ```python\n                {\n                    "invocation_context": {\n                        "documents": [...]\n                    },\n                }\n            ```\n            If you want to have finer-grained control, pass a list of the outputs you want to publish.\n        '
        super().__init__()
        self.function = REGISTERED_FUNCTIONS[func]
        self.outputs = outputs
        self.inputs = inputs or {}
        self.params = params or {}
        if isinstance(publish_outputs, bool):
            self.publish_outputs = self.outputs if publish_outputs else []
        else:
            self.publish_outputs = publish_outputs

    def run(self, query: Optional[str]=None, file_paths: Optional[List[str]]=None, labels: Optional[MultiLabel]=None, documents: Optional[List[Document]]=None, meta: Optional[dict]=None, invocation_context: Optional[Dict[str, Any]]=None) -> Tuple[Dict, str]:
        if False:
            return 10
        invocation_context = invocation_context or {}
        if query and 'query' not in invocation_context.keys():
            invocation_context['query'] = query
        if file_paths and 'file_paths' not in invocation_context.keys():
            invocation_context['file_paths'] = file_paths
        if labels and 'labels' not in invocation_context.keys():
            invocation_context['labels'] = labels
        if documents != None and 'documents' not in invocation_context.keys():
            invocation_context['documents'] = documents
        if meta and 'meta' not in invocation_context.keys():
            invocation_context['meta'] = meta
        input_values: Dict[str, Any] = {}
        for (key, value) in self.inputs.items():
            if isinstance(value, list):
                input_values[key] = []
                for v in value:
                    if v in invocation_context.keys() and v is not None:
                        input_values[key].append(invocation_context[v])
            elif value in invocation_context.keys() and value is not None:
                input_values[key] = invocation_context[value]
        function_params = inspect.signature(self.function).parameters
        for parameter in function_params.values():
            if parameter.name not in input_values.keys() and parameter.name not in self.params.keys() and (parameter.name in invocation_context.keys()):
                input_values[parameter.name] = invocation_context[parameter.name]
        input_values = {**self.params, **input_values}
        try:
            logger.debug('Shaper is invoking this function: %s(%s)', self.function.__name__, ', '.join([f'{key}={value}' for (key, value) in input_values.items()]))
            output_values = self.function(**input_values)
            if not isinstance(output_values, tuple):
                output_values = (output_values,)
        except TypeError as e:
            raise ValueError("Shaper couldn't apply the function to your inputs and parameters. Check the above stacktrace and make sure you provided all the correct inputs, parameters, and parameter types.") from e
        if len(self.outputs) < len(output_values):
            logger.warning('The number of outputs from function %s is %s. However, only %s output key(s) were provided. Only %s output(s) will be stored. Provide %s output keys to store all outputs.', self.function.__name__, len(output_values), len(self.outputs), len(self.outputs), len(output_values))
        if len(self.outputs) > len(output_values):
            logger.warning('The number of outputs from function %s is %s. However, %s output key(s) were provided. Only the first %s output key(s) will be used.', self.function.__name__, len(output_values), len(self.outputs), len(output_values))
        results = {}
        for (output_key, output_value) in zip(self.outputs, output_values):
            invocation_context[output_key] = output_value
            if output_key in self.publish_outputs:
                results[output_key] = output_value
        results['invocation_context'] = invocation_context
        return (results, 'output_1')

    def run_batch(self, query: Optional[str]=None, file_paths: Optional[List[str]]=None, labels: Optional[MultiLabel]=None, documents: Optional[List[Document]]=None, meta: Optional[dict]=None, invocation_context: Optional[Dict[str, Any]]=None) -> Tuple[Dict, str]:
        if False:
            while True:
                i = 10
        return self.run(query=query, file_paths=file_paths, labels=labels, documents=documents, meta=meta, invocation_context=invocation_context)