"""**Text Splitters** are classes for splitting text.


**Class hierarchy:**

.. code-block::

    BaseDocumentTransformer --> TextSplitter --> <name>TextSplitter  # Example: CharacterTextSplitter
                                                 RecursiveCharacterTextSplitter -->  <name>TextSplitter

Note: **MarkdownHeaderTextSplitter** does not derive from TextSplitter.


**Main helpers:**

.. code-block::

    Document, Tokenizer, Language, LineType, HeaderType

"""
from __future__ import annotations
import copy
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AbstractSet, Any, Callable, Collection, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Type, TypedDict, TypeVar, Union, cast
from azure.ai.generative.index._langchain.vendor.schema.document import Document
from azure.ai.generative.index._langchain.vendor.schema.document import BaseDocumentTransformer
logger = logging.getLogger(__name__)
TS = TypeVar('TS', bound='TextSplitter')

def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
    if False:
        print('Hello World!')
    if separator:
        if keep_separator:
            _splits = re.split(f'({separator})', text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != '']

class TextSplitter(BaseDocumentTransformer, ABC):
    """Interface for splitting text into chunks."""

    def __init__(self, chunk_size: int=4000, chunk_overlap: int=200, length_function: Callable[[str], int]=len, keep_separator: bool=False, add_start_index: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        "Create a new TextSplitter.\n\n        Args:\n            chunk_size: Maximum size of chunks to return\n            chunk_overlap: Overlap in characters between chunks\n            length_function: Function that measures the length of given chunks\n            keep_separator: Whether to keep the separator in the chunks\n            add_start_index: If `True`, includes chunk's start index in metadata\n        "
        if chunk_overlap > chunk_size:
            raise ValueError(f'Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smaller.')
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        if False:
            return 10
        'Split text into multiple components.'

    def create_documents(self, texts: List[str], metadatas: Optional[List[dict]]=None) -> List[Document]:
        if False:
            while True:
                i = 10
        'Create documents from a list of texts.'
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for (i, text) in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata['start_index'] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        if False:
            for i in range(10):
                print('nop')
        'Split documents.'
        (texts, metadatas) = ([], [])
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        if False:
            return 10
        text = separator.join(docs)
        text = text.strip()
        if text == '':
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        separator_len = self._length_function(separator)
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(f'Created a chunk of size {total}, which is longer than the specified {self._chunk_size}')
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > self._chunk_overlap or (total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size and total > 0):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
        if False:
            return 10
        'Text splitter that uses HuggingFace tokenizer to count length.'
        try:
            from transformers import PreTrainedTokenizerBase
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError('Tokenizer received was not an instance of PreTrainedTokenizerBase')

            def _huggingface_tokenizer_length(text: str) -> int:
                if False:
                    i = 10
                    return i + 15
                return len(tokenizer.encode(text))
        except ImportError:
            raise ValueError('Could not import transformers python package. Please install it with `pip install transformers`.')
        return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    @classmethod
    def from_tiktoken_encoder(cls: Type[TS], encoding_name: str='gpt2', model_name: Optional[str]=None, allowed_special: Union[Literal['all'], AbstractSet[str]]=set(), disallowed_special: Union[Literal['all'], Collection[str]]='all', **kwargs: Any) -> TS:
        if False:
            for i in range(10):
                print('nop')
        'Text splitter that uses tiktoken encoder to count length.'
        try:
            import tiktoken
        except ImportError:
            raise ImportError('Could not import tiktoken python package. This is needed in order to calculate max_tokens_for_prompt. Please install it with `pip install tiktoken`.')
        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str) -> int:
            if False:
                print('Hello World!')
            return len(enc.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special))
        if issubclass(cls, TokenTextSplitter):
            extra_kwargs = {'encoding_name': encoding_name, 'model_name': model_name, 'allowed_special': allowed_special, 'disallowed_special': disallowed_special}
            kwargs = {**kwargs, **extra_kwargs}
        return cls(length_function=_tiktoken_encoder, **kwargs)

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        if False:
            while True:
                i = 10
        'Transform sequence of documents by splitting them.'
        return self.split_documents(list(documents))

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Asynchronously transform a sequence of documents by splitting them."""
        raise NotImplementedError

class CharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(self, separator: str='\n\n', is_separator_regex: bool=False, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Create a new TextSplitter.'
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Split incoming text and return chunks.'
        separator = self._separator if self._is_separator_regex else re.escape(self._separator)
        splits = _split_text_with_regex(text, separator, self._keep_separator)
        _separator = '' if self._keep_separator else self._separator
        return self._merge_splits(splits, _separator)

class LineType(TypedDict):
    """Line type as typed dict."""
    metadata: Dict[str, str]
    content: str

class HeaderType(TypedDict):
    """Header type as typed dict."""
    level: int
    name: str
    data: str

class MarkdownHeaderTextSplitter:
    """Splitting markdown files based on specified headers."""

    def __init__(self, headers_to_split_on: List[Tuple[str, str]], return_each_line: bool=False):
        if False:
            i = 10
            return i + 15
        'Create a new MarkdownHeaderTextSplitter.\n\n        Args:\n            headers_to_split_on: Headers we want to track\n            return_each_line: Return each line w/ associated headers\n        '
        self.return_each_line = return_each_line
        self.headers_to_split_on = sorted(headers_to_split_on, key=lambda split: len(split[0]), reverse=True)

    def aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[Document]:
        if False:
            return 10
        'Combine lines with common metadata into chunks\n        Args:\n            lines: Line of text / associated header metadata\n        '
        aggregated_chunks: List[LineType] = []
        for line in lines:
            if aggregated_chunks and aggregated_chunks[-1]['metadata'] == line['metadata']:
                aggregated_chunks[-1]['content'] += '  \n' + line['content']
            else:
                aggregated_chunks.append(line)
        return [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in aggregated_chunks]

    def split_text(self, text: str) -> List[Document]:
        if False:
            while True:
                i = 10
        'Split markdown file\n        Args:\n            text: Markdown file'
        lines = text.split('\n')
        lines_with_metadata: List[LineType] = []
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        header_stack: List[HeaderType] = []
        initial_metadata: Dict[str, str] = {}
        for line in lines:
            stripped_line = line.strip()
            for (sep, name) in self.headers_to_split_on:
                if stripped_line.startswith(sep) and (len(stripped_line) == len(sep) or stripped_line[len(sep)] == ' '):
                    if name is not None:
                        current_header_level = sep.count('#')
                        while header_stack and header_stack[-1]['level'] >= current_header_level:
                            popped_header = header_stack.pop()
                            if popped_header['name'] in initial_metadata:
                                initial_metadata.pop(popped_header['name'])
                        header: HeaderType = {'level': current_header_level, 'name': name, 'data': stripped_line[len(sep):].strip()}
                        header_stack.append(header)
                        initial_metadata[name] = header['data']
                    if current_content:
                        lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata.copy()})
                        current_content.clear()
                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata.copy()})
                    current_content.clear()
            current_metadata = initial_metadata.copy()
        if current_content:
            lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata})
        if not self.return_each_line:
            return self.aggregate_lines_to_chunks(lines_with_metadata)
        else:
            return [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in lines_with_metadata]

@dataclass(frozen=True)
class Tokenizer:
    chunk_overlap: int
    tokens_per_chunk: int
    decode: Callable[[list[int]], str]
    encode: Callable[[str], List[int]]

def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    if False:
        print('Hello World!')
    'Split incoming text and return chunks using tokenizer.'
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits

class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(self, encoding_name: str='gpt2', model_name: Optional[str]=None, allowed_special: Union[Literal['all'], AbstractSet[str]]=set(), disallowed_special: Union[Literal['all'], Collection[str]]='all', **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Create a new TextSplitter.'
        super().__init__(**kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError('Could not import tiktoken python package. This is needed in order to for TokenTextSplitter. Please install it with `pip install tiktoken`.')
        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        if False:
            print('Hello World!')

        def _encode(_text: str) -> List[int]:
            if False:
                while True:
                    i = 10
            return self._tokenizer.encode(_text, allowed_special=self._allowed_special, disallowed_special=self._disallowed_special)
        tokenizer = Tokenizer(chunk_overlap=self._chunk_overlap, tokens_per_chunk=self._chunk_size, decode=self._tokenizer.decode, encode=_encode)
        return split_text_on_tokens(text=text, tokenizer=tokenizer)

class SentenceTransformersTokenTextSplitter(TextSplitter):
    """Splitting text to tokens using sentence model tokenizer."""

    def __init__(self, chunk_overlap: int=50, model_name: str='sentence-transformers/all-mpnet-base-v2', tokens_per_chunk: Optional[int]=None, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Create a new TextSplitter.'
        super().__init__(**kwargs, chunk_overlap=chunk_overlap)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError('Could not import sentence_transformer python package. This is needed in order to for SentenceTransformersTokenTextSplitter. Please install it with `pip install sentence-transformers`.')
        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name)
        self.tokenizer = self._model.tokenizer
        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

    def _initialize_chunk_configuration(self, *, tokens_per_chunk: Optional[int]) -> None:
        if False:
            return 10
        self.maximum_tokens_per_chunk = cast(int, self._model.max_seq_length)
        if tokens_per_chunk is None:
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk
        if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
            raise ValueError(f"The token limit of the models '{self.model_name}' is: {self.maximum_tokens_per_chunk}. Argument tokens_per_chunk={self.tokens_per_chunk} > maximum token limit.")

    def split_text(self, text: str) -> List[str]:
        if False:
            i = 10
            return i + 15

        def encode_strip_start_and_stop_token_ids(text: str) -> List[int]:
            if False:
                for i in range(10):
                    print('nop')
            return self._encode(text)[1:-1]
        tokenizer = Tokenizer(chunk_overlap=self._chunk_overlap, tokens_per_chunk=self.tokens_per_chunk, decode=self.tokenizer.decode, encode=encode_strip_start_and_stop_token_ids)
        return split_text_on_tokens(text=text, tokenizer=tokenizer)

    def count_tokens(self, *, text: str) -> int:
        if False:
            return 10
        return len(self._encode(text))
    _max_length_equal_32_bit_integer: int = 2 ** 32

    def _encode(self, text: str) -> List[int]:
        if False:
            print('Hello World!')
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(text, max_length=self._max_length_equal_32_bit_integer, truncation='do_not_truncate')
        return token_ids_with_start_and_end_token_ids

class Language(str, Enum):
    """Enum of the programming languages."""
    CPP = 'cpp'
    GO = 'go'
    JAVA = 'java'
    JS = 'js'
    PHP = 'php'
    PROTO = 'proto'
    PYTHON = 'python'
    RST = 'rst'
    RUBY = 'ruby'
    RUST = 'rust'
    SCALA = 'scala'
    SWIFT = 'swift'
    MARKDOWN = 'markdown'
    LATEX = 'latex'
    HTML = 'html'
    SOL = 'sol'

class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(self, separators: Optional[List[str]]=None, keep_separator: bool=True, is_separator_regex: bool=False, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Create a new TextSplitter.'
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ['\n\n', '\n', ' ', '']
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Split incoming text and return chunks.'
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for (i, _s) in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == '':
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break
        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)
        _good_splits = []
        _separator = '' if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(cls, language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter:
        if False:
            i = 10
            return i + 15
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        if False:
            while True:
                i = 10
        if language == Language.CPP:
            return ['\nclass ', '\nvoid ', '\nint ', '\nfloat ', '\ndouble ', '\nif ', '\nfor ', '\nwhile ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.GO:
            return ['\nfunc ', '\nvar ', '\nconst ', '\ntype ', '\nif ', '\nfor ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.JAVA:
            return ['\nclass ', '\npublic ', '\nprotected ', '\nprivate ', '\nstatic ', '\nif ', '\nfor ', '\nwhile ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.JS:
            return ['\nfunction ', '\nconst ', '\nlet ', '\nvar ', '\nclass ', '\nif ', '\nfor ', '\nwhile ', '\nswitch ', '\ncase ', '\ndefault ', '\n\n', '\n', ' ', '']
        elif language == Language.PHP:
            return ['\nfunction ', '\nclass ', '\nif ', '\nforeach ', '\nwhile ', '\ndo ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.PROTO:
            return ['\nmessage ', '\nservice ', '\nenum ', '\noption ', '\nimport ', '\nsyntax ', '\n\n', '\n', ' ', '']
        elif language == Language.PYTHON:
            return ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
        elif language == Language.RST:
            return ['\n=+\n', '\n-+\n', '\n\\*+\n', '\n\n.. *\n\n', '\n\n', '\n', ' ', '']
        elif language == Language.RUBY:
            return ['\ndef ', '\nclass ', '\nif ', '\nunless ', '\nwhile ', '\nfor ', '\ndo ', '\nbegin ', '\nrescue ', '\n\n', '\n', ' ', '']
        elif language == Language.RUST:
            return ['\nfn ', '\nconst ', '\nlet ', '\nif ', '\nwhile ', '\nfor ', '\nloop ', '\nmatch ', '\nconst ', '\n\n', '\n', ' ', '']
        elif language == Language.SCALA:
            return ['\nclass ', '\nobject ', '\ndef ', '\nval ', '\nvar ', '\nif ', '\nfor ', '\nwhile ', '\nmatch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.SWIFT:
            return ['\nfunc ', '\nclass ', '\nstruct ', '\nenum ', '\nif ', '\nfor ', '\nwhile ', '\ndo ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.MARKDOWN:
            return ['\n#{1,6} ', '```\n', '\n\\*\\*\\*+\n', '\n---+\n', '\n___+\n', '\n\n', '\n', ' ', '']
        elif language == Language.LATEX:
            return ['\n\\\\chapter{', '\n\\\\section{', '\n\\\\subsection{', '\n\\\\subsubsection{', '\n\\\\begin{enumerate}', '\n\\\\begin{itemize}', '\n\\\\begin{description}', '\n\\\\begin{list}', '\n\\\\begin{quote}', '\n\\\\begin{quotation}', '\n\\\\begin{verse}', '\n\\\\begin{verbatim}', '\n\\\x08egin{align}', '$$', '$', ' ', '']
        elif language == Language.HTML:
            return ['<body', '<div', '<p', '<br', '<li', '<h1', '<h2', '<h3', '<h4', '<h5', '<h6', '<span', '<table', '<tr', '<td', '<th', '<ul', '<ol', '<header', '<footer', '<nav', '<head', '<style', '<script', '<meta', '<title', '']
        elif language == Language.SOL:
            return ['\npragma ', '\nusing ', '\ncontract ', '\ninterface ', '\nlibrary ', '\nconstructor ', '\ntype ', '\nfunction ', '\nevent ', '\nmodifier ', '\nerror ', '\nstruct ', '\nenum ', '\nif ', '\nfor ', '\nwhile ', '\ndo while ', '\nassembly ', '\n\n', '\n', ' ', '']
        else:
            raise ValueError(f'Language {language} is not supported! Please choose from {list(Language)}')

class NLTKTextSplitter(TextSplitter):
    """Splitting text using NLTK package."""

    def __init__(self, separator: str='\n\n', **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Initialize the NLTK splitter.'
        super().__init__(**kwargs)
        try:
            from nltk.tokenize import sent_tokenize
            self._tokenizer = sent_tokenize
        except ImportError:
            raise ImportError('NLTK is not installed, please install it with `pip install nltk`.')
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        if False:
            while True:
                i = 10
        'Split incoming text and return chunks.'
        splits = self._tokenizer(text)
        return self._merge_splits(splits, self._separator)

class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize a MarkdownTextSplitter.'
        separators = self.get_separators_for_language(Language.MARKDOWN)
        super().__init__(separators=separators, **kwargs)