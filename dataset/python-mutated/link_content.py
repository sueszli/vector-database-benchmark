import inspect
import io
import logging
from collections import defaultdict
from datetime import datetime
from http import HTTPStatus
from typing import Optional, Dict, List, Union, Callable, Any, Tuple
from urllib.parse import urlparse
import requests
from boilerpy3 import extractors
from requests import Response
from requests.exceptions import InvalidURL, HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState
from haystack import __version__
from haystack.lazy_imports import LazyImport
from haystack.nodes import PreProcessor, BaseComponent
from haystack.schema import Document, MultiLabel
logger = logging.getLogger(__name__)
with LazyImport("Run 'pip install farm-haystack[pdf]'") as fitz_import:
    import fitz

def html_content_handler(response: Response) -> Optional[str]:
    if False:
        while True:
            i = 10
    '\n    Extracts text from HTML response text using the boilerpy3 extractor.\n    :param response: Response object from the request.\n    :return: The extracted text.\n    '
    extractor = extractors.ArticleExtractor(raise_on_failure=False)
    return extractor.get_content(response.text)

def pdf_content_handler(response: Response) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    '\n    Extracts text from PDF response stream using the PyMuPDF library.\n\n    :param response: Response object from the request.\n    :return: The extracted text.\n    '
    file_path = io.BytesIO(response.content)
    with fitz.open(stream=file_path, filetype='pdf') as doc:
        text = '\x0c'.join([page.get_text() for page in doc])
    return text.encode('ascii', errors='ignore').decode()

class LinkContentFetcher(BaseComponent):
    """
    LinkContentFetcher fetches content from a URL and converts it into a list of Document objects.

    LinkContentFetcher supports the following content types:
    - HTML
    - PDF

    LinkContentFetcher offers a few options for customizing the content extraction process:
    - content_handlers: A dictionary of content handlers to use for extracting content from a response.
    - processor: PreProcessor to apply to the extracted text
    - raise_on_failure: A boolean indicating whether to raise an exception when a failure occurs

    One can use LinkContentFetcher as a standalone component or as part of a Pipeline. Here is an example of using
    LinkContentFetcher as a standalone component:

    ```python
    from haystack.nodes import LinkContentFetcher
    from haystack.schema import Document

    link_content_fetcher = LinkContentFetcher()
    dl_wiki: List[Document] = link_content_fetcher.fetch(url="https://en.wikipedia.org/wiki/Deep_learning")
    print(dl_wiki)
    ```

    One can also use LinkContentFetcher as part of a Pipeline. Here is an example of using LinkContentFetcher as part
    of a Pipeline:

    ```python
    import os
    from haystack.nodes import PromptNode, LinkContentFetcher, PromptTemplate
    from haystack import Pipeline

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")


    retriever = LinkContentFetcher() # optionally add additional user agents
    pt = PromptTemplate(
        "Given the content below, create a summary consisting of three sections: Objectives, "
        "Implementation and Learnings/Conclusions.
"
        "Each section should have at least three bullet points. 
"
        "In the content below disregard References section.

: {documents}"
    )

    prompt_node = PromptNode("claude-instant-1",
                              api_key=anthropic_key,
                              max_length=512,
                              default_prompt_template=pt,
                              model_kwargs={"stream": True}
                              )

    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

    research_papers = ["https://arxiv.org/pdf/2307.03172.pdf", "https://arxiv.org/pdf/1706.03762.pdf"]

    for research_paper in research_papers:
        print(f"Research paper summary: {research_paper}")
        pipeline.run(research_paper)
        print("


")
    """
    outgoing_edges = 1
    _USER_AGENT = f'haystack/LinkContentRetriever/{__version__}'
    _REQUEST_HEADERS = {'accept': '*/*', 'User-Agent': _USER_AGENT, 'Accept-Language': 'en-US,en;q=0.9,it;q=0.8,es;q=0.7', 'referer': 'https://www.google.com/'}

    def __init__(self, content_handlers: Optional[Dict[str, Callable]]=None, processor: Optional[PreProcessor]=None, raise_on_failure: Optional[bool]=False, user_agents: Optional[List[str]]=None, retry_attempts: Optional[int]=None):
        if False:
            print('Hello World!')
        '\n\n        Creates a LinkContentFetcher instance.\n        :param content_handlers: A dictionary of content handlers to use for extracting content from a response.\n        :param processor: PreProcessor to apply to the extracted text\n        :param raise_on_failure: A boolean indicating whether to raise an exception when a failure occurs\n                         during content extraction. If False, the error is simply logged and the program continues.\n                         Defaults to False.\n        :param user_agents: A list of user agents to use when fetching content. Defaults to None.\n        :param retry_attempts: The number of times to retry fetching content. Defaults to 2.\n        '
        super().__init__()
        self.processor = processor
        self.raise_on_failure = raise_on_failure
        self.user_agents = user_agents or [LinkContentFetcher._USER_AGENT]
        self.current_user_agent_idx: int = 0
        self.retry_attempts = retry_attempts or 2
        self.handlers: Dict[str, Callable] = defaultdict(lambda : html_content_handler)
        self._register_content_handler('text/html', html_content_handler)
        if fitz_import.is_successful():
            self._register_content_handler('application/pdf', pdf_content_handler)
        if content_handlers:
            for (content_type, handler) in content_handlers.items():
                self._register_content_handler(content_type, handler)

    def fetch(self, url: str, timeout: Optional[int]=3, doc_kwargs: Optional[dict]=None) -> List[Document]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetches content from a URL and converts it into a list of Document objects. If no content is extracted,\n        an empty list is returned.\n\n        :param url: URL to fetch content from.\n        :param timeout: Timeout in seconds for the request.\n        :param doc_kwargs: Optional kwargs to pass to the Document constructor.\n        :return: List of Document objects or an empty list if no content is extracted.\n        '
        if not self._is_valid_url(url):
            raise InvalidURL('Invalid or missing URL: {}'.format(url))
        doc_kwargs = doc_kwargs or {}
        extracted_doc: Dict[str, Union[str, dict]] = {'meta': {'url': url, 'timestamp': int(datetime.utcnow().timestamp())}}
        extracted_doc.update(doc_kwargs)
        response = self._get_response(url, timeout=timeout or 3)
        has_content = response.status_code == HTTPStatus.OK and (response.text or response.content)
        fetched_documents = []
        if has_content:
            extracted_content: str = ''
            handler: Callable = self._get_content_type_handler(response.headers.get('Content-Type', ''))
            try:
                extracted_content = handler(response)
            except Exception as e:
                if self.raise_on_failure:
                    raise e
                logger.warning('failed to extract content from %s', response.url)
            content = extracted_content or extracted_doc.get('snippet_text', '')
            if not content:
                return []
            if extracted_content:
                logger.debug('%s handler extracted content from %s', handler, url)
            extracted_doc['content'] = content
        else:
            extracted_doc['content'] = extracted_doc.get('snippet_text', '')
        document = Document.from_dict(extracted_doc)
        fetched_documents = self.processor.process(documents=[document]) if self.processor else [document]
        return fetched_documents

    def run(self, query: Optional[str]=None, file_paths: Optional[List[str]]=None, labels: Optional[MultiLabel]=None, documents: Optional[List[Document]]=None, meta: Optional[dict]=None) -> Tuple[Dict, str]:
        if False:
            return 10
        '\n        Fetches content from a URL specified by query parameter and converts it into a list of Document objects.\n\n        param query: The query - a URL to fetch content from.\n        param file_paths: Not used.\n        param labels: Not used.\n        param documents: Not used.\n        param meta: Not used.\n\n        return: List of Document objects.\n        '
        if not query:
            raise ValueError('LinkContentFetcher run requires the `query` parameter')
        documents = self.fetch(url=query)
        return ({'documents': documents}, 'output_1')

    def run_batch(self, queries: Optional[Union[str, List[str]]]=None, file_paths: Optional[List[str]]=None, labels: Optional[Union[MultiLabel, List[MultiLabel]]]=None, documents: Optional[Union[List[Document], List[List[Document]]]]=None, meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]=None, params: Optional[dict]=None, debug: Optional[bool]=None):
        if False:
            print('Hello World!')
        '\n        Takes a list of queries, where each query is expected to be a URL. For each query, the method\n        fetches content from the specified URL and transforms it into a list of Document objects. The output is a list\n        of these document lists, where each individual list of Document objects corresponds to the content retrieved\n\n        param queries: List of queries - URLs to fetch content from.\n        param file_paths: Not used.\n        param labels: Not used.\n        param documents: Not used.\n        param meta: Not used.\n        param params: Not used.\n        param debug: Not used.\n\n        return: List of lists of Document objects.\n        '
        results = []
        if isinstance(queries, str):
            queries = [queries]
        elif not isinstance(queries, list):
            raise ValueError('LinkContentFetcher run_batch requires the `queries` parameter to be Union[str, List[str]]')
        for query in queries:
            results.append(self.fetch(url=query))
        return ({'documents': results}, 'output_1')

    def _register_content_handler(self, content_type: str, handler: Callable):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register a new content handler for a specific content type.\n        If a handler for the given content type already exists, it will be overridden.\n\n        :param content_type: The content type for which the handler should be used.\n        :param handler: The handler function. This function should accept a requests.Response object parameter,\n        and return the extracted text (or None).\n        '
        if not callable(handler):
            raise ValueError(f'handler must be a callable, but got {type(handler).__name__}')
        params = inspect.signature(handler).parameters
        if len(params) != 1 or list(params.keys()) != ['response']:
            raise ValueError(f"{content_type} handler must accept 'response: requests.Response' as a single parameter")
        self.handlers[content_type] = handler

    def _get_response(self, url: str, timeout: Optional[int]=None) -> requests.Response:
        if False:
            while True:
                i = 10
        '\n        Fetches content from a URL. Returns a response object.\n        :param url: The URL to fetch content from.\n        :param timeout: The timeout in seconds.\n        :return: A response object.\n        '

        @retry(reraise=True, stop=stop_after_attempt(self.retry_attempts), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type((HTTPError, requests.RequestException)), after=self._switch_user_agent)
        def _request():
            if False:
                i = 10
                return i + 15
            headers = self._REQUEST_HEADERS.copy()
            headers['User-Agent'] = self.user_agents[self.current_user_agent_idx]
            r = requests.get(url, headers=headers, timeout=timeout or 3)
            r.raise_for_status()
            return r
        try:
            response = _request()
        except Exception as e:
            if self.raise_on_failure:
                raise e
            logger.warning("Couldn't retrieve content from %s", url)
            response = requests.Response()
        finally:
            self.current_user_agent_idx = 0
        return response

    def _get_content_type_handler(self, content_type: str) -> Callable:
        if False:
            while True:
                i = 10
        '\n        Get the appropriate content handler based on the content type.\n        :param content_type: The content type of the response.\n        :return: The matching content handler callable or the default html_content_handler if no match is found.\n        '
        content_type_lookup: str = (content_type or '').split(';')[0]
        return self.handlers[content_type_lookup]

    def _switch_user_agent(self, retry_state: RetryCallState) -> None:
        if False:
            while True:
                i = 10
        '\n        Switches the User-Agent for this LinkContentRetriever to the next one in the list of user agents.\n        :param retry_state: The retry state (unused, required by tenacity).\n        '
        self.current_user_agent_idx = (self.current_user_agent_idx + 1) % len(self.user_agents)

    def _is_valid_url(self, url: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Checks if a URL is valid.\n\n        :param url: The URL to check.\n        :return: True if the URL is valid, False otherwise.\n        '
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])