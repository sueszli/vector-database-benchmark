import json
import os
import logging
from typing import Dict, List, Optional, Any
import requests
from haystack.preview import Document, component, default_to_dict, ComponentError
logger = logging.getLogger(__name__)
SERPERDEV_BASE_URL = 'https://google.serper.dev/search'

class SerperDevError(ComponentError):
    ...

@component
class SerperDevWebSearch:
    """
    Search engine using SerperDev API. Given a query, it returns a list of URLs that are the most relevant.

    See the [Serper Dev website](https://serper.dev/) for more details.
    """

    def __init__(self, api_key: Optional[str]=None, top_k: Optional[int]=10, allowed_domains: Optional[List[str]]=None, search_params: Optional[Dict[str, Any]]=None):
        if False:
            while True:
                i = 10
        "\n        :param api_key: API key for the SerperDev API.  It can be\n        explicitly provided or automatically read from the\n        environment variable SERPERDEV_API_KEY (recommended).\n        :param top_k: Number of documents to return.\n        :param allowed_domains: List of domains to limit the search to.\n        :param search_params: Additional parameters passed to the SerperDev API.\n        For example, you can set 'num' to 20 to increase the number of search results.\n        See the [Serper Dev website](https://serper.dev/) for more details.\n        "
        if api_key is None:
            try:
                api_key = os.environ['SERPERDEV_API_KEY']
            except KeyError as e:
                raise ValueError('SerperDevWebSearch expects an API key. Set the SERPERDEV_API_KEY environment variable (recommended) or pass it explicitly.') from e
            raise ValueError('API key for SerperDev API must be set.')
        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.search_params = search_params or {}

    def to_dict(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Serialize this component to a dictionary.\n        '
        return default_to_dict(self, top_k=self.top_k, allowed_domains=self.allowed_domains, search_params=self.search_params)

    @component.output_types(documents=List[Document], links=List[str])
    def run(self, query: str):
        if False:
            while True:
                i = 10
        '\n        Search the SerperDev API for the given query and return the results as a list of Documents and a list of links.\n\n        :param query: Query string.\n        '
        query_prepend = 'OR '.join((f'site:{domain} ' for domain in self.allowed_domains)) if self.allowed_domains else ''
        payload = json.dumps({'q': query_prepend + query, 'gl': 'us', 'hl': 'en', 'autocorrect': True, **self.search_params})
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
        try:
            response = requests.post(SERPERDEV_BASE_URL, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
        except requests.Timeout:
            raise TimeoutError(f'Request to {self.__class__.__name__} timed out.')
        except requests.RequestException as e:
            raise SerperDevError(f'An error occurred while querying {self.__class__.__name__}. Error: {e}') from e
        json_result = response.json()
        organic = [Document(meta={k: v for (k, v) in d.items() if k != 'snippet'}, content=d['snippet']) for d in json_result['organic']]
        answer_box = []
        if 'answerBox' in json_result:
            answer_dict = json_result['answerBox']
            highlighted_answers = answer_dict.get('snippetHighlighted')
            answer_box_content = None
            if isinstance(highlighted_answers, list) and len(highlighted_answers) > 0:
                answer_box_content = highlighted_answers[0]
            elif isinstance(highlighted_answers, str):
                answer_box_content = highlighted_answers
            if not answer_box_content:
                for key in ['snippet', 'answer', 'title']:
                    if key in answer_dict:
                        answer_box_content = answer_dict[key]
                        break
            if answer_box_content:
                answer_box = [Document(content=answer_box_content, meta={'title': answer_dict.get('title', ''), 'link': answer_dict.get('link', '')})]
        people_also_ask = []
        if 'peopleAlsoAsk' in json_result:
            for result in json_result['peopleAlsoAsk']:
                title = result.get('title', '')
                people_also_ask.append(Document(content=result['snippet'] if result.get('snippet') else title, meta={'title': title, 'link': result.get('link', None)}))
        documents = answer_box + organic + people_also_ask
        links = [result['link'] for result in json_result['organic']]
        logger.debug("Serper Dev returned %s documents for the query '%s'", len(documents), query)
        return {'documents': documents[:self.top_k], 'links': links[:self.top_k]}