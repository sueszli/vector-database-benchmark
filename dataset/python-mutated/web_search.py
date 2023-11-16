"""Commands to search the web with"""
from __future__ import annotations
COMMAND_CATEGORY = 'web_search'
COMMAND_CATEGORY_TITLE = 'Web Search'
import json
import time
from itertools import islice
from duckduckgo_search import DDGS
from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import ConfigurationError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
DUCKDUCKGO_MAX_ATTEMPTS = 3

@command('web_search', 'Searches the web', {'query': JSONSchema(type=JSONSchema.Type.STRING, description='The search query', required=True)}, aliases=['search'])
def web_search(query: str, agent: Agent, num_results: int=8) -> str:
    if False:
        print('Hello World!')
    'Return the results of a Google search\n\n    Args:\n        query (str): The search query.\n        num_results (int): The number of results to return.\n\n    Returns:\n        str: The results of the search.\n    '
    search_results = []
    attempts = 0
    while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
        if not query:
            return json.dumps(search_results)
        results = DDGS().text(query)
        search_results = list(islice(results, num_results))
        if search_results:
            break
        time.sleep(1)
        attempts += 1
    search_results = [{'title': r['title'], 'url': r['href'], **({'exerpt': r['body']} if r.get('body') else {})} for r in search_results]
    results = '## Search results\n' + '\n\n'.join((f'''### "{r['title']}"\n**URL:** {r['url']}  \n**Excerpt:** ''' + (f'"{exerpt}"' if (exerpt := r.get('exerpt')) else 'N/A') for r in search_results))
    return safe_google_results(results)

@command('google', 'Google Search', {'query': JSONSchema(type=JSONSchema.Type.STRING, description='The search query', required=True)}, lambda config: bool(config.google_api_key) and bool(config.google_custom_search_engine_id), 'Configure google_api_key and custom_search_engine_id.', aliases=['search'])
def google(query: str, agent: Agent, num_results: int=8) -> str | list[str]:
    if False:
        while True:
            i = 10
    'Return the results of a Google search using the official Google API\n\n    Args:\n        query (str): The search query.\n        num_results (int): The number of results to return.\n\n    Returns:\n        str: The results of the search.\n    '
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    try:
        api_key = agent.legacy_config.google_api_key
        custom_search_engine_id = agent.legacy_config.google_custom_search_engine_id
        service = build('customsearch', 'v1', developerKey=api_key)
        result = service.cse().list(q=query, cx=custom_search_engine_id, num=num_results).execute()
        search_results = result.get('items', [])
        search_results_links = [item['link'] for item in search_results]
    except HttpError as e:
        error_details = json.loads(e.content.decode())
        if error_details.get('error', {}).get('code') == 403 and 'invalid API key' in error_details.get('error', {}).get('message', ''):
            raise ConfigurationError('The provided Google API key is invalid or missing.')
        raise
    return safe_google_results(search_results_links)

def safe_google_results(results: str | list) -> str:
    if False:
        print('Hello World!')
    '\n        Return the results of a Google search in a safe format.\n\n    Args:\n        results (str | list): The search results.\n\n    Returns:\n        str: The results of the search.\n    '
    if isinstance(results, list):
        safe_message = json.dumps([result.encode('utf-8', 'ignore').decode('utf-8') for result in results])
    else:
        safe_message = results.encode('utf-8', 'ignore').decode('utf-8')
    return safe_message