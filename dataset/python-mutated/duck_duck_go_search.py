import json
import requests
from typing import Type, Optional, Union
import time
from superagi.helper.error_handler import ErrorHandler
from superagi.lib.logger import logger
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from itertools import islice
from superagi.helper.token_counter import TokenCounter
from superagi.llms.base_llm import BaseLlm
from superagi.models.agent_execution import AgentExecution
from superagi.models.agent_execution_feed import AgentExecutionFeed
from superagi.tools.base_tool import BaseTool
from superagi.helper.webpage_extractor import WebpageExtractor
DUCKDUCKGO_MAX_ATTEMPTS = 3
WEBPAGE_EXTRACTOR_MAX_ATTEMPTS = 2
MAX_LINKS_TO_SCRAPE = 3
NUM_RESULTS_TO_USE = 10

class DuckDuckGoSearchSchema(BaseModel):
    query: str = Field(..., description='The search query for duckduckgo search.')

class DuckDuckGoSearchTool(BaseTool):
    """
    Duck Duck Go Search tool

    Attributes:
        name : The name.
        description : The description.
        args_schema : The args schema.
    """
    llm: Optional[BaseLlm] = None
    name = 'DuckDuckGoSearch'
    agent_id: int = None
    agent_execution_id: int = None
    description = 'A tool for performing a DuckDuckGo search and extracting snippets and webpages.Input should be a search query.'
    args_schema: Type[DuckDuckGoSearchSchema] = DuckDuckGoSearchSchema

    class Config:
        arbitrary_types_allowed = True

    def _execute(self, query: str) -> tuple:
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute the DuckDuckGo search tool.\n\n        Args:\n            query : The query to search for.\n\n        Returns:\n            Search result summary along with related links\n        '
        search_results = self.get_raw_duckduckgo_results(query)
        links = []
        for result in search_results:
            links.append(result['href'])
        webpages = self.get_content_from_url(links)
        results = self.get_formatted_webpages(search_results, webpages)
        summary = self.summarise_result(query, results)
        links = [result['links'] for result in results if len(result['links']) > 0]
        if len(links) > 0:
            return summary + '\n\nLinks:\n' + '\n'.join(('- ' + link for link in links[:3]))
        return summary

    def get_formatted_webpages(self, search_results, webpages):
        if False:
            while True:
                i = 10
        '\n        Generate an array of formatted webpages which can be passed to the summarizer function (summarise_result).\n\n        Args:\n            search_results : The array of objects which were fetched by DuckDuckGo.\n\n        Returns:\n            Returns the result array which is an array of objects\n        '
        results = []
        i = 0
        for webpage in webpages:
            results.append({'title': search_results[i]['title'], 'body': webpage, 'links': search_results[i]['href']})
            i += 1
            if TokenCounter.count_text_tokens(json.dumps(results)) > 3000:
                break
        return results

    def get_content_from_url(self, links):
        if False:
            i = 10
            return i + 15
        '\n        Generates a webpage array which stores the content fetched from the links\n        Args:\n            links : The array of URLs which were fetched by DuckDuckGo.\n\n        Returns:\n            Returns a webpage array which stores the content fetched from the links\n        '
        webpages = []
        if links:
            for i in range(0, MAX_LINKS_TO_SCRAPE):
                time.sleep(3)
                content = WebpageExtractor().extract_with_bs4(links[i])
                max_length = len(' '.join(content.split(' ')[:500]))
                content = content[:max_length]
                attempts = 0
                while content == '' and attempts < WEBPAGE_EXTRACTOR_MAX_ATTEMPTS:
                    attempts += 1
                    content = WebpageExtractor().extract_with_bs4(links[i])
                    content = content[:max_length]
                webpages.append(content)
        return webpages

    def get_raw_duckduckgo_results(self, query):
        if False:
            return 10
        '\n        Gets raw search results from the duckduckgosearch python package\n        Args:\n            query : The query to search for.\n\n        Returns:\n            Returns raw search results from the duckduckgosearch python package\n        '
        search_results = []
        attempts = 0
        while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
            if not query:
                return json.dumps(search_results)
            results = DDGS().text(query)
            search_results = list(islice(results, NUM_RESULTS_TO_USE))
            if search_results:
                break
            attempts += 1
        return search_results

    def summarise_result(self, query, snippets):
        if False:
            print('Hello World!')
        '\n        Summarise the result of a DuckDuckGo search.\n\n        Args:\n            query : The query to search for.\n            snippets (list): A list of snippets from the search.\n\n        Returns:\n            A summary of the search result.\n        '
        summarize_prompt = 'Summarize the following text `{snippets}`\n            Write a concise or as descriptive as necessary and attempt to\n            answer the query: `{query}` as best as possible. Use markdown formatting for\n            longer responses.'
        summarize_prompt = summarize_prompt.replace('{snippets}', str(snippets))
        summarize_prompt = summarize_prompt.replace('{query}', query)
        messages = [{'role': 'system', 'content': summarize_prompt}]
        result = self.llm.chat_completion(messages, max_tokens=self.max_token_limit)
        if 'error' in result and result['message'] is not None:
            ErrorHandler.handle_openai_errors(self.toolkit_config.session, self.agent_id, self.agent_execution_id, result['message'])
        return result['content']