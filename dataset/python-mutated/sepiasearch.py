"""SepiaSearch uses the same languages as :py:obj:`Peertube
<searx.engines.peertube>` and the response is identical to the response from the
peertube engines.

"""
from typing import TYPE_CHECKING
from urllib.parse import urlencode
from datetime import datetime
from searx.engines.peertube import fetch_traits
from searx.engines.peertube import video_response, safesearch_table, time_range_table
from searx.enginelib.traits import EngineTraits
if TYPE_CHECKING:
    import logging
    logger: logging.Logger
traits: EngineTraits
about = {'website': 'https://sepiasearch.org', 'wikidata_id': None, 'official_api_documentation': 'https://docs.joinpeertube.org/api-rest-reference.html#tag/Search/operation/searchVideos', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
categories = ['videos']
paging = True
base_url = 'https://sepiasearch.org'
time_range_support = True
safesearch = True

def request(query, params):
    if False:
        for i in range(10):
            print('nop')
    'Assemble request for the SepiaSearch API'
    if not query:
        return False
    eng_lang = traits.get_language(params['searxng_locale'], None)
    params['url'] = base_url.rstrip('/') + '/api/v1/search/videos?' + urlencode({'search': query, 'start': (params['pageno'] - 1) * 10, 'count': 10, 'sort': '-match', 'nsfw': safesearch_table[params['safesearch']]})
    if eng_lang is not None:
        params['url'] += '&languageOneOf[]=' + eng_lang
        params['url'] += '&boostLanguages[]=' + eng_lang
    if params['time_range'] in time_range_table:
        time = datetime.now().date() + time_range_table[params['time_range']]
        params['url'] += '&startDate=' + time.isoformat()
    return params

def response(resp):
    if False:
        i = 10
        return i + 15
    return video_response(resp)