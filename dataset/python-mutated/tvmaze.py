"""
  TVmaze
  Show Search
"""
from urllib.parse import urlencode
from json import loads
from searx.utils import html_to_text
about = {'website': 'https://www.tvmaze.com/', 'wikidata_id': 'Q84863617', 'official_api_documentation': 'https://www.tvmaze.com/api', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
categories = ['general']
paging = False
base_url = 'https://api.tvmaze.com/search/'
search_string = 'shows?{query}'

def request(query, params):
    if False:
        i = 10
        return i + 15
    search = search_string.format(query=urlencode({'q': query}))
    params['url'] = base_url + search
    return params

def response(resp):
    if False:
        for i in range(10):
            print('nop')
    results = []
    search_res = loads(resp.text)
    for result in search_res:
        res = result['show']
        results.append({'url': res['url'], 'title': res['name'], 'content': html_to_text(res['summary'] or '')})
    return results