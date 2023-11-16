"""
 Omnom (General)
"""
from json import loads
from urllib.parse import urlencode
about = {'website': 'https://github.com/asciimoo/omnom', 'wikidata_id': None, 'official_api_documentation': 'http://your.omnom.host/api', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
categories = ['general']
paging = True
base_url = None
search_path = 'bookmarks?{query}&pageno={pageno}&format=json'
bookmark_path = 'bookmark?id='

def request(query, params):
    if False:
        while True:
            i = 10
    params['url'] = base_url + search_path.format(query=urlencode({'query': query}), pageno=params['pageno'])
    return params

def response(resp):
    if False:
        print('Hello World!')
    results = []
    json = loads(resp.text)
    for r in json.get('Bookmarks', {}):
        content = r['url']
        if r.get('notes'):
            content += ' - ' + r['notes']
        results.append({'title': r['title'], 'content': content, 'url': base_url + bookmark_path + str(r['id'])})
    return results