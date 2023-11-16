"""
 Searchcode (IT)
"""
from json import loads
from urllib.parse import urlencode
about = {'website': 'https://searchcode.com/', 'wikidata_id': None, 'official_api_documentation': 'https://searchcode.com/api/', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
categories = ['it']
paging = True
url = 'https://searchcode.com/'
search_url = url + 'api/codesearch_I/?{query}&p={pageno}'
code_endings = {'cs': 'c#', 'h': 'c', 'hpp': 'cpp', 'cxx': 'cpp'}

def request(query, params):
    if False:
        while True:
            i = 10
    params['url'] = search_url.format(query=urlencode({'q': query}), pageno=params['pageno'] - 1)
    return params

def response(resp):
    if False:
        print('Hello World!')
    results = []
    search_results = loads(resp.text)
    for result in search_results.get('results', []):
        href = result['url']
        title = '' + result['name'] + ' - ' + result['filename']
        repo = result['repo']
        lines = dict()
        for (line, code) in result['lines'].items():
            lines[int(line)] = code
        code_language = code_endings.get(result['filename'].split('.')[-1].lower(), result['filename'].split('.')[-1].lower())
        results.append({'url': href, 'title': title, 'content': '', 'repository': repo, 'codelines': sorted(lines.items()), 'code_language': code_language, 'template': 'code.html'})
    return results