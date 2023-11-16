"""
 ScanR Structures (Science)
"""
from json import loads, dumps
from searx.utils import html_to_text
about = {'website': 'https://scanr.enseignementsup-recherche.gouv.fr', 'wikidata_id': 'Q44105684', 'official_api_documentation': 'https://scanr.enseignementsup-recherche.gouv.fr/opendata', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
categories = ['science']
paging = True
page_size = 20
url = 'https://scanr.enseignementsup-recherche.gouv.fr/'
search_url = url + 'api/structures/search'

def request(query, params):
    if False:
        return 10
    params['url'] = search_url
    params['method'] = 'POST'
    params['headers']['Content-type'] = 'application/json'
    params['data'] = dumps({'query': query, 'searchField': 'ALL', 'sortDirection': 'ASC', 'sortOrder': 'RELEVANCY', 'page': params['pageno'], 'pageSize': page_size})
    return params

def response(resp):
    if False:
        for i in range(10):
            print('nop')
    results = []
    search_res = loads(resp.text)
    if search_res.get('total', 0) < 1:
        return []
    for result in search_res['results']:
        if 'id' not in result:
            continue
        thumbnail = None
        if 'logo' in result:
            thumbnail = result['logo']
            if thumbnail[0] == '/':
                thumbnail = url + thumbnail
        content = None
        if 'highlights' in result:
            content = result['highlights'][0]['value']
        results.append({'url': url + 'structure/' + result['id'], 'title': result['label'], 'img_src': thumbnail, 'content': html_to_text(content)})
    return results