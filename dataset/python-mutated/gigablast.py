"""
 Gigablast (Web)
"""
import re
from json import loads, JSONDecodeError
from urllib.parse import urlencode
from searx.exceptions import SearxEngineResponseException
from searx.poolrequests import get
about = {'website': 'https://www.gigablast.com', 'wikidata_id': 'Q3105449', 'official_api_documentation': 'https://gigablast.com/api.html', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
categories = ['general']
collections = 'main'
search_type = ''
fast = 0
paging = False
safesearch = True
base_url = 'https://gigablast.com'
extra_param = ''
extra_param_path = '/search?c=main&qlangcountry=en-us&q=south&s=10'
_wait_for_results_msg = 'Loading results takes too long. Please enable fast option in gigablast engine.'

def parse_extra_param(text):
    if False:
        i = 10
        return i + 15
    global extra_param
    re_var = None
    for line in text.splitlines():
        if re_var is None and extra_param_path in line:
            var = line.split('=')[0].split()[1]
            re_var = re.compile(var + '\\s*=\\s*' + var + "\\s*\\+\\s*'" + '(.*)' + "'(.*)")
            extra_param = line.split("'")[1][len(extra_param_path):]
            continue
        if re_var is not None and re_var.search(line):
            extra_param += re_var.search(line).group(1)
            break

def init(engine_settings=None):
    if False:
        print('Hello World!')
    parse_extra_param(get(base_url + extra_param_path).text)

def request(query, params):
    if False:
        while True:
            i = 10
    query_args = {'c': collections, 'format': 'json', 'q': query, 'dr': 1, 'showgoodimages': 0, 'fast': fast}
    if search_type != '':
        query_args['searchtype'] = search_type
    if params['language'] and params['language'] != 'all':
        query_args['qlangcountry'] = params['language']
        query_args['qlang'] = params['language'].split('-')[0]
    if params['safesearch'] >= 1:
        query_args['ff'] = 1
    search_url = '/search?' + urlencode(query_args)
    params['url'] = base_url + search_url + extra_param
    return params

def response(resp):
    if False:
        return 10
    results = []
    try:
        response_json = loads(resp.text)
    except JSONDecodeError as e:
        if 'Waiting for results' in resp.text:
            raise SearxEngineResponseException(message=_wait_for_results_msg)
        raise e
    for result in response_json['results']:
        title = result.get('title')
        if len(title) < 2:
            continue
        url = result.get('url')
        if len(url) < 9:
            continue
        content = result.get('sum')
        if len(content) < 5:
            continue
        subtitle = result.get('title')
        if len(subtitle) > 3 and subtitle != title:
            title += ' - ' + subtitle
        results.append(dict(url=url, title=title, content=content))
    return results