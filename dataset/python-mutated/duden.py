"""
 Duden
"""
import re
from urllib.parse import quote, urljoin
from lxml import html
from searx.utils import extract_text, eval_xpath, eval_xpath_list, eval_xpath_getindex
from searx.network import raise_for_httperror
about = {'website': 'https://www.duden.de', 'wikidata_id': 'Q73624591', 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'HTML', 'language': 'de'}
categories = ['dictionaries']
paging = True
base_url = 'https://www.duden.de/'
search_url = base_url + 'suchen/dudenonline/{query}?search_api_fulltext=&page={offset}'

def request(query, params):
    if False:
        return 10
    "pre-request callback\n    params<dict>:\n      method  : POST/GET\n      headers : {}\n      data    : {} # if method == POST\n      url     : ''\n      category: 'search category'\n      pageno  : 1 # number of the requested page\n    "
    offset = params['pageno'] - 1
    if offset == 0:
        search_url_fmt = base_url + 'suchen/dudenonline/{query}'
        params['url'] = search_url_fmt.format(query=quote(query))
    else:
        params['url'] = search_url.format(offset=offset, query=quote(query))
    params['soft_max_redirects'] = 1
    params['raise_for_httperror'] = False
    return params

def response(resp):
    if False:
        while True:
            i = 10
    'post-response callback\n    resp: requests response object\n    '
    results = []
    if resp.status_code == 404:
        return results
    raise_for_httperror(resp)
    dom = html.fromstring(resp.text)
    number_of_results_element = eval_xpath_getindex(dom, '//a[@class="active" and contains(@href,"/suchen/dudenonline")]/span/text()', 0, default=None)
    if number_of_results_element is not None:
        number_of_results_string = re.sub('[^0-9]', '', number_of_results_element)
        results.append({'number_of_results': int(number_of_results_string)})
    for result in eval_xpath_list(dom, '//section[not(contains(@class, "essay"))]'):
        url = eval_xpath_getindex(result, './/h2/a', 0).get('href')
        url = urljoin(base_url, url)
        title = eval_xpath(result, 'string(.//h2/a)').strip()
        content = extract_text(eval_xpath(result, './/p'))
        results.append({'url': url, 'title': title, 'content': content})
    return results