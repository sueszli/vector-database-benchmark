"""
 Doku Wiki
"""
from urllib.parse import urlencode
from lxml.html import fromstring
from searx.utils import extract_text, eval_xpath
about = {'website': 'https://www.dokuwiki.org/', 'wikidata_id': 'Q851864', 'official_api_documentation': 'https://www.dokuwiki.org/devel:xmlrpc', 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['general']
paging = False
number_of_results = 5
base_url = 'http://localhost:8090'
search_url = '/?do=search&{query}'

def request(query, params):
    if False:
        while True:
            i = 10
    params['url'] = base_url + search_url.format(query=urlencode({'id': query}))
    return params

def response(resp):
    if False:
        return 10
    results = []
    doc = fromstring(resp.text)
    for r in eval_xpath(doc, '//div[@class="search_quickresult"]/ul/li'):
        try:
            res_url = eval_xpath(r, './/a[@class="wikilink1"]/@href')[-1]
        except:
            continue
        if not res_url:
            continue
        title = extract_text(eval_xpath(r, './/a[@class="wikilink1"]/@title'))
        results.append({'title': title, 'content': '', 'url': base_url + res_url})
    for r in eval_xpath(doc, '//dl[@class="search_results"]/*'):
        try:
            if r.tag == 'dt':
                res_url = eval_xpath(r, './/a[@class="wikilink1"]/@href')[-1]
                title = extract_text(eval_xpath(r, './/a[@class="wikilink1"]/@title'))
            elif r.tag == 'dd':
                content = extract_text(eval_xpath(r, '.'))
                results.append({'title': title, 'content': content, 'url': base_url + res_url})
        except:
            continue
        if not res_url:
            continue
    return results