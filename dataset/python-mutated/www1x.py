"""1x (Images)

"""
from urllib.parse import urlencode, urljoin
from lxml import html, etree
from searx.utils import extract_text, eval_xpath_list, eval_xpath_getindex
about = {'website': 'https://1x.com/', 'wikidata_id': None, 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['images']
paging = False
base_url = 'https://1x.com'
search_url = base_url + '/backend/search.php?{query}'
gallery_url = 'https://gallery.1x.com/'

def request(query, params):
    if False:
        for i in range(10):
            print('nop')
    params['url'] = search_url.format(query=urlencode({'q': query}))
    return params

def response(resp):
    if False:
        for i in range(10):
            print('nop')
    results = []
    xmldom = etree.fromstring(resp.content)
    xmlsearchresult = eval_xpath_getindex(xmldom, '//data', 0)
    dom = html.fragment_fromstring(xmlsearchresult.text, create_parent='div')
    for link in eval_xpath_list(dom, '//a'):
        url = urljoin(base_url, link.attrib.get('href'))
        title = extract_text(link)
        thumbnail_src = urljoin(gallery_url, eval_xpath_getindex(link, './/img', 0).attrib['src'].replace(base_url, ''))
        results.append({'url': url, 'title': title, 'img_src': thumbnail_src, 'content': '', 'thumbnail_src': thumbnail_src, 'template': 'images.html'})
    return results