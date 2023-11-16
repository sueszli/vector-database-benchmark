"""
 FramaLibre (It)
"""
from html import escape
from urllib.parse import urljoin, urlencode
from lxml import html
from searx.utils import extract_text
about = {'website': 'https://framalibre.org/', 'wikidata_id': 'Q30213882', 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['it']
paging = True
base_url = 'https://framalibre.org/'
search_url = base_url + 'recherche-par-crit-res?{query}&page={offset}'
results_xpath = '//div[@class="nodes-list-row"]/div[contains(@typeof,"sioc:Item")]'
link_xpath = './/h3[@class="node-title"]/a[@href]'
thumbnail_xpath = './/img[@class="media-object img-responsive"]/@src'
content_xpath = './/div[@class="content"]//p'

def request(query, params):
    if False:
        while True:
            i = 10
    offset = params['pageno'] - 1
    params['url'] = search_url.format(query=urlencode({'keys': query}), offset=offset)
    return params

def response(resp):
    if False:
        i = 10
        return i + 15
    results = []
    dom = html.fromstring(resp.text)
    for result in dom.xpath(results_xpath):
        link = result.xpath(link_xpath)[0]
        href = urljoin(base_url, link.attrib.get('href'))
        title = escape(extract_text(link))
        thumbnail_tags = result.xpath(thumbnail_xpath)
        thumbnail = None
        if len(thumbnail_tags) > 0:
            thumbnail = extract_text(thumbnail_tags[0])
            if thumbnail[0] == '/':
                thumbnail = base_url + thumbnail
        content = escape(extract_text(result.xpath(content_xpath)))
        results.append({'url': href, 'title': title, 'img_src': thumbnail, 'content': content})
    return results