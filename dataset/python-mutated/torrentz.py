"""
 Torrentz2.is (BitTorrent meta-search engine)
"""
import re
from urllib.parse import urlencode
from lxml import html
from datetime import datetime
from searx.utils import extract_text, get_torrent_size
about = {'website': 'https://torrentz2.is/', 'wikidata_id': 'Q1156687', 'official_api_documentation': 'https://torrentz.is/torrentz.btsearch', 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['files', 'videos', 'music']
paging = True
base_url = 'https://torrentz2.is/'
search_url = base_url + 'search?{query}'

def request(query, params):
    if False:
        i = 10
        return i + 15
    page = params['pageno'] - 1
    query = urlencode({'f': query, 'p': page})
    params['url'] = search_url.format(query=query)
    return params

def response(resp):
    if False:
        print('Hello World!')
    results = []
    dom = html.fromstring(resp.text)
    for result in dom.xpath('//div[@class="results"]/dl'):
        name_cell = result.xpath('./dt')[0]
        title = extract_text(name_cell)
        links = name_cell.xpath('./a')
        if len(links) != 1:
            continue
        link = links[0].attrib.get('href').lstrip('/')
        seed = 0
        leech = 0
        try:
            seed = int(result.xpath('./dd/span[4]/text()')[0].replace(',', ''))
            leech = int(result.xpath('./dd/span[5]/text()')[0].replace(',', ''))
        except:
            pass
        params = {'url': base_url + link, 'title': title, 'seed': seed, 'leech': leech, 'template': 'torrent.html'}
        try:
            filesize_info = result.xpath('./dd/span[3]/text()')[0]
            (filesize, filesize_multiplier) = filesize_info.split()
            filesize = get_torrent_size(filesize, filesize_multiplier)
            params['filesize'] = filesize
        except:
            pass
        if re.compile('[0-9a-fA-F]{40}').match(link):
            params['magnetlink'] = 'magnet:?xt=urn:btih:' + link
        try:
            date_ts = result.xpath('./dd/span[2]')[0].attrib.get('title')
            date = datetime.fromtimestamp(float(date_ts))
            params['publishedDate'] = date
        except:
            pass
        results.append(params)
    return results