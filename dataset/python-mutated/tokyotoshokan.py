"""
 Tokyo Toshokan (A BitTorrent Library for Japanese Media)
"""
import re
from urllib.parse import urlencode
from lxml import html
from datetime import datetime
from searx.utils import extract_text, get_torrent_size, int_or_zero
about = {'website': 'https://www.tokyotosho.info/', 'wikidata_id': None, 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['files']
paging = True
base_url = 'https://www.tokyotosho.info/'
search_url = base_url + 'search.php?{query}'

def request(query, params):
    if False:
        print('Hello World!')
    query = urlencode({'page': params['pageno'], 'terms': query})
    params['url'] = search_url.format(query=query)
    return params

def response(resp):
    if False:
        for i in range(10):
            print('nop')
    results = []
    dom = html.fromstring(resp.text)
    rows = dom.xpath('//table[@class="listing"]//tr[contains(@class, "category_0")]')
    if len(rows) == 0 or len(rows) % 2 != 0:
        return []
    size_re = re.compile('Size:\\s*([\\d.]+)(TB|GB|MB|B)', re.IGNORECASE)
    for i in range(0, len(rows), 2):
        name_row = rows[i]
        links = name_row.xpath('./td[@class="desc-top"]/a')
        params = {'template': 'torrent.html', 'url': links[-1].attrib.get('href'), 'title': extract_text(links[-1])}
        if len(links) == 2:
            magnet = links[0].attrib.get('href')
            if magnet.startswith('magnet'):
                params['magnetlink'] = magnet
        info_row = rows[i + 1]
        desc = extract_text(info_row.xpath('./td[@class="desc-bot"]')[0])
        for item in desc.split('|'):
            item = item.strip()
            if item.startswith('Size:'):
                try:
                    groups = size_re.match(item).groups()
                    params['filesize'] = get_torrent_size(groups[0], groups[1])
                except:
                    pass
            elif item.startswith('Date:'):
                try:
                    date = datetime.strptime(item, 'Date: %Y-%m-%d %H:%M UTC')
                    params['publishedDate'] = date
                except:
                    pass
            elif item.startswith('Comment:'):
                params['content'] = item
        stats = info_row.xpath('./td[@class="stats"]/span')
        if len(stats) == 3:
            params['seed'] = int_or_zero(extract_text(stats[0]))
            params['leech'] = int_or_zero(extract_text(stats[1]))
        results.append(params)
    return results