"""
 Yggtorrent (Videos, Music, Files)
"""
from lxml import html
from operator import itemgetter
from datetime import datetime
from urllib.parse import quote
from searx.utils import extract_text, get_torrent_size
from searx.poolrequests import get as http_get
about = {'website': 'https://www4.yggtorrent.li/', 'wikidata_id': None, 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['videos', 'music', 'files']
paging = True
url = 'https://www4.yggtorrent.li/'
search_url = url + 'engine/search?name={search_term}&do=search&page={pageno}&category={search_type}'
search_types = {'files': 'all', 'music': '2139', 'videos': '2145'}
cookies = dict()

def init(engine_settings=None):
    if False:
        return 10
    global cookies
    resp = http_get(url)
    if resp.ok:
        for r in resp.history:
            cookies.update(r.cookies)
        cookies.update(resp.cookies)

def request(query, params):
    if False:
        i = 10
        return i + 15
    search_type = search_types.get(params['category'], 'all')
    pageno = (params['pageno'] - 1) * 50
    params['url'] = search_url.format(search_term=quote(query), search_type=search_type, pageno=pageno)
    params['cookies'] = cookies
    return params

def response(resp):
    if False:
        i = 10
        return i + 15
    results = []
    dom = html.fromstring(resp.text)
    search_res = dom.xpath('//section[@id="#torrents"]/div/table/tbody/tr')
    if not search_res:
        return []
    for result in search_res:
        link = result.xpath('.//a[@id="torrent_name"]')[0]
        href = link.attrib.get('href')
        title = extract_text(link)
        seed = result.xpath('.//td[8]/text()')[0]
        leech = result.xpath('.//td[9]/text()')[0]
        if seed.isdigit():
            seed = int(seed)
        else:
            seed = 0
        if leech.isdigit():
            leech = int(leech)
        else:
            leech = 0
        params = {'url': href, 'title': title, 'seed': seed, 'leech': leech, 'template': 'torrent.html'}
        try:
            filesize_info = result.xpath('.//td[6]/text()')[0]
            filesize = filesize_info[:-2]
            filesize_multiplier = filesize_info[-2:].lower()
            multiplier_french_to_english = {'to': 'TiB', 'go': 'GiB', 'mo': 'MiB', 'ko': 'KiB'}
            filesize = get_torrent_size(filesize, multiplier_french_to_english[filesize_multiplier])
            params['filesize'] = filesize
        except:
            pass
        try:
            date_ts = result.xpath('.//td[5]/div/text()')[0]
            date = datetime.fromtimestamp(float(date_ts))
            params['publishedDate'] = date
        except:
            pass
        results.append(params)
    return sorted(results, key=itemgetter('seed'), reverse=True)