"""
 Nyaa.si (Anime Bittorrent tracker)
"""
from lxml import html
from urllib.parse import urlencode
from searx.utils import extract_text, get_torrent_size, int_or_zero
about = {'website': 'https://nyaa.si/', 'wikidata_id': None, 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['files']
paging = True
base_url = 'https://nyaa.si/'
search_url = base_url + '?page=search&{query}&offset={offset}'
xpath_results = '//table[contains(@class, "torrent-list")]//tr[not(th)]'
xpath_category = './/td[1]/a[1]'
xpath_title = './/td[2]/a[last()]'
xpath_torrent_links = './/td[3]/a'
xpath_filesize = './/td[4]/text()'
xpath_seeds = './/td[6]/text()'
xpath_leeches = './/td[7]/text()'
xpath_downloads = './/td[8]/text()'

def request(query, params):
    if False:
        return 10
    query = urlencode({'term': query})
    params['url'] = search_url.format(query=query, offset=params['pageno'])
    return params

def response(resp):
    if False:
        return 10
    results = []
    dom = html.fromstring(resp.text)
    for result in dom.xpath(xpath_results):
        filesize = 0
        magnet_link = ''
        torrent_link = ''
        try:
            category = result.xpath(xpath_category)[0].attrib.get('title')
        except:
            pass
        page_a = result.xpath(xpath_title)[0]
        title = extract_text(page_a)
        href = base_url + page_a.attrib.get('href')
        for link in result.xpath(xpath_torrent_links):
            url = link.attrib.get('href')
            if 'magnet' in url:
                magnet_link = url
            else:
                torrent_link = url
        seed = int_or_zero(result.xpath(xpath_seeds))
        leech = int_or_zero(result.xpath(xpath_leeches))
        downloads = int_or_zero(result.xpath(xpath_downloads))
        try:
            filesize_info = result.xpath(xpath_filesize)[0]
            (filesize, filesize_multiplier) = filesize_info.split()
            filesize = get_torrent_size(filesize, filesize_multiplier)
        except:
            pass
        content = 'Category: "{category}". Downloaded {downloads} times.'
        content = content.format(category=category, downloads=downloads)
        results.append({'url': href, 'title': title, 'content': content, 'seed': seed, 'leech': leech, 'filesize': filesize, 'torrentfile': torrent_link, 'magnetlink': magnet_link, 'template': 'torrent.html'})
    return results