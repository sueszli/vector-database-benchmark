def rss_parser(soup):
    if False:
        while True:
            i = 10
    torrent_titles = []
    torrent_urls = []
    torrent_homepage = []
    for item in soup.findall('./channel/item'):
        torrent_titles.append(item.find('title').text)
        enclosure = item.find('enclosure')
        if enclosure is not None:
            torrent_homepage.append(item.find('link').text)
            torrent_urls.append(enclosure.attrib.get('url'))
        else:
            torrent_urls.append(item.find('link').text)
            torrent_homepage.append('')
    return (torrent_titles, torrent_urls, torrent_homepage)

def mikan_title(soup):
    if False:
        return 10
    return soup.find('title').text