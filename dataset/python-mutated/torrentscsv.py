import json
from urllib.parse import urlencode
from novaprinter import prettyPrinter
from helpers import retrieve_url

class torrentscsv(object):
    url = 'https://torrents-csv.ml'
    name = 'torrents-csv'
    supported_categories = {'all': ''}
    trackers_list = ['udp://tracker.internetwarriors.net:1337/announce', 'udp://tracker.opentrackr.org:1337/announce', 'udp://p4p.arenabg.ch:1337/announce', 'udp://tracker.openbittorrent.com:6969/announce', 'udp://www.torrent.eu.org:451/announce', 'udp://tracker.torrent.eu.org:451/announce', 'udp://retracker.lanta-net.ru:2710/announce', 'udp://open.stealth.si:80/announce', 'udp://exodus.desync.com:6969/announce', 'udp://tracker.tiny-vps.com:6969/announce']
    trackers = '&'.join((urlencode({'tr': tracker}) for tracker in trackers_list))

    def search(self, what, cat='all'):
        if False:
            print('Hello World!')
        search_url = '{}/service/search?size=300&q={}'.format(self.url, what)
        desc_url = '{}/#/search/torrent/{}/1'.format(self.url, what)
        response = retrieve_url(search_url)
        response_json = json.loads(response)
        for result in response_json:
            res = {'link': self.download_link(result), 'name': result['name'], 'size': str(result['size_bytes']) + ' B', 'seeds': result['seeders'], 'leech': result['leechers'], 'engine_url': self.url, 'desc_link': desc_url}
            prettyPrinter(res)

    def download_link(self, result):
        if False:
            print('Hello World!')
        return 'magnet:?xt=urn:btih:{}&{}&{}'.format(result['infohash'], urlencode({'dn': result['name']}), self.trackers)