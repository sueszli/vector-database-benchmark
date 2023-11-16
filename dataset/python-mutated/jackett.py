import json
import os
import xml.etree.ElementTree
from urllib.parse import urlencode, unquote
from urllib import request as urllib_request
from http.cookiejar import CookieJar
from multiprocessing.dummy import Pool
from threading import Lock
from novaprinter import prettyPrinter
from helpers import download_file
CONFIG_FILE = 'jackett.json'
CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), CONFIG_FILE)
CONFIG_DATA = {'api_key': 'YOUR_API_KEY_HERE', 'url': 'http://127.0.0.1:9117', 'tracker_first': False, 'thread_count': 20}
PRINTER_THREAD_LOCK = Lock()

def load_configuration():
    if False:
        for i in range(10):
            print('nop')
    global CONFIG_PATH, CONFIG_DATA
    try:
        with open(CONFIG_PATH) as f:
            CONFIG_DATA = json.load(f)
    except ValueError:
        CONFIG_DATA['malformed'] = True
    except Exception:
        save_configuration()
    if any((item not in CONFIG_DATA for item in ['api_key', 'tracker_first', 'url'])):
        CONFIG_DATA['malformed'] = True
    if 'thread_count' not in CONFIG_DATA:
        CONFIG_DATA['thread_count'] = 20
        save_configuration()

def save_configuration():
    if False:
        print('Hello World!')
    global CONFIG_PATH, CONFIG_DATA
    with open(CONFIG_PATH, 'w') as f:
        f.write(json.dumps(CONFIG_DATA, indent=4, sort_keys=True))
load_configuration()

class jackett(object):
    name = 'Jackett'
    url = CONFIG_DATA['url'] if CONFIG_DATA['url'][-1] != '/' else CONFIG_DATA['url'][:-1]
    api_key = CONFIG_DATA['api_key']
    thread_count = CONFIG_DATA['thread_count']
    supported_categories = {'all': None, 'anime': ['5070'], 'books': ['8000'], 'games': ['1000', '4000'], 'movies': ['2000'], 'music': ['3000'], 'software': ['4000'], 'tv': ['5000']}

    def download_torrent(self, download_url):
        if False:
            i = 10
            return i + 15
        if download_url.startswith('magnet:?'):
            print(download_url + ' ' + download_url)
        response = self.get_response(download_url)
        if response is not None and response.startswith('magnet:?'):
            print(response + ' ' + download_url)
        else:
            print(download_file(download_url))

    def search(self, what, cat='all'):
        if False:
            print('Hello World!')
        what = unquote(what)
        category = self.supported_categories[cat.lower()]
        if 'malformed' in CONFIG_DATA:
            self.handle_error('malformed configuration file', what)
            return
        if self.api_key == 'YOUR_API_KEY_HERE':
            self.handle_error('api key error', what)
            return
        if self.thread_count > 1:
            args = []
            indexers = self.get_jackett_indexers(what)
            for indexer in indexers:
                args.append((what, category, indexer))
            with Pool(min(len(indexers), self.thread_count)) as pool:
                pool.starmap(self.search_jackett_indexer, args)
        else:
            self.search_jackett_indexer(what, category, 'all')

    def get_jackett_indexers(self, what):
        if False:
            while True:
                i = 10
        params = [('apikey', self.api_key), ('t', 'indexers'), ('configured', 'true')]
        params = urlencode(params)
        jacket_url = self.url + '/api/v2.0/indexers/all/results/torznab/api?%s' % params
        response = self.get_response(jacket_url)
        if response is None:
            self.handle_error('connection error getting indexer list', what)
            return
        response_xml = xml.etree.ElementTree.fromstring(response)
        indexers = []
        for indexer in response_xml.findall('indexer'):
            indexers.append(indexer.attrib['id'])
        return indexers

    def search_jackett_indexer(self, what, category, indexer_id):
        if False:
            for i in range(10):
                print('nop')
        params = [('apikey', self.api_key), ('q', what)]
        if category is not None:
            params.append(('cat', ','.join(category)))
        params = urlencode(params)
        jacket_url = self.url + '/api/v2.0/indexers/' + indexer_id + '/results/torznab/api?%s' % params
        response = self.get_response(jacket_url)
        if response is None:
            self.handle_error('connection error for indexer: ' + indexer_id, what)
            return
        response_xml = xml.etree.ElementTree.fromstring(response)
        for result in response_xml.find('channel').findall('item'):
            res = {}
            title = result.find('title')
            if title is not None:
                title = title.text
            else:
                continue
            tracker = result.find('jackettindexer')
            tracker = '' if tracker is None else tracker.text
            if CONFIG_DATA['tracker_first']:
                res['name'] = '[%s] %s' % (tracker, title)
            else:
                res['name'] = '%s [%s]' % (title, tracker)
            res['link'] = result.find(self.generate_xpath('magneturl'))
            if res['link'] is not None:
                res['link'] = res['link'].attrib['value']
            else:
                res['link'] = result.find('link')
                if res['link'] is not None:
                    res['link'] = res['link'].text
                else:
                    continue
            res['size'] = result.find('size')
            res['size'] = -1 if res['size'] is None else res['size'].text + ' B'
            res['seeds'] = result.find(self.generate_xpath('seeders'))
            res['seeds'] = -1 if res['seeds'] is None else int(res['seeds'].attrib['value'])
            res['leech'] = result.find(self.generate_xpath('peers'))
            res['leech'] = -1 if res['leech'] is None else int(res['leech'].attrib['value'])
            if res['seeds'] != -1 and res['leech'] != -1:
                res['leech'] -= res['seeds']
            res['desc_link'] = result.find('comments')
            if res['desc_link'] is not None:
                res['desc_link'] = res['desc_link'].text
            else:
                res['desc_link'] = result.find('guid')
                res['desc_link'] = '' if res['desc_link'] is None else res['desc_link'].text
            res['engine_url'] = self.url
            self.pretty_printer_thread_safe(res)

    def generate_xpath(self, tag):
        if False:
            return 10
        return './{http://torznab.com/schemas/2015/feed}attr[@name="%s"]' % tag

    def get_response(self, query):
        if False:
            while True:
                i = 10
        response = None
        try:
            opener = urllib_request.build_opener(urllib_request.HTTPCookieProcessor(CookieJar()))
            response = opener.open(query).read().decode('utf-8')
        except urllib_request.HTTPError as e:
            if e.code == 302:
                response = e.url
        except Exception:
            pass
        return response

    def handle_error(self, error_msg, what):
        if False:
            for i in range(10):
                print('nop')
        self.pretty_printer_thread_safe({'seeds': -1, 'size': -1, 'leech': -1, 'engine_url': self.url, 'link': self.url, 'desc_link': 'https://github.com/qbittorrent/search-plugins/wiki/How-to-configure-Jackett-plugin', 'name': "Jackett: %s! Right-click this row and select 'Open description page' to open help. Configuration file: '%s' Search: '%s'" % (error_msg, CONFIG_PATH, what)})

    def pretty_printer_thread_safe(self, dictionary):
        if False:
            for i in range(10):
                print('nop')
        global PRINTER_THREAD_LOCK
        with PRINTER_THREAD_LOCK:
            prettyPrinter(self.escape_pipe(dictionary))

    def escape_pipe(self, dictionary):
        if False:
            i = 10
            return i + 15
        for key in dictionary.keys():
            if isinstance(dictionary[key], str):
                dictionary[key] = dictionary[key].replace('|', '%7C')
        return dictionary
if __name__ == '__main__':
    jackett_se = jackett()
    jackett_se.search('ubuntu server', 'software')