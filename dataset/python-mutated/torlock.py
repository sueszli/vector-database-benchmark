from re import compile as re_compile
from html.parser import HTMLParser
from novaprinter import prettyPrinter
from helpers import retrieve_url, download_file

class torlock(object):
    url = 'https://www.torlock2.com'
    name = 'TorLock'
    supported_categories = {'all': 'all', 'anime': 'anime', 'software': 'software', 'games': 'game', 'movies': 'movie', 'music': 'music', 'tv': 'television', 'books': 'ebooks'}

    def download_torrent(self, info):
        if False:
            print('Hello World!')
        print(download_file(info))

    class MyHtmlParser(HTMLParser):
        """ Sub-class for parsing results """

        def __init__(self, url):
            if False:
                for i in range(10):
                    print('nop')
            HTMLParser.__init__(self)
            self.url = url
            self.article_found = False
            self.item_found = False
            self.item_bad = False
            self.current_item = None
            self.item_name = None
            self.parser_class = {'ts': 'size', 'tul': 'seeds', 'tdl': 'leech'}

        def handle_starttag(self, tag, attrs):
            if False:
                while True:
                    i = 10
            params = dict(attrs)
            if self.item_found:
                if tag == 'td':
                    if 'class' in params:
                        self.item_name = self.parser_class.get(params['class'], None)
                        if self.item_name:
                            self.current_item[self.item_name] = ''
            elif self.article_found and tag == 'a':
                if 'href' in params:
                    link = params['href']
                    if link.startswith('/torrent'):
                        self.current_item['desc_link'] = ''.join((self.url, link))
                        self.current_item['link'] = ''.join((self.url, '/tor/', link.split('/')[2], '.torrent'))
                        self.current_item['engine_url'] = self.url
                        self.item_found = True
                        self.item_name = 'name'
                        self.current_item['name'] = ''
                        self.item_bad = 'rel' in params and params['rel'] == 'nofollow'
            elif tag == 'article':
                self.article_found = True
                self.current_item = {}

        def handle_data(self, data):
            if False:
                print('Hello World!')
            if self.item_name:
                self.current_item[self.item_name] += data

        def handle_endtag(self, tag):
            if False:
                i = 10
                return i + 15
            if tag == 'article':
                self.article_found = False
            elif self.item_name and (tag == 'a' or tag == 'td'):
                self.item_name = None
            elif self.item_found and tag == 'tr':
                self.item_found = False
                if not self.item_bad:
                    prettyPrinter(self.current_item)
                self.current_item = {}

    def search(self, query, cat='all'):
        if False:
            while True:
                i = 10
        ' Performs search '
        query = query.replace('%20', '-')
        parser = self.MyHtmlParser(self.url)
        page = ''.join((self.url, '/', self.supported_categories[cat], '/torrents/', query, '.html?sort=seeds&page=1'))
        html = retrieve_url(page)
        parser.feed(html)
        counter = 1
        additional_pages = re_compile('/{0}/torrents/{1}.html\\?sort=seeds&page=[0-9]+'.format(self.supported_categories[cat], query))
        list_searches = additional_pages.findall(html)[:-1]
        for page in map(lambda link: ''.join((self.url, link)), list_searches):
            html = retrieve_url(page)
            parser.feed(html)
            counter += 1
            if counter > 3:
                break
        parser.close()