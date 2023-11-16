from pyspider.libs.base_handler import *

class Handler(BaseHandler):
    crawl_config = {}

    @every(minutes=24 * 60)
    def on_start(self):
        if False:
            while True:
                i = 10
        self.crawl('__START_URL__', callback=self.index_page)

    @config(age=10 * 24 * 60 * 60)
    def index_page(self, response):
        if False:
            print('Hello World!')
        for each in response.doc('a[href^="http"]').items():
            self.crawl(each.attr.href, callback=self.detail_page)

    @config(priority=2)
    def detail_page(self, response):
        if False:
            i = 10
            return i + 15
        return {'url': response.url, 'title': response.doc('title').text()}