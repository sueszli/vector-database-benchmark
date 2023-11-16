from pyspider.libs.base_handler import *

class Handler(BaseHandler):

    @not_send_status
    def not_send_status(self, response):
        if False:
            i = 10
            return i + 15
        self.crawl('http://www.baidu.com/')
        return response.text

    def url_deduplicated(self, response):
        if False:
            while True:
                i = 10
        self.crawl('http://www.baidu.com/')
        self.crawl('http://www.google.com/')
        self.crawl('http://www.baidu.com/')
        self.crawl('http://www.google.com/')
        self.crawl('http://www.google.com/')

    @catch_status_code_error
    def catch_http_error(self, response):
        if False:
            print('Hello World!')
        self.crawl('http://www.baidu.com/')
        return response.status_code

    def json(self, response):
        if False:
            for i in range(10):
                print('nop')
        return response.json

    def html(self, response):
        if False:
            print('Hello World!')
        return response.doc('h1').text()

    def links(self, response):
        if False:
            return 10
        self.crawl([x.attr.href for x in response.doc('a').items()], callback=self.links)

    def cookies(self, response):
        if False:
            i = 10
            return i + 15
        return response.cookies

    def get_save(self, response):
        if False:
            print('Hello World!')
        return response.save

    def get_process_save(self, response):
        if False:
            i = 10
            return i + 15
        return self.save

    def set_process_save(self, response):
        if False:
            return 10
        self.save['roy'] = 'binux'

class IgnoreHandler(BaseHandler):
    pass
__handler_cls__ = Handler