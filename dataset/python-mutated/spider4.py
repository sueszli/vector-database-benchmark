from scrapy.spiders import Spider

class Spider4(Spider):
    name = 'spider4'
    allowed_domains = ['spider4.com']

    @classmethod
    def handles_request(cls, request):
        if False:
            for i in range(10):
                print('nop')
        return request.url == 'http://spider4.com/onlythis'