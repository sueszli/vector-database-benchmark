import re
import scrapy
from scrapy import Selector
from cartoon.items import ComicItem

class ComicSpider(scrapy.Spider):
    name = 'comic'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.server_img = 'http://n.1whour.com/'
        self.server_link = 'http://comic.kukudm.com'
        self.allowed_domains = ['comic.kukudm.com']
        self.start_urls = ['http://comic.kukudm.com/comiclist/3/']
        self.pattern_img = re.compile('\\+"(.+)\\\'><span')

    def start_requests(self):
        if False:
            for i in range(10):
                print('nop')
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse1)

    def parse1(self, response):
        if False:
            for i in range(10):
                print('nop')
        hxs = Selector(response)
        items = []
        urls = hxs.xpath('//dd/a[1]/@href').extract()
        dir_names = hxs.xpath('//dd/a[1]/text()').extract()
        for index in range(len(urls)):
            item = ComicItem()
            item['link_url'] = self.server_link + urls[index]
            item['dir_name'] = dir_names[index]
            items.append(item)
        for item in items:
            yield scrapy.Request(url=item['link_url'], meta={'item': item}, callback=self.parse2)

    def parse2(self, response):
        if False:
            print('Hello World!')
        item = response.meta['item']
        item['link_url'] = response.url
        hxs = Selector(response)
        pre_img_url = hxs.xpath('//script/text()').extract()
        img_url = [self.server_img + re.findall(self.pattern_img, pre_img_url[0])[0]]
        item['img_url'] = img_url
        yield item
        page_num = hxs.xpath('//td[@valign="top"]/text()').re(u'共(\\d+)页')[0]
        pre_link = item['link_url'][:-5]
        for each_link in range(2, int(page_num) + 1):
            new_link = pre_link + str(each_link) + '.htm'
            yield scrapy.Request(url=new_link, meta={'item': item}, callback=self.parse3)

    def parse3(self, response):
        if False:
            for i in range(10):
                print('nop')
        item = response.meta['item']
        item['link_url'] = response.url
        hxs = Selector(response)
        pre_img_url = hxs.xpath('//script/text()').extract()
        img_url = [self.server_img + re.findall(self.pattern_img, pre_img_url[0])[0]]
        item['img_url'] = img_url
        yield item