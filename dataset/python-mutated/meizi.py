import scrapy
from scrapy.selector import Selector
from scrapy.contrib.loader import ItemLoader, Identity
from meizitu.items import MeizituItem

class MeiziSpider(scrapy.Spider):
    name = 'meizi'
    allowed_domains = ['meizitu.com']
    start_urls = ('http://www.meizitu.com/',)

    def parse(self, response):
        if False:
            i = 10
            return i + 15
        sel = Selector(response)
        for link in sel.xpath('//h2/a/@href').extract():
            request = scrapy.Request(link, callback=self.parse_item)
            yield request
        pages = sel.xpath('//*[@id="wp_page_numbers"]/ul/li/a/@href').extract()
        print('pages: %s' % pages)
        if len(pages) > 2:
            page_link = pages[-2]
            page_link = page_link.replace('/a/', '')
            request = scrapy.Request('http://www.meizitu.com/a/%s' % page_link, callback=self.parse)
            yield request

    def parse_item(self, response):
        if False:
            print('Hello World!')
        l = ItemLoader(item=MeizituItem(), response=response)
        l.add_xpath('name', '//h2/a/text()')
        l.add_xpath('tags', "//div[@id='maincontent']/div[@class='postmeta  clearfix']/div[@class='metaRight']/p")
        l.add_xpath('image_urls', "//div[@id='picture']/p/img/@src", Identity())
        l.add_value('url', response.url)
        return l.load_item()