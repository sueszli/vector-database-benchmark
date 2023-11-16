from scrapy.spider import BaseSpider
from scrapy.selector import Selector
from Wechatproject.items import WechatprojectItem
from bs4 import BeautifulSoup
from scrapy.http import Request

class WechatSpider(BaseSpider):
    """微信搜索程序"""
    name = 'wechat'
    start_urls = []
    querystring = u'清华'
    type = 2
    for i in range(1, 50, 1):
        start_urls.append('http://weixin.sogou.com/weixin?type=%d&query=%s&page=%d' % (type, querystring, i))

    def parse(self, response):
        if False:
            for i in range(10):
                print('nop')
        sel = Selector(response)
        sites = sel.xpath('//div[@class="txt-box"]/h4/a')
        for site in sites:
            item = WechatprojectItem()
            item['title'] = site.xpath('text()').extract()
            item['link'] = site.xpath('@href').extract()
            next_url = item['link'][0]
            yield Request(url=next_url, meta={'item': item}, callback=self.parse2)

    def parse(self, response):
        if False:
            while True:
                i = 10
        soup = BeautifulSoup(response.body)
        tags = soup.findAll('h4')
        for tag in tags:
            item = WechatprojectItem()
            item['title'] = tag.text
            item['link'] = tag.find('a').get('href')
            next_url = item['link']
            yield Request(url=next_url, meta={'item': item}, callback=self.parse2)

    def parse2(self, response):
        if False:
            while True:
                i = 10
        soup = BeautifulSoup(response.body)
        tag = soup.find('div', attrs={'class': 'rich_media_content', 'id': 'js_content'})
        content_list = [tag_i.text for tag_i in tag.findAll('p')]
        content = ''.join(content_list)
        item = response.meta['item']
        item['content'] = content
        return item