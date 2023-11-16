import scrapy

class TourismBeachSpider0(scrapy.Spider):
    name = 'beach_0'
    start_urls = ['http://www.classification-tourism.ru/index.php/displayBeach/index']

    def parse(self, response):
        if False:
            for i in range(10):
                print('nop')
        for obj in response.css('a.field.object-title'):
            yield {'title': obj.css('::text').extract_first()}