from lxml import etree
from .parser import Parser

class Javday(Parser):
    source = 'javday'
    expr_url = '/html/head/meta[@property="og:url"]/@content'
    expr_cover = '/html/head/meta[@property="og:image"]/@content'
    expr_tags = '/html/head/meta[@name="keywords"]/@content'
    expr_title = '/html/head/title/text()'
    expr_actor = "//span[@class='vod_actor']/a/text()"
    expr_studio = '//span[@class="producer"]/a/text()'
    expr_number = '//span[@class="jpnum"]/text()'

    def extraInit(self):
        if False:
            i = 10
            return i + 15
        self.imagecut = 4
        self.uncensored = True

    def search(self, number):
        if False:
            for i in range(10):
                print('nop')
        self.number = number.strip().upper()
        if self.specifiedUrl:
            self.detailurl = self.specifiedUrl
        else:
            self.detailurl = 'https://javday.tv/videos/' + self.number.replace('-', '') + '/'
        self.htmlcode = self.getHtml(self.detailurl)
        if self.htmlcode == 404:
            return 404
        htmltree = etree.fromstring(self.htmlcode, etree.HTMLParser())
        self.detailurl = self.getTreeElement(htmltree, self.expr_url)
        result = self.dictformat(htmltree)
        return result

    def getTitle(self, htmltree):
        if False:
            return 10
        title = super().getTitle(htmltree)
        result = title.replace(self.number, '').replace('- JAVDAY.TV', '').strip()
        return result

    def getTags(self, htmltree) -> list:
        if False:
            while True:
                i = 10
        tags = super().getTags(htmltree)
        return [tag for tag in tags if 'JAVDAY.TV' not in tag]