import re
from lxml import etree
from .parser import Parser
from datetime import datetime

class Pissplay(Parser):
    source = 'pissplay'
    expr_number = '//*[@id="video_title"]/text()'
    expr_title = '//*[@id="video_title"]/text()'
    expr_cover = '/html/head//meta[@property="og:image"]/@content'
    expr_tags = '//div[@id="video_tags"]/a/text()'
    expr_release = '//div[@class="video_date"]/text()'
    expr_outline = '//*[@id="video_description"]/p//text()'

    def extraInit(self):
        if False:
            for i in range(10):
                print('nop')
        self.imagecut = 0
        self.specifiedSource = None

    def search(self, number):
        if False:
            return 10
        self.number = number.strip().upper()
        if self.specifiedUrl:
            self.detailurl = self.specifiedUrl
        else:
            newName = re.sub('[^a-zA-Z0-9 ]', '', number)
            self.detailurl = 'https://pissplay.com/videos/' + newName.lower().replace(' ', '-') + '/'
        self.htmlcode = self.getHtml(self.detailurl)
        if self.htmlcode == 404:
            return 404
        htmltree = etree.fromstring(self.htmlcode, etree.HTMLParser())
        result = self.dictformat(htmltree)
        return result

    def getNum(self, htmltree):
        if False:
            for i in range(10):
                print('nop')
        title = self.getTitle(htmltree)
        return title

    def getTitle(self, htmltree):
        if False:
            return 10
        title = super().getTitle(htmltree)
        title = re.sub('[^a-zA-Z0-9 ]', '', title)
        return title

    def getCover(self, htmltree):
        if False:
            i = 10
            return i + 15
        url = super().getCover(htmltree)
        if not url.startswith('http'):
            url = 'https:' + url
        return url

    def getRelease(self, htmltree):
        if False:
            print('Hello World!')
        releaseDate = super().getRelease(htmltree)
        isoData = datetime.strptime(releaseDate, '%d %b %Y').strftime('%Y-%m-%d')
        return isoData

    def getStudio(self, htmltree):
        if False:
            for i in range(10):
                print('nop')
        return 'PissPlay'

    def getTags(self, htmltree):
        if False:
            i = 10
            return i + 15
        tags = self.getTreeAll(htmltree, self.expr_tags)
        if 'Guests' in tags:
            if tags[0] == 'Collaboration' or tags[0] == 'Toilet for a Day' or tags[0] == 'Collaboration':
                del tags[1]
            else:
                tags = tags[1:]
        return tags

    def getActors(self, htmltree) -> list:
        if False:
            while True:
                i = 10
        tags = self.getTreeAll(htmltree, self.expr_tags)
        if 'Guests' in tags:
            if tags[0] == 'Collaboration' or tags[0] == 'Toilet for a Day' or tags[0] == 'Collaboration':
                return [tags[1]]
            else:
                return [tags[0]]
        else:
            return ['Bruce and Morgan']

    def getOutline(self, htmltree):
        if False:
            while True:
                i = 10
        outline = self.getTreeAll(htmltree, self.expr_outline)
        if '– Morgan xx' in outline:
            num = outline.index('– Morgan xx')
            outline = outline[:num]
        rstring = ''.join(outline).replace('&', 'and')
        return rstring