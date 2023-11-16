import json
import re
from .parser import Parser
from .javbus import Javbus

class Airav(Parser):
    source = 'airav'
    expr_title = '/html/head/title/text()'
    expr_number = '/html/head/title/text()'
    expr_studio = '//a[contains(@href,"?video_factory=")]/text()'
    expr_release = '//li[contains(text(),"發片日期")]/text()'
    expr_outline = "string(//div[@class='d-flex videoDataBlock']/div[@class='synopsis']/p)"
    expr_actor = '//ul[@class="videoAvstarList"]/li/a[starts-with(@href,"/idol/")]/text()'
    expr_cover = '//img[contains(@src,"/storage/big_pic/")]/@src'
    expr_tags = '//div[@class="tagBtnMargin"]/a/text()'
    expr_extrafanart = '//div[@class="mobileImgThumbnail"]/a/@href'

    def extraInit(self):
        if False:
            return 10
        self.specifiedSource = None
        self.addtion_Javbus = True

    def search(self, number):
        if False:
            print('Hello World!')
        self.number = number
        if self.specifiedUrl:
            self.detailurl = self.specifiedUrl
        else:
            self.detailurl = 'https://www.airav.wiki/api/video/barcode/' + self.number.upper() + '?lng=zh-CN'
        if self.addtion_Javbus:
            engine = Javbus()
            javbusinfo = engine.scrape(self.number, self)
            if javbusinfo == 404:
                self.javbus = {'title': ''}
            else:
                self.javbus = json.loads(javbusinfo)
        self.htmlcode = self.getHtml(self.detailurl)
        htmltree = json.loads(self.htmlcode)['result']
        result = self.dictformat(htmltree)
        return result

    def getNum(self, htmltree):
        if False:
            i = 10
            return i + 15
        result = htmltree['barcode']
        return result

    def getTitle(self, htmltree):
        if False:
            print('Hello World!')
        result = htmltree['name']
        return result

    def getStudio(self, htmltree):
        if False:
            for i in range(10):
                print('nop')
        if self.addtion_Javbus:
            result = self.javbus.get('studio')
            if isinstance(result, str) and len(result):
                return result
        return super().getStudio(htmltree)

    def getRelease(self, htmltree):
        if False:
            i = 10
            return i + 15
        if self.addtion_Javbus:
            result = self.javbus.get('release')
            if isinstance(result, str) and len(result):
                return result
        try:
            return re.search('\\d{4}-\\d{2}-\\d{2}', str(super().getRelease(htmltree))).group()
        except:
            return ''

    def getYear(self, htmltree):
        if False:
            i = 10
            return i + 15
        if self.addtion_Javbus:
            result = self.javbus.get('year')
            if isinstance(result, str) and len(result):
                return result
        release = self.getRelease(htmltree)
        return str(re.findall('\\d{4}', release)).strip(" ['']")

    def getOutline(self, htmltree):
        if False:
            print('Hello World!')
        try:
            result = htmltree['description']
        except:
            result = ''
        return result

    def getRuntime(self, htmltree):
        if False:
            while True:
                i = 10
        if self.addtion_Javbus:
            result = self.javbus.get('runtime')
            if isinstance(result, str) and len(result):
                return result
        return ''

    def getDirector(self, htmltree):
        if False:
            for i in range(10):
                print('nop')
        if self.addtion_Javbus:
            result = self.javbus.get('director')
            if isinstance(result, str) and len(result):
                return result
        return ''

    def getActors(self, htmltree):
        if False:
            i = 10
            return i + 15
        a = htmltree['actors']
        if a:
            b = []
            for i in a:
                b.append(i['name'])
        else:
            b = []
        return b

    def getCover(self, htmltree):
        if False:
            for i in range(10):
                print('nop')
        if self.addtion_Javbus:
            result = self.javbus.get('cover')
            if isinstance(result, str) and len(result):
                return result
        result = htmltree['img_url']
        if isinstance(result, str) and len(result):
            return result
        return super().getCover(htmltree)

    def getSeries(self, htmltree):
        if False:
            print('Hello World!')
        if self.addtion_Javbus:
            result = self.javbus.get('series')
            if isinstance(result, str) and len(result):
                return result
        return ''

    def getExtrafanart(self, htmltree):
        if False:
            while True:
                i = 10
        try:
            result = htmltree['images']
        except:
            result = ''
        return result

    def getTags(self, htmltree):
        if False:
            print('Hello World!')
        try:
            tag = htmltree['tags']
            tags = []
            for i in tag:
                tags.append(i['name'])
        except:
            tags = []
        return tags