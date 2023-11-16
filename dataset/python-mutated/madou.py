import re
from lxml import etree
from urllib.parse import urlparse, unquote
from .parser import Parser
NUM_RULES3 = ['(mmz{2,4})-?(\\d{2,})(-ep\\d*|-\\d*)?.*', '(msd)-?(\\d{2,})(-ep\\d*|-\\d*)?.*', '(yk)-?(\\d{2,})(-ep\\d*|-\\d*)?.*', '(pm)-?(\\d{2,})(-ep\\d*|-\\d*)?.*', '(mky-[a-z]{2,2})-?(\\d{2,})(-ep\\d*|-\\d*)?.*']

def change_number(number):
    if False:
        i = 10
        return i + 15
    number = number.lower().strip()
    m = re.search('(md[a-z]{0,2})-?(\\d{2,})(-ep\\d*|-\\d*)?.*', number, re.I)
    if m:
        return f"{m.group(1)}{m.group(2).zfill(4)}{m.group(3) or ''}"
    for rules in NUM_RULES3:
        m = re.search(rules, number, re.I)
        if m:
            return f"{m.group(1)}{m.group(2).zfill(3)}{m.group(3) or ''}"
    return number

class Madou(Parser):
    source = 'madou'
    expr_url = '//a[@class="share-weixin"]/@data-url'
    expr_title = '/html/head/title/text()'
    expr_studio = '//a[@rel="category tag"]/text()'
    expr_tags = '/html/head/meta[@name="keywords"]/@content'

    def extraInit(self):
        if False:
            i = 10
            return i + 15
        self.imagecut = 4
        self.uncensored = True
        self.allow_number_change = True

    def search(self, number):
        if False:
            print('Hello World!')
        self.number = change_number(number)
        if self.specifiedUrl:
            self.detailurl = self.specifiedUrl
        else:
            self.detailurl = 'https://madou.club/' + number + '.html'
        self.htmlcode = self.getHtml(self.detailurl)
        if self.htmlcode == 404:
            return 404
        htmltree = etree.fromstring(self.htmlcode, etree.HTMLParser())
        self.detailurl = self.getTreeElement(htmltree, self.expr_url)
        result = self.dictformat(htmltree)
        return result

    def getNum(self, htmltree):
        if False:
            while True:
                i = 10
        try:
            filename = unquote(urlparse(self.detailurl).path)
            result = filename[1:-5].upper().strip()
            if result.upper() != self.number.upper():
                result = re.split('[^\\x00-\\x7F]+', result, 1)[0]
            return result.strip('-')
        except:
            return ''

    def getTitle(self, htmltree):
        if False:
            for i in range(10):
                print('nop')
        browser_title = str(super().getTitle(htmltree))
        title = str(re.findall('^[A-Z0-9 /／\\-]*(.*)-麻豆社$', browser_title)[0]).strip()
        return title

    def getCover(self, htmltree):
        if False:
            i = 10
            return i + 15
        try:
            url = str(re.findall("shareimage      : '(.*?)'", self.htmlcode)[0])
            return url.strip()
        except:
            return ''

    def getTags(self, htmltree):
        if False:
            i = 10
            return i + 15
        studio = self.getStudio(htmltree)
        tags = super().getTags(htmltree)
        return [tag for tag in tags if studio not in tag and '麻豆' not in tag]