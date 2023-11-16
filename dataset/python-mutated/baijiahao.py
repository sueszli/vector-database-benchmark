"""本模块是为了解决获取百家号url并且从这个url里面获取我们想要的新闻"""
import re
import time
import bs4
import requests
from selenium import webdriver

class sobaidu:
    """sobaidu类实现通过百度搜索获取真实的url并且把url写入数据库"""

    def __init__(self):
        if False:
            print('Hello World!')
        self.KEYFILENAME = 'keylist.txt'
        self.URLFILENAME = 'urllist.txt'
        self.KEYLIST = set()
        self.URLLIST = set()
        self.URLFILE = open(self.URLFILENAME, 'w')

    def _readkey(self):
        if False:
            for i in range(10):
                print('nop')
        '读取百度搜索所需要的所有关键词'
        with open(self.KEYFILENAME) as keyklistfile:
            for i in keyklistfile.readlines():
                self.KEYLIST.add(i)

    def _changeurl(self, url):
        if False:
            print('Hello World!')
        '百度搜索结果url转换为真实的url'
        try:
            req = requests.get(url + '&wd=')
            regx = 'http[s]*://baijiahao.baidu.com/[\\S]*id=[0-9]*'
            pattern = re.compile(regx)
            match = re.findall(pattern, req.text)
            return match[0]
        except Exception as e:
            print(e)

    def _writetomysql(self):
        if False:
            while True:
                i = 10
        '将真实url写入数据库'
        pass

    def _writetofile(self, url):
        if False:
            print('Hello World!')
        self.URLFILE.write(url)
        self.URLFILE.write('\n')

    def sobaidu(self):
        if False:
            i = 10
            return i + 15
        '调用以上函数解决我们的问题'
        browser = webdriver.PhantomJS()
        num = 0
        for key in self.KEYLIST:
            "'doc"
            num += 1
            now_num = 0
            browser.implicitly_wait(30)
            browser.get('https://www.baidu.com/s?wd=site:(baijiahao.baidu.com)' + key)
            while True:
                if now_num == 1:
                    try:
                        browser.find_element_by_xpath('//*[@id="page"]/a[10]').click()
                        time.sleep(2)
                    except Exception as e:
                        print(e)
                        print('有问题')
                        break
                now_num += 1
                print(now_num)
                source = browser.page_source
                soup = bs4.BeautifulSoup(source, 'lxml')
                print('next_page')
                for i in soup.findAll(class_='result c-container '):
                    url = i.find(class_='t').find('a').get('href')
                    self._writetofile(self._changeurl(url))
                time.sleep(1)
                if now_num > 1:
                    try:
                        browser.find_element_by_xpath('//*[@id="page"]/a[11]').click()
                        time.sleep(1)
                    except:
                        print('not find next_button may be for the page end!!!')
                        break

class getappid:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.URLFILENAME = 'urllist.txt'
        self.APPIDLIST = 'appid.txt'
        self.URLLIST = set()
        self.APPIDFILE = open(self.APPIDLIST, 'w')

    def _readurl(self):
        if False:
            while True:
                i = 10
        '读取新闻页的url'
        with open(self.URLFILENAME) as urllistfile:
            for i in urllistfile.readlines():
                self.URLLIST.add(i)

    def _writeappid(self, appid):
        if False:
            for i in range(10):
                print('nop')
        self.APPIDFILE.write(appid)
        self.APPIDFILE.write('\n')
        print('写入成功')

    def getid(self):
        if False:
            for i in range(10):
                print('nop')
        browser = webdriver.Chrome()
        browser.implicitly_wait(10)
        for url in self.URLLIST:
            browser.get(url)
            regx = 'http[s]*://baijiahao.baidu.com/u[\\S]*id=[0-9]*'
            pattern = re.compile(regx)
            match = re.findall(pattern, browser.page_source)
            time.sleep(1)
            try:
                print(match[0])
                self._writeappid(match[0])
            except Exception as e:
                print('匹配失败')

def main():
    if False:
        return 10
    dsfsd = sobaidu()
    dsfsd._readkey()
    print(len(dsfsd.KEYLIST))
    dsfsd.sobaidu()
    dsfsd.URLFILE.close()

def getid():
    if False:
        for i in range(10):
            print('nop')
    dsfsd = getappid()
    dsfsd._readurl()
    print(len(dsfsd.URLLIST))
    dsfsd.getid()
    dsfsd.APPIDFILE.close()
if __name__ == '__main__':
    getid()