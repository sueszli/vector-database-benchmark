from bs4 import BeautifulSoup
import requests, sys
'\n类说明:下载《笔趣看》网小说《一念永恒》\nParameters:\n\t无\nReturns:\n\t无\nModify:\n\t2017-09-13\n'

class downloader(object):

    def __init__(self):
        if False:
            return 10
        self.server = 'http://www.biqukan.com/'
        self.target = 'http://www.biqukan.com/1_1094/'
        self.names = []
        self.urls = []
        self.nums = 0
    '\n\t函数说明:获取下载链接\n\tParameters:\n\t\t无\n\tReturns:\n\t\t无\n\tModify:\n\t\t2017-09-13\n\t'

    def get_download_url(self):
        if False:
            print('Hello World!')
        req = requests.get(url=self.target)
        html = req.text
        div_bf = BeautifulSoup(html)
        div = div_bf.find_all('div', class_='listmain')
        a_bf = BeautifulSoup(str(div[0]))
        a = a_bf.find_all('a')
        self.nums = len(a[15:])
        for each in a[15:]:
            self.names.append(each.string)
            self.urls.append(self.server + each.get('href'))
    '\n\t函数说明:获取章节内容\n\tParameters:\n\t\ttarget - 下载连接(string)\n\tReturns:\n\t\ttexts - 章节内容(string)\n\tModify:\n\t\t2017-09-13\n\t'

    def get_contents(self, target):
        if False:
            return 10
        req = requests.get(url=target)
        html = req.text
        bf = BeautifulSoup(html)
        texts = bf.find_all('div', class_='showtxt')
        texts = texts[0].text.replace('\xa0' * 8, '\n\n')
        return texts
    '\n\t函数说明:将爬取的文章内容写入文件\n\tParameters:\n\t\tname - 章节名称(string)\n\t\tpath - 当前路径下,小说保存名称(string)\n\t\ttext - 章节内容(string)\n\tReturns:\n\t\t无\n\tModify:\n\t\t2017-09-13\n\t'

    def writer(self, name, path, text):
        if False:
            return 10
        write_flag = True
        with open(path, 'a', encoding='utf-8') as f:
            f.write(name + '\n')
            f.writelines(text)
            f.write('\n\n')
if __name__ == '__main__':
    dl = downloader()
    dl.get_download_url()
    print('《一年永恒》开始下载：')
    for i in range(dl.nums):
        dl.writer(dl.names[i], '一念永恒.txt', dl.get_contents(dl.urls[i]))
        sys.stdout.write('  已下载:%.3f%%' % float(i / dl.nums * 100) + '\r')
        sys.stdout.flush()
    print('《一年永恒》下载完成')