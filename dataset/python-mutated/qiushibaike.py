import gevent.monkey
gevent.monkey.patch_all()
from gevent.pool import Pool
from queue import Queue
import requests
from lxml import etree

class QiushiSpider:

    def __init__(self, max_page):
        if False:
            i = 10
            return i + 15
        self.max_page = max_page
        self.pool = Pool(5)
        self.base_url = 'http://www.qiushibaike.com/8hr/page/{}/'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
        self.url_queue = Queue()
        pass

    def get_url_list(self):
        if False:
            i = 10
            return i + 15
        '\n        获取 url 列表放入到 url 容器中\n        :return:\n        '
        for page in range(1, self.max_page, 1):
            url = self.base_url.format(page)
            self.url_queue.put(url)

    def exec_task(self):
        if False:
            while True:
                i = 10
        url = self.url_queue.get()
        response = requests.get(url, headers=self.headers)
        html = response.text
        eroot = etree.HTML(html)
        titles = eroot.xpath('//a[@class="recmd-content"]/text()')
        for title in titles:
            item = {}
            item['title'] = title
            print(item)
        self.url_queue.task_done()

    def exec_task_finished(self, result):
        if False:
            i = 10
            return i + 15
        print('result:', result)
        print('执行任务完成')
        self.pool.apply_async(self.exec_task, callback=self.exec_task_finished)

    def run(self):
        if False:
            while True:
                i = 10
        self.get_url_list()
        for i in range(5):
            self.pool.apply_async(self.exec_task, callback=self.exec_task_finished)
        self.url_queue.join()
        pass
if __name__ == '__main__':
    max_page = input('请输入您需要多少页内容：')
    spider = QiushiSpider(int(max_page))
    spider.run()