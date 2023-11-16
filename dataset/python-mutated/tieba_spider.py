"""
info:
author:CriseLYJ
github:https://github.com/CriseLYJ/
update_time:2019-3-6
"""
'\n请求URL分析\thttps://tieba.baidu.com/f?kw=魔兽世界&ie=utf-8&pn=50\n请求方式分析\tGET\n请求参数分析\tpn每页50发生变化，其他参数固定不变\n请求头分析\t只需要添加User-Agent\n'
import requests

class TieBa_Spier:

    def __init__(self, max_pn, kw):
        if False:
            for i in range(10):
                print('nop')
        self.max_pn = max_pn
        self.kw = kw
        self.base_url = 'https://tieba.baidu.com/f?kw={}&ie=utf-8&pn={}'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}

    def get_url_list(self):
        if False:
            i = 10
            return i + 15
        '获取url列表'
        return [self.base_url.format(self.kw, pn) for pn in range(0, self.max_pn, 50)]

    def get_content(self, url):
        if False:
            return 10
        '发送请求获取响应内容'
        response = requests.get(url=url, headers=self.headers)
        return response.content

    def save_items(self, content, idx):
        if False:
            i = 10
            return i + 15
        '从响应内容中提取数据'
        with open('{}.html'.format(idx), 'wb') as f:
            f.write(content)
        return None

    def run(self):
        if False:
            while True:
                i = 10
        '运行程序'
        url_list = self.get_url_list()
        for url in url_list:
            content = self.get_content(url)
            items = self.save_items(content, url_list.index(url) + 1)
if __name__ == '__main__':
    spider = TieBa_Spier(200, '英雄联盟')
    spider.run()