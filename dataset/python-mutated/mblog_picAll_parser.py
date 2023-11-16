from .parser import Parser
from .util import handle_html

class MblogPicAllParser(Parser):

    def __init__(self, cookie, weibo_id):
        if False:
            while True:
                i = 10
        self.cookie = cookie
        self.url = 'https://weibo.cn/mblog/picAll/' + weibo_id + '?rl=1'
        self.selector = handle_html(self.cookie, self.url)

    def extract_preview_picture_list(self):
        if False:
            print('Hello World!')
        return self.selector.xpath('//img/@src')