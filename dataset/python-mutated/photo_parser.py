from .parser import Parser
from .util import handle_html

class PhotoParser(Parser):

    def __init__(self, cookie, user_id):
        if False:
            i = 10
            return i + 15
        self.cookie = cookie
        self.url = 'https://weibo.cn/' + str(user_id) + '/photo?tf=6_008'
        self.selector = handle_html(self.cookie, self.url)
        self.user_id = user_id

    def extract_avatar_album_url(self):
        if False:
            print('Hello World!')
        result = self.selector.xpath('//img[@alt="头像相册"]/../@href')
        if len(result) > 0:
            return 'https://weibo.cn' + result[0]
        else:
            return 'https://weibo.cn/' + str(self.user_id) + '/avatar?rl=0'