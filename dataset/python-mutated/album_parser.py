from .parser import Parser
from .util import handle_html

class AlbumParser(Parser):

    def __init__(self, cookie, album_url):
        if False:
            for i in range(10):
                print('nop')
        self.cookie = cookie
        self.url = album_url
        self.selector = handle_html(self.cookie, self.url)

    def extract_pic_urls(self):
        if False:
            i = 10
            return i + 15
        pic_list = self.selector.xpath('//div[@class="c"]//img/@src')
        for (i, pic) in enumerate(pic_list):
            if '?' in pic:
                pic = pic[:pic.index('?')]
            pic_list[i] = pic
        return pic_list