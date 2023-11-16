from .baseInfo import BaseInfo

class PageInfo(BaseInfo):

    def __init__(self, title, url, content_length, status_code):
        if False:
            i = 10
            return i + 15
        self.title = title
        self.url = url
        self.content_length = content_length
        self.status_code = status_code

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, PageInfo):
            if self.url == other.url:
                return True

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.url)

    def _dump_json(self):
        if False:
            print('Hello World!')
        item = {'title': self.title, 'url': self.url, 'content_length': self.content_length, 'status_code': self.status_code}
        return item