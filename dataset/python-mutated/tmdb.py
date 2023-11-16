from .parser import Parser

class Tmdb(Parser):
    """
    两种实现,带apikey与不带key
    apikey
    """
    source = 'tmdb'
    imagecut = 0
    apikey = None
    expr_title = '//head/meta[@property="og:title"]/@content'
    expr_release = '//div/span[@class="release"]/text()'
    expr_cover = '//head/meta[@property="og:image"]/@content'
    expr_outline = '//head/meta[@property="og:description"]/@content'

    def queryNumberUrl(self, number):
        if False:
            print('Hello World!')
        '\n        TODO 区分 ID 与 名称\n        '
        id = number
        movieUrl = 'https://www.themoviedb.org/movie/' + id + '?language=zh-CN'
        return movieUrl

    def getCover(self, htmltree):
        if False:
            return 10
        return 'https://www.themoviedb.org' + self.getTreeElement(htmltree, self.expr_cover)