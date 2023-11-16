from sphinx.search import SearchLanguage
from smallseg import SEG

class SearchChinese(SearchLanguage):
    lang = 'zh'

    def init(self, options):
        if False:
            return 10
        print('reading Chiniese dictionary')
        self.seg = SEG()

    def split(self, input):
        if False:
            return 10
        return self.seg.cut(input.encode('utf8'))

    def word_filter(self, stemmed_word):
        if False:
            while True:
                i = 10
        return len(stemmed_word) > 1