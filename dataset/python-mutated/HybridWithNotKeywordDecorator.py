from robot.api.deco import not_keyword

class HybridWithNotKeywordDecorator:

    def get_keyword_names(self):
        if False:
            print('Hello World!')
        return ['exposed_in_hybrid', 'not_exposed_in_hybrid']

    def exposed_in_hybrid(self):
        if False:
            while True:
                i = 10
        pass

    @not_keyword
    def not_exposed_in_hybrid(self):
        if False:
            return 10
        pass