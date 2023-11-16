from robot.api.deco import not_keyword

class ClassWithNotKeywordDecorator:

    def exposed_in_class(self):
        if False:
            i = 10
            return i + 15
        pass

    @not_keyword
    def not_exposed_in_class(self):
        if False:
            return 10
        pass