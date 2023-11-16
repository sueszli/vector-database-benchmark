from robot.api.deco import keyword, library

@library(scope='global', auto_keywords=True)
class LibraryDecoratorWithAutoKeywords:

    def undecorated_method_is_keyword(self):
        if False:
            i = 10
            return i + 15
        pass

    @keyword
    def decorated_method_is_keyword_as_well(self):
        if False:
            while True:
                i = 10
        pass