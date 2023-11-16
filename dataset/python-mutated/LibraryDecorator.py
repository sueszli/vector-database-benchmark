from robot.api.deco import keyword, library

@library
class LibraryDecorator:

    def not_keyword(self):
        if False:
            return 10
        raise RuntimeError('Should not be executed!')

    @keyword
    def decorated_method_is_keyword(self):
        if False:
            i = 10
            return i + 15
        print('Decorated methods are keywords.')

    @staticmethod
    @keyword
    def decorated_static_method_is_keyword():
        if False:
            i = 10
            return i + 15
        print('Decorated static methods are keywords.')

    @classmethod
    @keyword
    def decorated_class_method_is_keyword(cls):
        if False:
            return 10
        print('Decorated class methods are keywords.')