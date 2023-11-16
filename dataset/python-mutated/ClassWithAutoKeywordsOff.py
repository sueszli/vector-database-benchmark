from robot.api.deco import keyword

class ClassWithAutoKeywordsOff:
    ROBOT_AUTO_KEYWORDS = False

    def public_method_is_not_keyword(self):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('Should not be executed!')

    @keyword(name='Decorated Method Is Keyword')
    def decorated_method(self):
        if False:
            return 10
        print('Decorated methods are keywords.')

    def _private_method_is_not_keyword(self):
        if False:
            return 10
        raise RuntimeError('Should not be executed!')

    @keyword
    def _private_decorated_method_is_keyword(self):
        if False:
            print('Hello World!')
        print('Decorated private methods are keywords.')