from robot.api.deco import keyword, library

class Listener:
    ROBOT_LISTENER_API_VERSION = 3

@library(scope='TEST SUITE', version='1.2.3', listener=Listener())
class LibraryDecoratorWithArgs:

    def not_keyword_v2(self):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('Should not be executed!')

    @keyword(name='Decorated method is keyword v.2')
    def decorated_method_is_keyword(self):
        if False:
            print('Hello World!')
        print('Decorated methods are keywords.')