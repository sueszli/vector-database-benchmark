from robot.api.deco import keyword
ROBOT_AUTO_KEYWORDS = False

def public_method_is_not_keyword():
    if False:
        print('Hello World!')
    raise RuntimeError('Should not be executed!')

@keyword(name='Decorated Method In Module Is Keyword')
def decorated_method():
    if False:
        for i in range(10):
            print('nop')
    print('Decorated methods are keywords.')

def _private_method_is_not_keyword():
    if False:
        while True:
            i = 10
    raise RuntimeError('Should not be executed!')

@keyword
def _private_decorated_method_in_module_is_keyword():
    if False:
        i = 10
        return i + 15
    print('Decorated private methods are keywords.')