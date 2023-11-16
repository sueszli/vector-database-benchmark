class TestLibrary:

    def __init__(self, name='TestLibrary'):
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def get_name(self):
        if False:
            while True:
                i = 10
        return self.name
    get_library_name = get_name

    def no_operation(self):
        if False:
            i = 10
            return i + 15
        return self.name

def get_name_with_search_order(name):
    if False:
        i = 10
        return i + 15
    raise AssertionError('Should not be run due to search order having higher precedence.')

def get_best_match_ever_with_search_order():
    if False:
        i = 10
        return i + 15
    raise AssertionError('Should not be run due to search order having higher precedence.')