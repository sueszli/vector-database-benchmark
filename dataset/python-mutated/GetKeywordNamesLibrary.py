from robot.api.deco import keyword

def passing_handler(*args):
    if False:
        return 10
    for arg in args:
        print(arg, end=' ')
    return ', '.join(args)

def failing_handler(*args):
    if False:
        while True:
            i = 10
    raise AssertionError('Failure: %s' % ' '.join(args) if args else 'Failure')

class GetKeywordNamesLibrary:

    def __init__(self):
        if False:
            return 10
        self.not_method_or_function = 'This is just a string!!'

    def get_keyword_names(self):
        if False:
            while True:
                i = 10
        marked = [name for name in dir(self) if hasattr(getattr(self, name), 'robot_name')]
        other = ['Get Keyword That Passes', 'Get Keyword That Fails', 'keyword_in_library_itself', '_starting_with_underscore_is_ok', 'Non-existing attribute', 'not_method_or_function', 'Unexpected error getting attribute', '__init__']
        return marked + other

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name == 'Get Keyword That Passes':
            return passing_handler
        if name == 'Get Keyword That Fails':
            return failing_handler
        if name == 'Unexpected error getting attribute':
            raise TypeError('Oooops!')
        raise AttributeError("Non-existing attribute '%s'" % name)

    def keyword_in_library_itself(self):
        if False:
            i = 10
            return i + 15
        msg = 'No need for __getattr__ here!!'
        print(msg)
        return msg

    def _starting_with_underscore_is_ok(self):
        if False:
            print('Hello World!')
        print("This is explicitly returned from 'get_keyword_names' anyway.")

    @keyword("Name set using 'robot_name' attribute")
    def name_set_in_method_signature(self):
        if False:
            print('Hello World!')
        pass

    @keyword
    def keyword_name_should_not_change(self):
        if False:
            i = 10
            return i + 15
        pass

    @keyword('Add ${count} copies of ${item} to cart')
    def add_copies_to_cart(self, count, item):
        if False:
            while True:
                i = 10
        return (count, item)