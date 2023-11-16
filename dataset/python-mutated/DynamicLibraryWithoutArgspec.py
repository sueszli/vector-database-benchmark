class DynamicLibraryWithoutArgspec:

    def get_keyword_names(self):
        if False:
            for i in range(10):
                print('nop')
        return [name for name in dir(self) if name.startswith('do_')]

    def run_keyword(self, name, args):
        if False:
            i = 10
            return i + 15
        return getattr(self, name)(*args)

    def do_something(self, x):
        if False:
            while True:
                i = 10
        print(x)

    def do_something_else(self, x, y=0):
        if False:
            for i in range(10):
                print('nop')
        print('x: %s, y: %s' % (x, y))

    def do_something_third(self, a, b=2, c=3):
        if False:
            print('Hello World!')
        print(a, b, c)