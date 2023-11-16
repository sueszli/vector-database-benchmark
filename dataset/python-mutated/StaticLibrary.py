from robot.libraries.BuiltIn import BuiltIn

class StaticLibrary:

    def add_static_keyword(self, name):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            'This doc for static'
            return x
        setattr(self, name, f)
        BuiltIn().reload_library(self)