from robot.libraries.BuiltIn import BuiltIn, register_run_keyword

class RegisteredClass:

    def run_keyword_if_method(self, expression, name, *args):
        if False:
            print('Hello World!')
        return BuiltIn().run_keyword_if(expression, name, *args)

    def run_keyword_method(self, name, *args):
        if False:
            for i in range(10):
                print('nop')
        return BuiltIn().run_keyword(name, *args)
register_run_keyword('RegisteredClass', 'Run Keyword If Method', 2)
register_run_keyword('RegisteredClass', 'run_keyword_method', 1)