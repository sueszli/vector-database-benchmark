from robot.libraries.BuiltIn import BuiltIn, register_run_keyword

class DynamicRegisteredLibrary:

    def get_keyword_names(self):
        if False:
            while True:
                i = 10
        return ['dynamic_run_keyword']

    def run_keyword(self, name, args):
        if False:
            return 10
        dynamic_run_keyword(*args)

def dynamic_run_keyword(name, *args):
    if False:
        for i in range(10):
            print('nop')
    return BuiltIn().run_keyword(name, *args)
register_run_keyword('DynamicRegisteredLibrary', 'dynamic_run_keyword', 1)