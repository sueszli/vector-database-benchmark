from robot.libraries.BuiltIn import BuiltIn, register_run_keyword

class MyLibrary2:

    def keyword_only_in_library_2(self):
        if False:
            print('Hello World!')
        print('Keyword from library 2')

    def keyword_in_both_libraries(self):
        if False:
            i = 10
            return i + 15
        print('Keyword from library 2')

    def keyword_in_all_resources_and_libraries(self):
        if False:
            for i in range(10):
                print('nop')
        print('Keyword from library 2')

    def keyword_everywhere(self):
        if False:
            return 10
        print('Keyword from library 2')

    def keyword_in_tc_file_overrides_others(self):
        if False:
            i = 10
            return i + 15
        raise Exception('This keyword should not be called')

    def keyword_in_resource_overrides_libraries(self):
        if False:
            while True:
                i = 10
        raise Exception('This keyword should not be called')

    def no_operation(self):
        if False:
            return 10
        print('Overrides keyword from BuiltIn library')

    def replace_string(self):
        if False:
            i = 10
            return i + 15
        print('Overrides keyword from String library')
        return 'I replace nothing!'

    def run_keyword_if(self, expression, name, *args):
        if False:
            return 10
        return BuiltIn().run_keyword_if(expression, name, *args)
register_run_keyword('MyLibrary2', 'run_keyword_if', 2)