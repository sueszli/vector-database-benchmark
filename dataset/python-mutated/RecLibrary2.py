class RecLibrary2:

    def keyword_only_in_library_2(self):
        if False:
            for i in range(10):
                print('nop')
        print('Keyword from library 2')

    def keyword_in_both_libraries(self):
        if False:
            print('Hello World!')
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

    def no_operation(self):
        if False:
            print('Hello World!')
        print('Overrides keyword from BuiltIn library')

    def similar_kw_4(self):
        if False:
            for i in range(10):
                print('nop')
        pass