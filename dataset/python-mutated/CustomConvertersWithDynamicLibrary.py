from CustomConverters import Number, string_to_int

class CustomConvertersWithDynamicLibrary:
    ROBOT_LIBRARY_CONVERTERS = {Number: string_to_int}

    def get_keyword_names(self):
        if False:
            i = 10
            return i + 15
        return ['dynamic keyword']

    def run_keyword(self, name, args, named):
        if False:
            i = 10
            return i + 15
        self._validate(*args, **named)

    def _validate(self, argument, expected):
        if False:
            print('Hello World!')
        assert argument == expected

    def get_keyword_arguments(self, name):
        if False:
            while True:
                i = 10
        return ['argument', 'expected']

    def get_keyword_types(self, name):
        if False:
            return 10
        return [Number, int]