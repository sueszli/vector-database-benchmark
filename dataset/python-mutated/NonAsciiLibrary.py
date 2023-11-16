MESSAGES = ['Circle is 360°', 'Hyvää üötä', 'উৄ ৰ ৺ ট ৫ ৪ হ']

class NonAsciiLibrary:

    def print_non_ascii_strings(self):
        if False:
            for i in range(10):
                print('nop')
        'Prints message containing non-ASCII characters'
        for msg in MESSAGES:
            print('*INFO*' + msg)

    def print_and_return_non_ascii_object(self):
        if False:
            return 10
        'Prints object with non-ASCII `str()` and returns it.'
        obj = NonAsciiObject()
        print(obj)
        return obj

    def raise_non_ascii_error(self):
        if False:
            for i in range(10):
                print('nop')
        raise AssertionError(', '.join(MESSAGES))

class NonAsciiObject:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.message = ', '.join(MESSAGES)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.message

    def __repr__(self):
        if False:
            while True:
                i = 10
        return repr(self.message)