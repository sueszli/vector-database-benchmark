class TraceLogArgsLibrary:

    def only_mandatory(self, mand1, mand2):
        if False:
            i = 10
            return i + 15
        pass

    def mandatory_and_default(self, mand, default='default value'):
        if False:
            i = 10
            return i + 15
        pass

    def multiple_default_values(self, a=1, a2=2, a3=3, a4=4):
        if False:
            for i in range(10):
                print('nop')
        pass

    def mandatory_and_varargs(self, mand, *varargs):
        if False:
            i = 10
            return i + 15
        pass

    def kwargs(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def all_args(self, positional, *varargs, **kwargs):
        if False:
            print('Hello World!')
        pass

    def return_object_with_non_ascii_repr(self):
        if False:
            while True:
                i = 10

        class NonAsciiRepr:

            def __repr__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'Hyvä'
        return NonAsciiRepr()

    def return_object_with_invalid_repr(self):
        if False:
            for i in range(10):
                print('nop')

        class InvalidRepr:

            def __repr__(self):
                if False:
                    return 10
                raise ValueError
        return InvalidRepr()

    def embedded_arguments(self, *args):
        if False:
            i = 10
            return i + 15
        assert args == ('bar', 'Embedded Arguments')
    embedded_arguments.robot_name = 'Embedded Arguments "${a}" and "${b}"'