class Mandatory:

    def __init__(self, mandatory1, mandatory2):
        if False:
            for i in range(10):
                print('nop')
        self.mandatory1 = mandatory1
        self.mandatory2 = mandatory2

    def get_args(self):
        if False:
            i = 10
            return i + 15
        return (self.mandatory1, self.mandatory2)

class Defaults:

    def __init__(self, mandatory, default1='value', default2=None):
        if False:
            print('Hello World!')
        self.mandatory = mandatory
        self.default1 = default1
        self.default2 = default2

    def get_args(self):
        if False:
            i = 10
            return i + 15
        return (self.mandatory, self.default1, self.default2)

class Varargs(Mandatory):

    def __init__(self, mandatory, *varargs):
        if False:
            i = 10
            return i + 15
        Mandatory.__init__(self, mandatory, ' '.join((str(a) for a in varargs)))

class Mixed(Defaults):

    def __init__(self, mandatory, default=42, *extra):
        if False:
            while True:
                i = 10
        Defaults.__init__(self, mandatory, default, ' '.join((str(a) for a in extra)))