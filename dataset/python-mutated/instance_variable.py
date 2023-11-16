class Foo:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.attr1 = None
        self.attr2 = None

class Bar(Foo):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.attr2 = None
        self.attr3 = None
        self.attr4 = None