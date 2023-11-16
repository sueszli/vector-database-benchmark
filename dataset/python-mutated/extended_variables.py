class ExampleObject:

    def __init__(self, name='<noname>'):
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def greet(self, name=None):
        if False:
            while True:
                i = 10
        if not name:
            return '%s says hi!' % self.name
        if name == 'FAIL':
            raise ValueError
        return '%s says hi to %s!' % (self.name, name)

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.name

    def __repr__(self):
        if False:
            while True:
                i = 10
        return "'%s'" % self.name
OBJ = ExampleObject('dude')