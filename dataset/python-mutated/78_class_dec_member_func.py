@foo.bar
class C:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = 42