class LEDClass:

    def __init__(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.id = 'LED(%d):' % id

    def value(self, v):
        if False:
            for i in range(10):
                print('nop')
        print(self.id, v)

    def on(self):
        if False:
            i = 10
            return i + 15
        self.value(1)

    def off(self):
        if False:
            for i in range(10):
                print('nop')
        self.value(0)
LED = LEDClass(1)
LED2 = LEDClass(12)