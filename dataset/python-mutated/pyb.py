import time

def delay(n):
    if False:
        for i in range(10):
            print('nop')
    pass
rand_seed = 1

def rng():
    if False:
        i = 10
        return i + 15
    global rand_seed
    rand_seed = rand_seed * 653276 % 8388593
    return rand_seed

class LCD:

    def __init__(self, port):
        if False:
            while True:
                i = 10
        self.width = 128
        self.height = 32
        self.buf1 = [[0 for x in range(self.width)] for y in range(self.height)]
        self.buf2 = [[0 for x in range(self.width)] for y in range(self.height)]

    def light(self, value):
        if False:
            for i in range(10):
                print('nop')
        pass

    def fill(self, value):
        if False:
            i = 10
            return i + 15
        for y in range(self.height):
            for x in range(self.width):
                self.buf1[y][x] = self.buf2[y][x] = value

    def show(self):
        if False:
            while True:
                i = 10
        print('')
        for y in range(self.height):
            for x in range(self.width):
                self.buf1[y][x] = self.buf2[y][x]
        for y in range(self.height):
            row = ''.join(['*' if self.buf1[y][x] else ' ' for x in range(self.width)])
            print(row)

    def get(self, x, y):
        if False:
            print('Hello World!')
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.buf1[y][x]
        else:
            return 0

    def pixel(self, x, y, value):
        if False:
            for i in range(10):
                print('nop')
        if 0 <= x < self.width and 0 <= y < self.height:
            self.buf2[y][x] = value