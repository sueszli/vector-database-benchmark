import getpass

class bcolors:
    PURPLE = ''
    BLUE = ''
    GREEN = ''
    YELLOW = ''
    RED = ''
    ENDC = ''

    def enable(self):
        if False:
            i = 10
            return i + 15
        self.PURPLE = '\x1b[95m'
        self.BLUE = '\x1b[94m'
        self.GREEN = '\x1b[92m'
        self.YELLOW = '\x1b[93m'
        self.RED = '\x1b[91m'
        self.ENDC = '\x1b[0m'

    def disable(self):
        if False:
            print('Hello World!')
        self.PURPLE = ''
        self.BLUE = ''
        self.GREEN = ''
        self.YELLOW = ''
        self.RED = ''
        self.ENDC = ''
b = bcolors()
b.enable()

def disable_colors():
    if False:
        while True:
            i = 10
    b.disable()

def enable_colors():
    if False:
        print('Hello World!')
    b.enable()

def green_print(*args):
    if False:
        for i in range(10):
            print('nop')
    if getpass.getuser() == 'jenkins':
        b.disable()
    for msg in args:
        print(b.GREEN + str(msg) + b.ENDC)
    print('')

def blue_print(*args):
    if False:
        while True:
            i = 10
    if getpass.getuser() == 'jenkins':
        b.disable()
    for msg in args:
        print(b.BLUE + str(msg) + b.ENDC)
    print('')

def yellow_print(*args):
    if False:
        print('Hello World!')
    if getpass.getuser() == 'jenkins':
        b.disable()
    for msg in args:
        print(b.YELLOW + str(msg) + b.ENDC)
    print('')

def red_print(*args):
    if False:
        for i in range(10):
            print('nop')
    if getpass.getuser() == 'jenkins':
        b.disable()
    for msg in args:
        print(b.RED + str(msg) + b.ENDC)
    print('')

def purple_print(*args):
    if False:
        i = 10
        return i + 15
    if getpass.getuser() == 'jenkins':
        b.disable()
    for msg in args:
        print(b.PURPLE + str(msg) + b.ENDC)
    print('')