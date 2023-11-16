class ArgumentsPython:

    def a_0(self):
        if False:
            for i in range(10):
                print('nop')
        '(0,0)'
        return 'a_0'

    def a_1(self, arg):
        if False:
            for i in range(10):
                print('nop')
        '(1,1)'
        return 'a_1: ' + arg

    def a_3(self, arg1, arg2, arg3):
        if False:
            return 10
        '(3,3)'
        return ' '.join(['a_3:', arg1, arg2, arg3])

    def a_0_1(self, arg='default'):
        if False:
            while True:
                i = 10
        '(0,1)'
        return 'a_0_1: ' + arg

    def a_1_3(self, arg1, arg2='default', arg3='default'):
        if False:
            i = 10
            return i + 15
        '(1,3)'
        return ' '.join(['a_1_3:', arg1, arg2, arg3])

    def a_0_n(self, *args):
        if False:
            for i in range(10):
                print('nop')
        '(0,sys.maxsize)'
        return ' '.join(['a_0_n:', ' '.join(args)])

    def a_1_n(self, arg, *args):
        if False:
            i = 10
            return i + 15
        '(1,sys.maxsize)'
        return ' '.join(['a_1_n:', arg, ' '.join(args)])

    def a_1_2_n(self, arg1, arg2='default', *args):
        if False:
            for i in range(10):
                print('nop')
        '(1,sys.maxsize)'
        return ' '.join(['a_1_2_n:', arg1, arg2, ' '.join(args)])