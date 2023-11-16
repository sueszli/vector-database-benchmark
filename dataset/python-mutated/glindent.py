"""GAM indent processing

"""

class GamIndent:
    INDENT_SPACES_PER_LEVEL = '  '

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.indent = 0

    def Reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.indent = 0

    def Increment(self):
        if False:
            print('Hello World!')
        self.indent += 1

    def Decrement(self):
        if False:
            for i in range(10):
                print('nop')
        self.indent -= 1

    def Spaces(self):
        if False:
            while True:
                i = 10
        return self.INDENT_SPACES_PER_LEVEL * self.indent

    def SpacesSub1(self):
        if False:
            for i in range(10):
                print('nop')
        return self.INDENT_SPACES_PER_LEVEL * (self.indent - 1)

    def MultiLineText(self, message, n=0):
        if False:
            print('Hello World!')
        return message.replace('\n', f'\n{self.INDENT_SPACES_PER_LEVEL * (self.indent + n)}').rstrip()