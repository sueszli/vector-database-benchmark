import os
print(eval('1+1'))
print(eval('os.getcwd()'))

class Class(object):

    def eval(self):
        if False:
            return 10
        print('hi')

    def foo(self):
        if False:
            return 10
        self.eval()