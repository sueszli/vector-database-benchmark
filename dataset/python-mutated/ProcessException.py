class ProcessException(Exception):

    def __init__(self, *args):
        if False:
            print('Hello World!')
        self.args = args

class JumpProcess(ProcessException):

    def __init__(self, index):
        if False:
            i = 10
            return i + 15
        self.index = index

class BreakProcess(ProcessException):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

class EndProcess(ProcessException):

    def __init__(self):
        if False:
            print('Hello World!')
        pass