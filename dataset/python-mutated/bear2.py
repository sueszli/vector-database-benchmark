from bear1 import TestBear as ImportedTestBear

class SubTestBear(ImportedTestBear):

    def __init__(self):
        if False:
            return 10
        ImportedTestBear.__init__(self)

    @staticmethod
    def kind():
        if False:
            print('Hello World!')
        return 'kind'

    def origin(self):
        if False:
            print('Hello World!')
        return __file__