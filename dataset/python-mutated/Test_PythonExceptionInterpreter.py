from AlgorithmImports import *

class Test_PythonExceptionInterpreter(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def key_error(self):
        if False:
            for i in range(10):
                print('nop')
        x = dict()['SPY']

    def no_method_match(self):
        if False:
            i = 10
            return i + 15
        self.SetCash('SPY')

    def unsupported_operand(self):
        if False:
            i = 10
            return i + 15
        x = None + 'Pepe Grillo'

    def zero_division_error(self):
        if False:
            i = 10
            return i + 15
        x = 1 / 0

    def dotnet_error(self):
        if False:
            print('Hello World!')
        self.MarketOrder(None, 1)