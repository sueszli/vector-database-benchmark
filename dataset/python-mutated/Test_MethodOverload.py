from AlgorithmImports import *

class Test_MethodOverload(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.AddEquity('SPY', Resolution.Second)
        self.sma = self.SMA('SPY', 20)
        self.std = self.STD('SPY', 20)
        self.a = A()

    def OnData(self, data):
        if False:
            print('Hello World!')
        pass

    def call_plot_std_test(self):
        if False:
            return 10
        self.Plot('STD', self.std)

    def call_plot_sma_test(self):
        if False:
            return 10
        self.Plot('SMA', self.sma)

    def call_plot_number_test(self):
        if False:
            while True:
                i = 10
        self.Plot('NUMBER', 0.1)

    def call_plot_throw_test(self):
        if False:
            for i in range(10):
                print('nop')
        self.Plot('ERROR', self.Name)

    def call_plot_throw_managed_test(self):
        if False:
            i = 10
            return i + 15
        self.Plot('ERROR', self.Portfolio)

    def call_plot_throw_pyobject_test(self):
        if False:
            while True:
                i = 10
        self.Plot('ERROR', self.a)

    def no_method_match(self):
        if False:
            print('Hello World!')
        self.Log(1)

class A(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass