from AlgorithmImports import *

class GetParameterRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.CheckParameter(None, self.GetParameter('non-existing'), 'GetParameter("non-existing")')
        self.CheckParameter('100', self.GetParameter('non-existing', '100'), 'GetParameter("non-existing", "100")')
        self.CheckParameter(100, self.GetParameter('non-existing', 100), 'GetParameter("non-existing", 100)')
        self.CheckParameter(100.0, self.GetParameter('non-existing', 100.0), 'GetParameter("non-existing", 100.0)')
        self.CheckParameter('10', self.GetParameter('ema-fast'), 'GetParameter("ema-fast")')
        self.CheckParameter(10, self.GetParameter('ema-fast', 100), 'GetParameter("ema-fast", 100)')
        self.CheckParameter(10.0, self.GetParameter('ema-fast', 100.0), 'GetParameter("ema-fast", 100.0)')
        self.Quit()

    def CheckParameter(self, expected, actual, call):
        if False:
            return 10
        if expected == None and actual != None:
            raise Exception(f'{call} should have returned null but returned {actual} ({type(actual)})')
        if expected != None and actual == None:
            raise Exception(f'{call} should have returned {expected} ({type(expected)}) but returned null')
        if expected != None and actual != None and (type(expected) != type(actual)) or expected != actual:
            raise Exception(f'{call} should have returned {expected} ({type(expected)}) but returned {actual} ({type(actual)})')