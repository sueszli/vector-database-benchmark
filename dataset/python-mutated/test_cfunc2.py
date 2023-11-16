import water.udf.CFunc2 as Func

class TestCFunc2(Func):
    """
    Compute sum of actual + predict
    """

    def apply(self, rowActual, rowPredict):
        if False:
            i = 10
            return i + 15
        return sum(rowActual.readDoubles()) + sum(rowPredict.readDoubles())