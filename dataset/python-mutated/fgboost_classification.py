from bigdl.dllib.utils.common import JavaValue

class FGBoostClassification(JavaValue):

    def __init__(self, jvalue, *args):
        if False:
            print('Hello World!')
        bigdl_type = 'float'
        super(JavaValue, self).__init__(jvalue, bigdl_type, *args)

    def fit(self, x, y, num_round):
        if False:
            for i in range(10):
                print('nop')
        pass

    def evaluate(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        pass

    def predict(self, x):
        if False:
            i = 10
            return i + 15
        pass