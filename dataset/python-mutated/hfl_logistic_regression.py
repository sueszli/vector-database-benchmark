from bigdl.dllib.utils.common import JavaValue

class HflLogisticRegression(JavaValue):

    def __init__(self, jvalue, *args):
        if False:
            print('Hello World!')
        bigdl_type = 'float'
        super(JavaValue, self).__init__(jvalue, bigdl_type, *args)

    def fit(self, x, y, epochs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def evaluate(self, x, y):
        if False:
            return 10
        pass

    def predict(self, x):
        if False:
            for i in range(10):
                print('nop')
        pass