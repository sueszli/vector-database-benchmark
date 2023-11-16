class VflLinearRegression(JavaValue):

    def __init__(self, jvalue, *args):
        if False:
            print('Hello World!')
        bigdl_type = 'float'
        super(JavaValue, self).__init__(jvalue, bigdl_type, *args)

    def fit(self, x, y, epochs):
        if False:
            print('Hello World!')
        pass

    def evaluate(self, x, y):
        if False:
            print('Hello World!')
        pass

    def predict(self, x):
        if False:
            return 10
        pass