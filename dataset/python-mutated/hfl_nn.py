from bigdl.dllib.utils.common import JavaValue

class HflNN(JavaValue):
    """
    HFL NN class, users could build custom NN structure in this class
    """

    def __init__(self, jvalue, *args):
        if False:
            print('Hello World!')
        bigdl_type = 'float'
        super(JavaValue, self).__init__(jvalue, bigdl_type, *args)

    def fit(self, x, y, epochs):
        if False:
            print('Hello World!')
        '\n        :param x: data, could be Numpy NdArray or Pandas DataFrame\n        :param y: label, could be Numpy NdArray or Pandas DataFrame\n        :param epochs: training epochs\n        :return:\n        '
        pass

    def evaluate(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param x: data, could be Numpy NdArray or Pandas DataFrame\n        :param y: label, could be Numpy NdArray or Pandas DataFrame\n        :return:\n        '
        pass

    def predict(self, x):
        if False:
            while True:
                i = 10
        '\n        :param x: data, could be Numpy NdArray or Pandas DataFrame\n        :return:\n        '
        pass